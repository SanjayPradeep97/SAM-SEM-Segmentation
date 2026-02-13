"""
SEM Particle Analysis - Gradio Application

A professional tabbed interface for particle segmentation and analysis.
Features: File browsers, image gallery, interactive refinement, and comprehensive results.
"""

import os
import sys
import json
from pathlib import Path
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "sem_particle_analysis"))

from sem_particle_analysis import (
    SAMModel,
    ScaleDetector,
    ParticleSegmenter,
    ParticleAnalyzer,
    ResultsManager
)
from sem_particle_analysis.utils import load_image, find_images_in_folder
from visualization import (
    create_mask_overlay,
    create_particle_visualization,
    create_point_refine_visualization,
    visualize_three_masks,
    visualize_scale_verification,
    draw_zoom_inset,
    draw_manual_scale_overlay,
    draw_ocr_box_overlay,
    create_results_dataframe,
    create_summary_statistics_table
)


# ============================================================================
# Global State Management
# ============================================================================

class AppState:
    """Enhanced application state with image processing tracking."""

    def __init__(self):
        # Model components
        self.sam_model = None
        self.scale_detector = None
        self.segmenter = None
        self.results_manager = None
        self.resumed_csv_path = None  # Path to resumed CSV, if any

        # Image management
        self.image_paths = []
        self.current_index = 0
        self.processed_images = {}  # {index: {"status": "completed", "num_particles": 23}}

        # Current processing state
        self.current_image = None
        self.cropped_image = None
        self.scale_info = None
        self.masks = None
        self.scores = None
        self.selected_mask_index = None
        self.analyzer = None
        self.min_particle_size = 30  # Minimum particle size in pixels for filtering
        self.show_particle_numbers = True  # Whether to show particle numbers in visualization
        self.crop_percent = 7.0  # Percentage to crop from bottom (for scale bar removal)

        # Scale detection mode state
        self.scale_mode = "Metadata"       # "Metadata", "OCR", or "Manual"
        self.ocr_click_points = []         # [(x1,y1), (x2,y2)] for OCR box corners
        self.manual_click_points = []      # [(x1,y1), (x2,y2)] for manual endpoints

        # Refinement state
        self.click_mode = "delete"  # "delete", "add", "merge", "point_refine"
        self.pending_deletes = []  # List of particle labels to delete
        self.pending_add_points = []  # List of (x, y) click points for addition
        self.pending_add_masks = []  # List of SAM-generated masks for preview
        self.pending_merge = []  # List of particle labels to merge
        self.point_refine_particle = None  # Particle label being refined in point_refine mode
        self.point_refine_base_mask = None  # Base mask of selected particle for ROI/IoU
        self.point_refine_points = []  # List of (x, y) points for refinement
        self.point_refine_labels = []  # List of point labels (1=positive, 0=negative)
        self.point_refine_preview_mask = None  # Live preview mask from SAM
        self.point_refine_logits = None  # SAM logits for iterative refinement
        self.point_type = "positive"  # "positive" or "negative" for point_refine mode

        # Undo history
        self.undo_history = []  # Stack of previous states (labeled_mask, regions)

    def reset_image_state(self):
        """Reset processing state for new image."""
        self.current_image = None
        self.cropped_image = None
        self.scale_info = None
        self.ocr_click_points = []
        self.manual_click_points = []
        self.masks = None
        self.scores = None
        self.selected_mask_index = None
        self.analyzer = None
        self.click_mode = "delete"
        self.pending_deletes = []
        self.pending_add_points = []
        self.pending_add_masks = []
        self.pending_merge = []
        self.point_refine_particle = None
        self.point_refine_base_mask = None
        self.point_refine_points = []
        self.point_refine_labels = []
        self.point_refine_preview_mask = None
        self.point_refine_logits = None
        self.point_type = "positive"
        self.undo_history = []

    def mark_processed(self, index, num_particles):
        """Mark an image as processed."""
        self.processed_images[index] = {
            "status": "completed",
            "num_particles": num_particles
        }

    def is_processed(self, index):
        """Check if an image has been processed."""
        return index in self.processed_images

    def save_state(self):
        """Save processing state to file."""
        if not self.image_paths:
            return

        state_file = Path(self.image_paths[0]).parent / ".analysis_state.json"
        state_data = {
            "processed_images": {
                str(k): v for k, v in self.processed_images.items()
            }
        }
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)

    def save_pending_state(self):
        """Save current pending changes state to undo history (for undoing individual clicks)."""
        import copy
        self.undo_history.append({
            'mode': self.click_mode,
            'pending_deletes': self.pending_deletes.copy(),
            'pending_add_points': self.pending_add_points.copy(),
            'pending_add_masks': [mask.copy() for mask in self.pending_add_masks],  # Deep copy numpy arrays
            'pending_merge': self.pending_merge.copy(),
            'point_refine_particle': self.point_refine_particle,
            'point_refine_base_mask': self.point_refine_base_mask.copy() if self.point_refine_base_mask is not None else None,
            'point_refine_points': self.point_refine_points.copy(),
            'point_refine_labels': self.point_refine_labels.copy(),
            'point_refine_preview_mask': self.point_refine_preview_mask.copy() if self.point_refine_preview_mask is not None else None,
            'point_refine_logits': self.point_refine_logits.copy() if self.point_refine_logits is not None else None
        })

    def load_state(self):
        """Load processing state from file and validate against results CSV."""
        if not self.image_paths:
            return

        state_file = Path(self.image_paths[0]).parent / ".analysis_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state_data = json.load(f)
                loaded_processed_images = {
                    int(k): v for k, v in state_data.get("processed_images", {}).items()
                }

                # Validate against results CSV if results manager exists
                if self.results_manager is not None:
                    results_df = self.results_manager.get_results()
                    saved_filenames = set(results_df['file_name'].tolist())

                    # Only keep processed status if file actually exists in results CSV
                    validated_processed_images = {}
                    for idx, data in loaded_processed_images.items():
                        if idx < len(self.image_paths):
                            filename = os.path.basename(self.image_paths[idx])
                            if filename in saved_filenames:
                                validated_processed_images[idx] = data

                    self.processed_images = validated_processed_images
                else:
                    # No results manager yet, clear processed images
                    self.processed_images = {}

    def sync_processed_from_csv(self):
        """Cross-reference loaded images with results CSV to set checkmarks.

        Returns:
            tuple: (matched_count, unmatched_count)
                matched_count: images in both CSV and loaded images
                unmatched_count: CSV entries with no matching loaded image
        """
        if self.results_manager is None or not self.image_paths:
            return 0, 0

        results_df = self.results_manager.get_results()
        if len(results_df) == 0:
            return 0, 0

        # Build lookup: filename -> num_particles
        csv_filenames = {}
        for _, row in results_df.iterrows():
            csv_filenames[row['file_name']] = int(row['num_particles'])

        # Match loaded images against CSV
        matched = 0
        for idx, img_path in enumerate(self.image_paths):
            basename = os.path.basename(img_path)
            if basename in csv_filenames:
                self.mark_processed(idx, csv_filenames[basename])
                matched += 1

        # Count CSV entries with no matching uploaded image
        image_basenames = {os.path.basename(p) for p in self.image_paths}
        unmatched = sum(1 for fn in csv_filenames if fn not in image_basenames)

        return matched, unmatched


# Global state instance
state = AppState()


# ============================================================================
# Tab 1: Setup - SAM Initialization and File Selection
# ============================================================================

def initialize_sam(sam_path, model_type, progress=gr.Progress()):
    """Initialize SAM model from file browser selection."""
    try:
        # Handle if sam_path is a list (Gradio file picker returns list)
        if isinstance(sam_path, list):
            if len(sam_path) == 0:
                return "❌ Please select a valid SAM checkpoint file", gr.update(interactive=False)
            sam_path = sam_path[0]

        if not sam_path or not os.path.exists(sam_path):
            return "❌ Please select a valid SAM checkpoint file", gr.update(interactive=False)

        progress(0.3, desc="Loading SAM model...")
        state.sam_model = SAMModel(sam_path, model_type=model_type)
        state.segmenter = ParticleSegmenter(state.sam_model)

        progress(0.7, desc="Initializing scale detector...")
        state.scale_detector = ScaleDetector(use_gpu=False)

        progress(1.0, desc="Complete!")

        return f"✅ SAM model loaded successfully ({model_type})", gr.update(interactive=True)

    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(interactive=False)


def load_images_from_folder(file_input, progress=gr.Progress()):
    """Load images from file browser."""
    try:
        folder_for_csv = None

        # Check if files were selected via browser
        if file_input is not None and (isinstance(file_input, list) and len(file_input) > 0 or isinstance(file_input, str)):
            progress(0.3, desc="Loading selected files...")

            # If list of files, use them directly
            if isinstance(file_input, list):
                # Extract paths from File objects if needed
                state.image_paths = []
                for item in file_input:
                    # Handle both string paths and File objects
                    if isinstance(item, str):
                        path = os.path.abspath(item)
                        state.image_paths.append(path)
                    elif hasattr(item, 'name'):  # Gradio File object
                        path = os.path.abspath(item.name)
                        state.image_paths.append(path)
                    else:
                        path = os.path.abspath(str(item))
                        state.image_paths.append(path)

                state.image_paths = sorted(state.image_paths)
                # Get folder from first file for CSV storage
                folder_for_csv = os.path.dirname(state.image_paths[0])
            else:
                # Single file
                if hasattr(file_input, 'name'):
                    path = os.path.abspath(file_input.name)
                    state.image_paths = [path]
                else:
                    path = os.path.abspath(str(file_input))
                    state.image_paths = [path]
                folder_for_csv = os.path.dirname(state.image_paths[0])

            source = "selected files"

        else:
            return "❌ Please select one or more images", None

        if not state.image_paths:
            return f"❌ No images found", None

        # Validate that all image files exist and are readable
        valid_paths = []
        invalid_count = 0
        for img_path in state.image_paths:
            if os.path.exists(img_path) and os.path.isfile(img_path):
                valid_paths.append(img_path)
            else:
                print(f"⚠️  Skipping non-existent file: {img_path}")
                invalid_count += 1

        state.image_paths = valid_paths

        if not state.image_paths:
            return f"❌ No valid images found ({invalid_count} files skipped)", None

        state.current_index = 0
        state.processed_images = {}

        # Initialize results manager — reuse resumed CSV if available
        if state.resumed_csv_path is not None and state.results_manager is not None:
            # User already resumed a session CSV, keep using it
            pass
        else:
            csv_path = os.path.join(folder_for_csv, "analysis_results.csv")
            state.results_manager = ResultsManager(csv_file=csv_path)

        # Load saved state if exists
        state.load_state()

        # Cross-reference images with results CSV to set checkmarks
        matched, unmatched = state.sync_processed_from_csv()

        progress(0.7, desc="Creating gallery...")
        gallery_data = create_image_gallery()

        progress(1.0, desc="Complete!")

        status_parts = [f"✅ Loaded {len(state.image_paths)} images from {source}. Ready to process!"]
        if matched > 0:
            status_parts.append(f"{matched} already analyzed (from CSV)")
        if unmatched > 0:
            status_parts.append(f"⚠️ {unmatched} CSV entries have no matching image")

        status = " | ".join(status_parts)

        return status, gallery_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", None


def resume_session(csv_file):
    """Resume a previous analysis session from a CSV results file.

    Loads the CSV as the active results file and cross-references filenames
    with uploaded images to mark already-analyzed images with checkmarks.
    """
    try:
        if csv_file is None:
            return "❌ No CSV file selected", None, ""

        # Handle Gradio File object or string path
        if hasattr(csv_file, 'name'):
            csv_path = os.path.abspath(csv_file.name)
        else:
            csv_path = os.path.abspath(str(csv_file))

        if not os.path.exists(csv_path):
            return "❌ CSV file not found", None, ""

        # Load the CSV via ResultsManager
        state.results_manager = ResultsManager(csv_file=csv_path, auto_create=False)
        state.resumed_csv_path = csv_path

        results_df = state.results_manager.get_results()
        csv_count = len(results_df)

        if csv_count == 0:
            return "⚠️ CSV loaded but contains no results", None, ""

        # If images are already loaded, cross-reference
        if state.image_paths:
            matched, unmatched = state.sync_processed_from_csv()
            state.save_state()

            # Regenerate gallery with updated checkmarks
            gallery_data = create_image_gallery()

            # Build status
            status_parts = [f"✅ Resumed: {csv_count} results in CSV, {matched} matched to loaded images"]
            if unmatched > 0:
                status_parts.append(f"⚠️ {unmatched} images in results CSV don't have a matching image")

            load_info = f"✅ {len(state.image_paths)} images loaded, {matched} already analyzed"

            return "\n".join(status_parts), gallery_data, load_info
        else:
            # No images loaded yet — just store the CSV for when images are loaded
            return f"✅ CSV loaded with {csv_count} results. Now load your images to see checkmarks.", None, ""

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", None, ""


# ============================================================================
# Tab 2: Image Gallery - Visual Overview
# ============================================================================

def create_image_gallery():
    """Create gallery view with processing status."""
    if not state.image_paths:
        return []

    gallery_items = []
    for idx, img_path in enumerate(state.image_paths):
        # Create thumbnail
        try:
            # Verify file exists and is readable
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
                continue

            # Load image as PIL Image for Gradio Gallery
            # This ensures compatibility across Windows and macOS
            img = Image.open(img_path)

            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            filename = os.path.basename(img_path)

            # Add status indicator to filename
            if state.is_processed(idx):
                particles = state.processed_images[idx].get("num_particles", "?")
                label = f"✅ {filename} ({particles} particles)"
            else:
                label = f"⚪ {filename}"

            # Gradio Gallery expects (image, caption) tuples where image is PIL Image or path
            gallery_items.append((img, label))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return gallery_items


def select_image_from_gallery(evt: gr.SelectData):
    """Handle image selection from gallery and switch to processing tab."""
    if not state.image_paths:
        return None, "No images loaded", gr.Tabs(selected=1), "delete"

    # evt.index gives us which image was clicked
    state.current_index = evt.index
    state.reset_image_state()

    try:
        state.current_image = load_image(state.image_paths[state.current_index])
        filename = os.path.basename(state.image_paths[state.current_index])
        info = f"Image {state.current_index + 1} / {len(state.image_paths)}: {filename}"

        # Return image, info, switch to processing tab (id=2), and reset mode radio to "delete"
        return state.current_image, info, gr.Tabs(selected=2), "delete"
    except Exception as e:
        return None, f"❌ Error loading image: {str(e)}", gr.Tabs(selected=1), "delete"


# ============================================================================
# Tab 3: Processing - Scale Detection and Segmentation
# ============================================================================

def detect_scale_metadata(progress=gr.Progress()):
    """Detect scale from TIFF metadata (Metadata mode)."""
    try:
        if state.current_image is None:
            return "❌ No image loaded", None

        if state.scale_detector is None:
            return "❌ Scale detector not initialized", None

        file_path = state.image_paths[state.current_index] if state.image_paths else None

        progress(0.2, desc="Reading TIFF metadata...")

        try:
            state.scale_info = state.scale_detector.detect_scale(
                state.current_image,
                file_path=file_path,
                method='metadata'
            )
        except ValueError as e:
            return f"❌ Metadata scale detection failed: {str(e)}", None

        progress(0.5, desc="Detecting databar...")

        databar_info = state.scale_detector.detect_databar(
            state.current_image,
            state.scale_info.get('raw_metadata')
        )

        # Crop if databar found
        if databar_info.get('has_databar'):
            crop_pct = databar_info['databar_fraction'] * 100
            state.cropped_image = state.scale_detector.crop_scale_bar(
                state.current_image, crop_percent=crop_pct
            )
        else:
            state.cropped_image = state.current_image.copy()

        progress(0.8, desc="Creating verification visualization...")

        scale_viz = visualize_scale_verification(
            state.current_image, state.scale_info, databar_info
        )

        progress(1.0, desc="Complete!")

        # Build status
        conversion = state.scale_info['conversion']
        manufacturer = state.scale_info.get('manufacturer', 'unknown')
        confidence = state.scale_info.get('confidence', 'unknown')
        metadata_source = state.scale_info.get('metadata_source', 'TIFF metadata')

        status_parts = []
        if manufacturer != 'unknown':
            status_parts.append(f"✅ {manufacturer.upper()} metadata: {conversion:.4f} nm/pixel [{confidence} confidence]")
        else:
            status_parts.append(f"✅ {metadata_source}: {conversion:.4f} nm/pixel [{confidence} confidence]")

        if databar_info.get('has_databar'):
            status_parts.append(f"Databar: {databar_info['databar_height']}px cropped")
        else:
            status_parts.append("No databar detected — full image preserved")

        return " | ".join(status_parts), scale_viz

    except Exception as e:
        return f"❌ Error: {str(e)}", None


def detect_scale_ocr_in_box():
    """Run OCR scale detection inside the user-defined box (OCR mode)."""
    try:
        if state.current_image is None:
            return "❌ No image loaded", None, ""

        if state.scale_detector is None:
            return "❌ Scale detector not initialized", None, ""

        if len(state.ocr_click_points) < 2:
            return "❌ Click two corners on the image first", None, ""

        (x1, y1), (x2, y2) = state.ocr_click_points
        bx0, by0 = min(int(x1), int(x2)), min(int(y1), int(y2))
        bx1, by1 = max(int(x1), int(x2)), max(int(y1), int(y2))

        H, W = state.current_image.shape[:2]
        bx0, by0 = max(0, bx0), max(0, by0)
        bx1, by1 = min(W, bx1), min(H, by1)

        if bx1 - bx0 < 10 or by1 - by0 < 10:
            return "❌ Box too small — click two wider-spaced corners", None, ""

        # Convert box to fractional region parameters for detect_scale_bar
        cx = (bx0 + bx1) / 2 / W
        cy = (by0 + by1) / 2 / H
        rw = (bx1 - bx0) / W
        rh = (by1 - by0) / H

        try:
            result = state.scale_detector.detect_scale_bar(
                state.current_image,
                region_x=cx, region_y=cy,
                region_width=rw, region_height=rh,
                polarity='auto', threshold=200
            )
        except ValueError as e:
            # Still show the box
            viz = draw_ocr_box_overlay(state.current_image, state.ocr_click_points)
            return f"❌ OCR failed: {str(e)}", viz, f"OCR failed: {e}"

        state.scale_info = result

        # Detect databar for cropping
        databar_info = state.scale_detector.detect_databar(state.current_image)
        if databar_info.get('has_databar'):
            crop_pct = databar_info['databar_fraction'] * 100
            state.cropped_image = state.scale_detector.crop_scale_bar(
                state.current_image, crop_percent=crop_pct
            )
        else:
            state.cropped_image = state.current_image.copy()

        # Draw overlay with OCR results
        ocr_result_for_viz = {
            'line_coords': result.get('line_coords'),
            'ocr_text': result.get('ocr_text', ''),
        }
        viz = draw_ocr_box_overlay(state.current_image, state.ocr_click_points, ocr_result_for_viz)

        conversion = result['conversion']
        ocr_text = result.get('ocr_text', '')
        pixel_length = result.get('pixel_length', 0)
        info_text = f"Bar: {pixel_length} px | OCR: \"{ocr_text}\" | {conversion:.4f} nm/px"

        return f"✅ OCR: {conversion:.4f} nm/pixel", viz, info_text

    except Exception as e:
        return f"❌ Error: {str(e)}", None, ""


def apply_manual_scale(um_value):
    """Apply manual scale from clicked endpoints + user-entered µm value (Manual mode)."""
    try:
        if state.current_image is None:
            return "❌ No image loaded", None, ""

        if state.scale_detector is None:
            return "❌ Scale detector not initialized", None, ""

        if len(state.manual_click_points) < 2:
            return "❌ Click two endpoints on the image first", None, ""

        if um_value is None or um_value <= 0:
            return "❌ Enter a positive scale bar length in µm", None, ""

        x1 = int(state.manual_click_points[0][0])
        x2 = int(state.manual_click_points[1][0])
        pixel_length = abs(x2 - x1)

        if pixel_length < 2:
            return "❌ Endpoints too close together", None, ""

        nm_value = float(um_value) * 1000  # µm → nm
        conversion = nm_value / pixel_length

        # Set scale via the detector
        state.scale_detector.set_manual_scale(conversion, 1)
        state.scale_info = state.scale_detector.last_detection

        # Detect databar for cropping
        databar_info = state.scale_detector.detect_databar(state.current_image)
        if databar_info.get('has_databar'):
            crop_pct = databar_info['databar_fraction'] * 100
            state.cropped_image = state.scale_detector.crop_scale_bar(
                state.current_image, crop_percent=crop_pct
            )
        else:
            state.cropped_image = state.current_image.copy()

        # Show overlay
        viz = draw_manual_scale_overlay(state.current_image, state.manual_click_points, pixel_length)

        info = f"Bar: {pixel_length} px = {um_value:.2f} µm → {conversion:.4f} nm/px"
        return f"✅ Manual: {conversion:.4f} nm/pixel ({pixel_length} px = {um_value} µm)", viz, info

    except Exception as e:
        return f"❌ Error: {str(e)}", None, ""


def handle_scale_click(evt: gr.SelectData):
    """Handle clicks on the image for OCR and Manual scale detection modes.

    Returns: (image, scale_status, ocr_info, manual_info)
    """
    try:
        if state.current_image is None:
            return None, "", "", ""

        x, y = evt.index[0], evt.index[1]

        if state.scale_mode == "Metadata":
            # No click interaction in Metadata mode
            return state.current_image, "", "", ""

        elif state.scale_mode == "OCR":
            if len(state.ocr_click_points) >= 2:
                # Already have 2 points — reset for new box
                state.ocr_click_points = []

            state.ocr_click_points.append((x, y))

            if len(state.ocr_click_points) == 1:
                # First corner — draw marker
                viz = draw_ocr_box_overlay(state.current_image, state.ocr_click_points)
                return viz, "", f"Corner 1 at ({x}, {y}). Click corner 2...", ""

            else:
                # Second corner — draw box and auto-run OCR
                status, viz, info = detect_scale_ocr_in_box()
                return viz, status, info, ""

        elif state.scale_mode == "Manual":
            if len(state.manual_click_points) >= 2:
                # Already have 2 points — reset for new measurement
                state.manual_click_points = []

            state.manual_click_points.append((x, y))

            if len(state.manual_click_points) == 1:
                # First endpoint — draw marker with zoom inset
                viz = draw_manual_scale_overlay(state.current_image, state.manual_click_points)
                return viz, "", "", f"Left endpoint at ({x}, {y}). Click the right end..."

            else:
                # Second endpoint — horizontal lock, draw line
                x1, y1 = state.manual_click_points[0]
                # Lock y to first point's y (horizontal constraint)
                state.manual_click_points[1] = (x, y1)
                viz = draw_manual_scale_overlay(state.current_image, state.manual_click_points)
                px_len = abs(int(x) - int(x1))
                return viz, "", "", f"Bar: {px_len} px. Enter length in µm and click Apply."

        return state.current_image, "", "", ""

    except Exception as e:
        return state.current_image, f"❌ Error: {str(e)}", "", ""


def set_scale_mode(mode):
    """Handle scale detection mode change."""
    state.scale_mode = mode
    state.ocr_click_points = []
    state.manual_click_points = []

    show_ocr = (mode == "OCR")
    show_manual = (mode == "Manual")

    # Restore original image display when switching modes
    img = state.current_image

    info_msg = ""
    if mode == "OCR":
        info_msg = "Click point 1 of 2..."
    elif mode == "Manual":
        info_msg = "Click point 1 of 2..."

    return (
        gr.update(visible=show_ocr),    # ocr_controls group
        gr.update(visible=show_manual),  # manual_controls group
        "Ready",                         # scale_status
        img,                             # current_image
        info_msg,                        # ocr_info
        info_msg,                        # manual_info
    )


def reset_scale_clicks():
    """Reset click points for current scale detection mode."""
    if state.scale_mode == "OCR":
        state.ocr_click_points = []
    elif state.scale_mode == "Manual":
        state.manual_click_points = []

    img = state.current_image
    return img, "Reset — click on the image to start."


def detect_scale_clicked(progress=gr.Progress()):
    """Handle the Detect Scale button click based on current mode."""
    if state.scale_mode == "Metadata":
        return detect_scale_metadata(progress)
    elif state.scale_mode == "OCR":
        if len(state.ocr_click_points) == 2:
            status, viz, info = detect_scale_ocr_in_box()
            return status, viz
        return "❌ Click two corners on the image first", state.current_image
    elif state.scale_mode == "Manual":
        return "Use the image clicks and µm input below", state.current_image
    return "❌ Unknown mode", None


def segment_with_sam(progress=gr.Progress()):
    """Segment image with SAM."""
    try:
        if state.cropped_image is None:
            return None, "❌ No image to segment (detect scale first)"

        if state.segmenter is None:
            return None, "❌ SAM not initialized"

        progress(0.3, desc="Running SAM...")
        state.masks, state.scores = state.segmenter.segment_image(
            state.cropped_image,
            multimask_output=True
        )

        progress(0.8, desc="Creating visualization...")
        mask_viz = visualize_three_masks(state.cropped_image, state.masks, state.scores)

        progress(1.0, desc="Complete!")

        return mask_viz, f"✅ Generated {len(state.masks)} mask candidates"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def select_mask_and_analyze(mask_choice, progress=gr.Progress()):
    """Select mask and run initial analysis."""
    try:
        if state.masks is None:
            return None, None, "❌ No masks available", None, None

        # Parse mask index
        mask_index = int(mask_choice.split()[1]) - 1
        state.selected_mask_index = mask_index
        state.segmenter.select_mask(mask_index)

        progress(0.3, desc="Analyzing particles...")

        # Create analyzer with user-specified minimum particle size
        state.analyzer = ParticleAnalyzer(
            conversion_factor=state.scale_info['conversion'],
            min_size=state.min_particle_size
        )

        # Get binary mask
        binary_mask = state.segmenter.get_binary_mask(invert=True)

        # Analyze (use single min_size parameter, remove redundant min_area)
        num_particles, regions = state.analyzer.analyze_mask(
            binary_mask,
            min_size=state.min_particle_size,
            remove_border=True,
            border_buffer=4
        )

        progress(0.7, desc="Creating visualization...")

        # Create visualization
        particle_viz = create_particle_visualization(
            state.cropped_image,
            state.analyzer.labeled_mask,
            state.analyzer.regions,
            show_labels=state.show_particle_numbers
        )

        # Get measurements
        measurements = state.analyzer.get_measurements(in_nm=True)
        results_df = create_results_dataframe(measurements)
        stats_df = create_summary_statistics_table(measurements)

        progress(1.0, desc="Complete!")

        status = f"✅ Detected {num_particles} particles - Ready for refinement"

        # Return: refine_viz, refine_results, analysis_status, current_results, current_stats
        return particle_viz, results_df, status, results_df, stats_df

    except Exception as e:
        return None, None, f"❌ Error: {str(e)}", None, None


# ============================================================================
# Tab 4: Refinement - Interactive Click Features
# ============================================================================

def get_current_visualization():
    """Get current particle visualization with all pending changes."""
    if state.analyzer is None:
        return None
    return create_particle_visualization(
        state.cropped_image,
        state.analyzer.labeled_mask,
        state.analyzer.regions,
        show_labels=state.show_particle_numbers,
        pending_deletes=state.pending_deletes,
        pending_add_masks=state.pending_add_masks,
        pending_merge=state.pending_merge
    )


def handle_image_click(evt: gr.SelectData):
    """Handle clicks on particle visualization for all refinement modes."""
    try:
        if state.analyzer is None:
            return get_current_visualization(), "❌ No analysis available"

        x, y = evt.index[0], evt.index[1]

        if state.click_mode == "delete":
            # DELETE MODE: Click particles to remove them
            region, idx, label = state.analyzer.find_particle_at_point(x, y)
            if region is not None:
                if label not in state.pending_deletes:
                    # Save pending state before modification (for undo of this click)
                    state.save_pending_state()
                    state.pending_deletes.append(label)

                particle_viz = create_particle_visualization(
                    state.cropped_image,
                    state.analyzer.labeled_mask,
                    state.analyzer.regions,
                    show_labels=state.show_particle_numbers,
                    pending_deletes=state.pending_deletes,
                    pending_add_masks=state.pending_add_masks
                )
                return particle_viz, f"🟡 Queued particle #{idx+1} for deletion (yellow outline)"
            else:
                return get_current_visualization(), "No particle found at this location"

        elif state.click_mode == "add":
            # ADD MODE: Click to add a single particle at this location
            try:
                # Save pending state before modification (for undo of this click)
                state.save_pending_state()

                # Use single positive point WITHOUT base_mask to segment just the clicked particle
                refined_mask, score = state.segmenter.refine_with_sam(
                    state.cropped_image,
                    [[x, y]],
                    [1],
                    base_mask=None,  # Don't constrain to existing mask - segment the clicked object
                    multimask_output=True,
                    image_already_encoded=True
                )

                state.pending_add_points.append((x, y))
                state.pending_add_masks.append(refined_mask)

                particle_viz = create_particle_visualization(
                    state.cropped_image,
                    state.analyzer.labeled_mask,
                    state.analyzer.regions,
                    show_labels=state.show_particle_numbers,
                    pending_deletes=state.pending_deletes,
                    pending_add_masks=state.pending_add_masks
                )
                return particle_viz, f"🟢 Added particle preview (green outline) - score: {score:.3f}"
            except Exception as e:
                return get_current_visualization(), f"❌ SAM refinement failed: {str(e)}"

        elif state.click_mode == "merge":
            # MERGE MODE: Click multiple particles to merge them
            region, idx, label = state.analyzer.find_particle_at_point(x, y)
            if region is not None:
                if label not in state.pending_merge:
                    # Save pending state before modification (for undo of this click)
                    state.save_pending_state()
                    state.pending_merge.append(label)

                # Visualization will show selected particles in different color
                particle_viz = create_particle_visualization(
                    state.cropped_image,
                    state.analyzer.labeled_mask,
                    state.analyzer.regions,
                    pending_deletes=state.pending_deletes,
                    pending_add_masks=state.pending_add_masks,
                    pending_merge=state.pending_merge
                )
                return particle_viz, f"🔵 Selected {len(state.pending_merge)} particles for merging"
            else:
                return get_current_visualization(), "No particle found at this location"

        elif state.click_mode == "point_refine":
            # POINT REFINE MODE: Click anywhere to add positive/negative points with live preview
            # Users can refine existing particles OR create new ones from scratch

            # Save pending state before adding point (for undo of this click)
            state.save_pending_state()

            # Check if user clicked on an existing particle (only on first click)
            if len(state.point_refine_points) == 0:
                region, idx, label = state.analyzer.find_particle_at_point(x, y)
                if region is not None:
                    # User clicked on existing particle - use it as base mask for IoU selection
                    state.point_refine_particle = label
                    state.point_refine_base_mask = (state.analyzer.labeled_mask == label).astype(bool)
                # If no particle found, that's OK - user is creating a new particle from scratch

            # Add the point
            point_label = 1 if state.point_type == "positive" else 0
            state.point_refine_points.append((x, y))
            state.point_refine_labels.append(point_label)

            # Generate live preview with SAM using all accumulated points
            try:
                # Compute ROI box from base mask (with padding)
                if state.point_refine_base_mask is not None and state.point_refine_base_mask.any():
                    ys, xs = np.where(state.point_refine_base_mask)
                    y0, y1 = int(ys.min()), int(ys.max())
                    x0, x1 = int(xs.min()), int(xs.max())
                    pad = 10
                    H, W = state.cropped_image.shape[:2]
                    roi_box = np.array([[
                        max(0, x0 - pad),
                        max(0, y0 - pad),
                        min(W - 1, x1 + pad),
                        min(H - 1, y1 + pad)
                    ]])
                else:
                    roi_box = None

                # Call SAM with all points using iterative refinement
                if state.point_refine_logits is not None:
                    # Use previous logits for iterative refinement (like the notebook)
                    masks_out, scores, logits_out = state.segmenter.sam_model.predictor.predict(
                        point_coords=np.array(state.point_refine_points, dtype=float),
                        point_labels=np.array(state.point_refine_labels, dtype=int),
                        box=roi_box,
                        mask_input=state.point_refine_logits[None, ...],  # Use previous mask logits
                        multimask_output=False  # Single mask output for iterative refinement
                    )
                    refined_mask = masks_out[0].astype(bool)
                    state.point_refine_logits = logits_out[0]  # Store for next iteration
                else:
                    # First point: get initial masks and select best one
                    masks_out, scores, logits_out = state.segmenter.sam_model.predictor.predict(
                        point_coords=np.array(state.point_refine_points, dtype=float),
                        point_labels=np.array(state.point_refine_labels, dtype=int),
                        box=roi_box,
                        multimask_output=True  # Multiple masks for initial selection
                    )

                    # Select best mask by IoU with base mask
                    if state.point_refine_base_mask is not None:
                        ious = []
                        for mask in masks_out:
                            intersection = np.logical_and(mask, state.point_refine_base_mask).sum()
                            union = np.logical_or(mask, state.point_refine_base_mask).sum()
                            iou = intersection / union if union > 0 else 0
                            ious.append(iou)
                        best_idx = int(np.argmax(ious))
                    else:
                        best_idx = int(np.argmax(scores))

                    refined_mask = masks_out[best_idx].astype(bool)
                    state.point_refine_logits = logits_out[best_idx]  # Store for next iteration

                # Clean up the refined mask using user-specified minimum size
                from skimage import morphology
                refined_mask = morphology.remove_small_objects(refined_mask, min_size=state.min_particle_size)

                # Store as preview (will be applied when user clicks Apply)
                state.point_refine_preview_mask = refined_mask

                # Create visualization with point markers overlaid
                particle_viz = create_point_refine_visualization(
                    state.cropped_image,
                    refined_mask,
                    state.point_refine_points,
                    state.point_refine_labels
                )

                point_type_str = "positive ✓" if point_label == 1 else "negative ✗"
                return particle_viz, f"➕ Added {point_type_str} point ({len(state.point_refine_points)} total)"

            except Exception as e:
                point_type_str = "positive ✓" if point_label == 1 else "negative ✗"
                return get_current_visualization(), f"➕ Added {point_type_str} point - Preview update failed: {str(e)}"

        return get_current_visualization(), "Click registered"

    except Exception as e:
        return get_current_visualization(), f"❌ Error: {str(e)}"


def set_min_particle_size(size):
    """Set minimum particle size for filtering."""
    state.min_particle_size = int(size)
    return f"✓ Minimum particle size set to {int(size)} pixels"


def toggle_particle_numbers(show_numbers):
    """Toggle visibility of particle numbers in visualization."""
    state.show_particle_numbers = show_numbers
    # Return updated visualization
    return get_current_visualization()


def set_point_type(point_type):
    """Set point type for point refine mode."""
    state.point_type = point_type
    return f"✓ Point type: {point_type.upper()}"


def reset_point_refine():
    """Reset point refine state and redraw visualization."""
    try:
        state.point_refine_particle = None
        state.point_refine_base_mask = None
        state.point_refine_points = []
        state.point_refine_labels = []
        state.point_refine_preview_mask = None
        state.point_refine_logits = None

        # Redraw regular visualization
        if state.analyzer is not None:
            particle_viz = create_particle_visualization(
                state.cropped_image,
                state.analyzer.labeled_mask,
                state.analyzer.regions,
                show_labels=state.show_particle_numbers,
                pending_deletes=state.pending_deletes,
                pending_add_masks=state.pending_add_masks,
                pending_merge=state.pending_merge
            )
            return particle_viz, "✅ Reset point refinement"
        else:
            return None, "✅ Reset point refinement"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def set_click_mode(mode):
    """Set click mode and return status message."""
    state.click_mode = mode

    # Reset mode-specific state
    if mode != "merge":
        state.pending_merge = []
    if mode != "point_refine":
        state.point_refine_particle = None
        state.point_refine_base_mask = None
        state.point_refine_points = []
        state.point_refine_labels = []
        state.point_refine_preview_mask = None
        state.point_refine_logits = None

    # Return appropriate status message and visibility for point refine controls
    mode_messages = {
        "delete": "🗑️ DELETE mode: Click particles to remove them",
        "add": "➕ ADD mode: Click empty areas to add new particles",
        "merge": "🔗 MERGE mode: Click multiple touching particles to merge them",
        "point_refine": "🎯 POINT REFINE mode: Add positive/negative points to refine or create particles"
    }

    # Show point refine controls only in point_refine mode
    show_controls = (mode == "point_refine")

    return mode_messages.get(mode, "Unknown mode"), gr.update(visible=show_controls)


def apply_refinement_changes(progress=gr.Progress()):
    """Apply all pending refinement changes (delete, add, merge, point_refine)."""
    try:
        if state.analyzer is None:
            return None, None, None, "❌ No analysis available"

        changes_made = False
        status_messages = []

        # Note: State is already saved on each click, no need to save here

        # Apply deletions
        if state.pending_deletes:
            progress(0.2, desc=f"Deleting {len(state.pending_deletes)} particles...")
            state.analyzer.delete_particles(state.pending_deletes)
            status_messages.append(f"Deleted {len(state.pending_deletes)} particles")
            state.pending_deletes = []
            changes_made = True

        # Apply additions using pre-generated masks
        if state.pending_add_masks:
            progress(0.4, desc=f"Adding {len(state.pending_add_masks)} particles...")
            for add_mask in state.pending_add_masks:
                state.analyzer.add_particle_from_sam(add_mask)
            status_messages.append(f"Added {len(state.pending_add_masks)} particles")
            state.pending_add_points = []
            state.pending_add_masks = []
            changes_made = True

        # Apply merge
        if state.pending_merge and len(state.pending_merge) >= 2:
            progress(0.6, desc=f"Merging {len(state.pending_merge)} particles...")
            state.analyzer.merge_particles(state.pending_merge)
            status_messages.append(f"Merged {len(state.pending_merge)} particles")
            state.pending_merge = []
            changes_made = True
        elif state.pending_merge:
            status_messages.append("⚠️ Need at least 2 particles to merge")
            state.pending_merge = []

        # Apply point refinement
        if state.point_refine_preview_mask is not None:
            progress(0.7, desc="Applying point refinement...")

            # Use the pre-generated preview mask (already refined during clicking)
            if state.point_refine_particle is not None:
                # Refining existing particle - delete old and add refined one
                state.analyzer.delete_particles([state.point_refine_particle])
                status_messages.append(f"Refined particle with {len(state.point_refine_points)} points")
            else:
                # Creating new particle from scratch
                status_messages.append(f"Created new particle with {len(state.point_refine_points)} points")

            # Add the refined/new particle
            state.analyzer.add_particle_from_sam(state.point_refine_preview_mask)

            state.point_refine_particle = None
            state.point_refine_base_mask = None
            state.point_refine_points = []
            state.point_refine_labels = []
            state.point_refine_preview_mask = None
            state.point_refine_logits = None
            changes_made = True

        if not changes_made:
            return None, None, "No changes to apply", None, None

        # Clear undo history since changes have been applied
        state.undo_history = []

        progress(0.9, desc="Updating visualization...")

        # Update visualization (no pending changes now)
        particle_viz = create_particle_visualization(
            state.cropped_image,
            state.analyzer.labeled_mask,
            state.analyzer.regions,
            show_labels=state.show_particle_numbers
        )

        # Update measurements
        measurements = state.analyzer.get_measurements(in_nm=True)
        results_df = create_results_dataframe(measurements)
        stats_df = create_summary_statistics_table(measurements)

        progress(1.0, desc="Complete!")

        num_particles = len(state.analyzer.regions)
        status = f"✅ {' | '.join(status_messages)} | Total: {num_particles} particles"

        # Return: refine_viz, refine_results, refine_status, current_results, current_stats
        return (
            particle_viz,
            results_df,
            status,
            results_df,
            stats_df
        )

    except Exception as e:
        return None, None, f"❌ Error: {str(e)}", None, None


def undo_last_action():
    """Undo the last click (removes last item from pending changes)."""
    try:
        if state.analyzer is None:
            return None, None, "❌ No analysis available"

        if not state.undo_history:
            return None, None, "❌ No actions to undo"

        # Restore previous pending state (before last click)
        previous_state = state.undo_history.pop()

        # Restore only the pending changes (not the mask itself)
        state.pending_deletes = previous_state['pending_deletes']
        state.pending_add_points = previous_state['pending_add_points']
        state.pending_add_masks = previous_state['pending_add_masks']
        state.pending_merge = previous_state['pending_merge']
        state.point_refine_particle = previous_state['point_refine_particle']
        state.point_refine_base_mask = previous_state['point_refine_base_mask']
        state.point_refine_points = previous_state['point_refine_points']
        state.point_refine_labels = previous_state['point_refine_labels']
        state.point_refine_preview_mask = previous_state['point_refine_preview_mask']
        state.point_refine_logits = previous_state['point_refine_logits']

        # Update visualization based on mode
        if state.click_mode == "point_refine" and state.point_refine_preview_mask is not None:
            # Show point refine visualization with restored points
            particle_viz = create_point_refine_visualization(
                state.cropped_image,
                state.point_refine_preview_mask,
                state.point_refine_points,
                state.point_refine_labels
            )
        else:
            # Show regular visualization with restored pending changes
            particle_viz = create_particle_visualization(
                state.cropped_image,
                state.analyzer.labeled_mask,
                state.analyzer.regions,
                show_labels=state.show_particle_numbers,
                pending_deletes=state.pending_deletes,
                pending_add_masks=state.pending_add_masks,
                pending_merge=state.pending_merge
            )

        # Measurements don't change (we're only undoing pending changes)
        measurements = state.analyzer.get_measurements(in_nm=True)
        results_df = create_results_dataframe(measurements)

        num_particles = len(state.analyzer.regions)
        status = f"↩️ Undone last click! {num_particles} particles | {len(state.undo_history)} undo steps remaining"

        return particle_viz, results_df, status

    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"


def clear_edge_particles(buffer_size):
    """Clear particles whose centroid is within buffer distance from edges."""
    try:
        if state.analyzer is None:
            return None, None, "❌ No analysis available", None, None

        buffer = int(buffer_size)
        H, W = state.analyzer.mask.shape

        # Identify particles to remove based on centroid position
        labels_to_remove = []
        for region in state.analyzer.regions:
            y, x = region.centroid
            # Check if centroid is within buffer distance from any edge
            if (x < buffer or x > W - buffer or
                y < buffer or y > H - buffer):
                labels_to_remove.append(region.label)

        n_removed = len(labels_to_remove)

        if n_removed > 0:
            state.analyzer.delete_particles(labels_to_remove)

        # Update visualization
        particle_viz = create_particle_visualization(
            state.cropped_image,
            state.analyzer.labeled_mask,
            state.analyzer.regions,
            show_labels=state.show_particle_numbers
        )

        # Update measurements
        measurements = state.analyzer.get_measurements(in_nm=True)
        results_df = create_results_dataframe(measurements)
        stats_df = create_summary_statistics_table(measurements)

        num_particles = len(state.analyzer.regions)
        status = f"✅ Removed {n_removed} edge particles (centroid within {buffer}px of edge) - Now {num_particles} total"

        # Return: refine_viz, refine_results, refine_status, current_results, current_stats
        return (
            particle_viz,
            results_df,
            status,
            results_df,
            stats_df
        )

    except Exception as e:
        return None, None, f"❌ Error: {str(e)}", None, None


def clear_all_particles():
    """Clear ALL particles from the mask, giving user a blank canvas to add particles manually."""
    try:
        if state.analyzer is None:
            return None, None, "❌ No analysis available", None, None

        # Get all current labels and delete them all
        all_labels = [r.label for r in state.analyzer.regions]
        n_removed = len(all_labels)

        if n_removed > 0:
            state.analyzer.delete_particles(all_labels)

        # Clear all pending state too
        state.pending_deletes = []
        state.pending_add_points = []
        state.pending_add_masks = []
        state.pending_merge = []
        state.point_refine_particle = None
        state.point_refine_base_mask = None
        state.point_refine_points = []
        state.point_refine_labels = []
        state.point_refine_preview_mask = None
        state.point_refine_logits = None
        state.undo_history = []

        # Redraw (empty image, no particles)
        particle_viz = create_particle_visualization(
            state.cropped_image,
            state.analyzer.labeled_mask,
            state.analyzer.regions,
            show_labels=state.show_particle_numbers
        )

        # Empty measurements
        measurements = state.analyzer.get_measurements(in_nm=True)
        results_df = create_results_dataframe(measurements)
        stats_df = create_summary_statistics_table(measurements)

        status = f"🗑️ Cleared all {n_removed} particles. Use 'add' mode to start segmenting."

        return particle_viz, results_df, status, results_df, stats_df

    except Exception as e:
        return None, None, f"❌ Error: {str(e)}", None, None


def clear_pending_changes():
    """Clear all pending changes and redraw visualization."""
    try:
        state.pending_deletes = []
        state.pending_add_points = []
        state.pending_add_masks = []
        state.pending_merge = []
        state.point_refine_particle = None
        state.point_refine_base_mask = None
        state.point_refine_points = []
        state.point_refine_labels = []
        state.point_refine_preview_mask = None
        state.point_refine_logits = None
        state.undo_history = []  # Clear undo history since all pending changes are cleared

        # Redraw visualization without pending changes
        if state.analyzer is not None:
            particle_viz = create_particle_visualization(
                state.cropped_image,
                state.analyzer.labeled_mask,
                state.analyzer.regions,
                pending_deletes=[],
                pending_add_masks=[],
                pending_merge=[]
            )
            return particle_viz, "✅ Cleared all pending changes"
        else:
            return None, "✅ Cleared all pending changes"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ============================================================================
# Tab 5: Plotting & Graphs
# ============================================================================

def update_histogram_plots():
    """Generate histogram plots for all particles from all images in the session."""
    try:
        if state.results_manager is None:
            return None, "❌ No results manager initialized. Save some image results first."

        # Get all results
        results_df = state.results_manager.get_results()

        if len(results_df) == 0:
            return None, "❌ No saved results to plot. Save at least one image's results first."

        # Collect all particle measurements from all images
        all_areas = []
        all_diameters = []

        for idx, row in results_df.iterrows():
            # Parse the string lists back to arrays
            areas_nm2 = eval(row['particle_areas_nm2']) if row['particle_areas_nm2'] != '[]' else []
            diams_nm = eval(row['equiv_diameters_nm']) if row['equiv_diameters_nm'] != '[]' else []

            all_areas.extend(areas_nm2)
            all_diameters.extend(diams_nm)

        total_particles = len(all_areas)

        if total_particles == 0:
            return None, "❌ No particles found in saved results"

        # Create combined measurements dictionary
        combined_measurements = {
            'num_particles': total_particles,
            'areas': all_areas,
            'diameters': all_diameters,
            'unit': 'nm'
        }

        # Create histograms
        from visualization import create_histogram_plots
        histogram_img = create_histogram_plots(combined_measurements)

        return histogram_img, f"✅ Generated histograms for {total_particles} particles from {len(results_df)} images"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ============================================================================
# Tab 6: Results - Save and Export
# ============================================================================

def save_current_results():
    """Save current image results and mark as processed."""
    try:
        if state.analyzer is None:
            return "❌ No analysis to save", None

        if state.results_manager is None:
            return "❌ Results manager not initialized", None

        # Get measurements
        measurements = state.analyzer.get_measurements(in_nm=True)

        # Save to CSV
        filename = os.path.basename(state.image_paths[state.current_index])
        state.results_manager.add_result(filename, measurements)

        # Mark as processed
        state.mark_processed(state.current_index, measurements['num_particles'])
        state.save_state()

        # Update gallery
        gallery_data = create_image_gallery()

        return (
            f"✅ Saved results for {filename}",
            gallery_data
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", None


def get_session_summary():
    """Get summary of all processed images with aggregate particle statistics."""
    try:
        if state.results_manager is None:
            return None, "No results available", "No particle statistics available", gr.update(choices=[], value=None)

        results_df = state.results_manager.get_results()

        if len(results_df) == 0:
            return None, "No images processed yet", "No particle statistics available", gr.update(choices=[], value=None)

        summary = state.results_manager.get_summary()

        # Progress summary (left column)
        progress_text = f"""
### Session Progress

- **Images Processed:** {summary['total_images']} / {len(state.image_paths)}
- **Total Particles:** {summary['total_particles']}
- **Average Particles/Image:** {summary['avg_particles_per_image']:.1f}
- **Min Particles/Image:** {summary['min_particles']}
- **Max Particles/Image:** {summary['max_particles']}
        """

        # Aggregate particle statistics (right column)
        # Collect all particle diameters from all images
        all_diameters = []
        for idx, row in results_df.iterrows():
            diams_nm = eval(row['equiv_diameters_nm']) if row['equiv_diameters_nm'] != '[]' else []
            all_diameters.extend(diams_nm)

        if len(all_diameters) > 0:
            import numpy as np
            mean_diam = np.mean(all_diameters) / 1000  # Convert to μm
            median_diam = np.median(all_diameters) / 1000
            std_diam = np.std(all_diameters) / 1000
            min_diam = np.min(all_diameters) / 1000
            max_diam = np.max(all_diameters) / 1000

            particle_stats_text = f"""
### Aggregate Particle Statistics

- **Mean Diameter:** {mean_diam:.3f} μm
- **Median Diameter:** {median_diam:.3f} μm
- **Std Deviation:** {std_diam:.3f} μm
- **Min Diameter:** {min_diam:.3f} μm
- **Max Diameter:** {max_diam:.3f} μm
            """
        else:
            particle_stats_text = "### Aggregate Particle Statistics\n\nNo particle data available"

        display_df = results_df[['file_name', 'num_particles']].copy()
        display_df.columns = ['Filename', 'Particle Count']

        # Create dropdown choices for delete functionality
        # Format: "index: filename" so user can see both
        dropdown_choices = [f"{i}: {row['file_name']}" for i, row in results_df.iterrows()]

        return display_df, progress_text, particle_stats_text, gr.update(choices=dropdown_choices, value=None)

    except Exception as e:
        return None, f"❌ Error: {str(e)}", f"❌ Error: {str(e)}", gr.update(choices=[], value=None)


def delete_result_row(selected_item):
    """Delete a specific row from the results table."""
    try:
        if state.results_manager is None:
            return None, None, None, gr.update(), "❌ No results available"

        if selected_item is None:
            return None, None, None, gr.update(), "❌ Please select a row to delete"

        results_df = state.results_manager.get_results()

        if len(results_df) == 0:
            return None, None, None, gr.update(), "❌ No results to delete"

        # Parse the selection format "index: filename"
        idx_str = selected_item.split(":")[0].strip()
        idx = int(idx_str)

        # Get filename for confirmation message
        filename = results_df.loc[idx, 'file_name']

        # Delete the row
        state.results_manager.delete_result(idx)

        # Get updated summary (includes new dropdown choices)
        display_df, progress_text, particle_stats_text, dropdown_update = get_session_summary()

        return display_df, progress_text, particle_stats_text, dropdown_update, f"✅ Deleted: {filename}"

    except IndexError as e:
        return None, None, None, gr.update(), f"❌ Index error: {str(e)}"
    except Exception as e:
        return None, None, None, gr.update(), f"❌ Error: {str(e)}"


def export_results():
    """Export all results to CSV."""
    try:
        if state.results_manager is None or len(state.results_manager.results_df) == 0:
            return None

        csv_path = state.results_manager.csv_file
        return csv_path

    except Exception as e:
        return None


def check_and_remove_duplicates():
    """Check for duplicate entries and remove them."""
    try:
        if state.results_manager is None:
            return None, "❌ No results manager initialized"

        # Find duplicates
        duplicates = state.results_manager.find_duplicates()

        if len(duplicates) == 0:
            return None, "✅ No duplicate entries found"

        # Delete duplicates (keep last occurrence)
        deleted_count = state.results_manager.delete_duplicates(keep='last')

        # Get updated summary
        results_df = state.results_manager.get_results()
        if len(results_df) == 0:
            display_df = None
        else:
            display_df = results_df[['file_name', 'num_particles']].copy()
            display_df.columns = ['Filename', 'Particle Count']

        return display_df, f"✅ Removed {deleted_count} duplicate entries (kept last occurrence)"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ============================================================================
# Create Gradio Interface
# ============================================================================

def create_interface():
    """Create the tabbed Gradio interface."""

    css = """
    .tabs {font-size: 16px; font-weight: 500;}
    .tab-nav button {padding: 12px 24px;}
    """

    with gr.Blocks(css=css, title="SAM-based SEM Particle Analysis", theme=gr.themes.Soft()) as app:

        gr.Markdown("# 🔬 SAM-based SEM Particle Analysis")
        gr.Markdown("Automated particle segmentation and analysis for scanning electron microscopy images using Segment Anything Model")

        with gr.Tabs() as tabs:

            # ================================================================
            # TAB 1: Setup
            # ================================================================
            with gr.Tab("⚙️ Setup", id=0):
                gr.Markdown("## Initialize SAM Model and Load Images")

                with gr.Row():
                    with gr.Column(scale=2):
                        sam_file = gr.File(
                            label="Select SAM Checkpoint File (.pth)",
                            file_types=[".pth"],
                            type="filepath"
                        )
                        model_type = gr.Radio(
                            choices=["vit_h", "vit_b"],
                            value="vit_h",
                            label="Model Type"
                        )
                        init_sam_btn = gr.Button("⚡ Initialize SAM Model", variant="primary", size="lg")
                        init_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=2):
                        gr.Markdown("**Load Images**")
                        gr.Markdown("Select one or more images. If you select multiple images from the same folder, all images in that folder will be loaded.")
                        file_input = gr.File(
                            label="Select Image Files",
                            file_count="multiple",
                            file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                            type="filepath"
                        )

                        load_btn = gr.Button("📁 Load Images", variant="primary", size="lg", interactive=False)
                        load_status = gr.Textbox(label="Status", interactive=False)

                        gr.Markdown("---")
                        gr.Markdown("**Resume Previous Session**")
                        gr.Markdown("Upload a previous results CSV to continue adding to it.")
                        resume_csv_input = gr.File(
                            label="Select Results CSV",
                            file_count="single",
                            file_types=[".csv"],
                            type="filepath"
                        )
                        resume_btn = gr.Button("📂 Resume Session", variant="secondary")
                        resume_status = gr.Textbox(label="Resume Status", interactive=False)

            # ================================================================
            # TAB 2: Image Gallery
            # ================================================================
            with gr.Tab("🖼️ Image Gallery", id=1):
                gr.Markdown("## Image Gallery - Click to Select")
                gr.Markdown("✅ = Processed | ⚪ = Not Processed")

                gallery = gr.Gallery(
                    label="All Images",
                    columns=5,
                    rows=4,
                    height=800,
                    object_fit="contain",
                    show_label=True,
                    allow_preview=True,
                    type="pil"  # Explicitly set to PIL Image type for cross-platform compatibility
                )

                selected_image_info = gr.Textbox(label="Selected Image", interactive=False)

            # ================================================================
            # TAB 3: Processing
            # ================================================================
            with gr.Tab("🔍 Processing", id=2):
                gr.Markdown("## Scale Detection and Segmentation")

                current_image = gr.Image(
                    label="",
                    type="numpy"
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Step 1: Scale Detection")
                        scale_mode_dropdown = gr.Dropdown(
                            choices=["Metadata", "OCR", "Manual"],
                            value="Metadata",
                            label="Detection Mode",
                            info="Metadata: TIFF tags | OCR: click box on image | Manual: click endpoints"
                        )
                        detect_btn = gr.Button("🔍 Detect Scale", variant="primary")
                        scale_status = gr.Textbox(label="Status", interactive=False)

                        # OCR mode controls (hidden by default)
                        with gr.Group(visible=False) as ocr_controls:
                            gr.Markdown("**OCR Mode:** Click two opposite corners on the image to define the search box.")
                            ocr_info = gr.Textbox(
                                label="OCR Info",
                                value="Click point 1 of 2...",
                                interactive=False
                            )
                            ocr_reset_btn = gr.Button("Reset OCR Box", size="sm")

                        # Manual mode controls (hidden by default)
                        with gr.Group(visible=False) as manual_controls:
                            gr.Markdown("**Manual Mode:** Click the left end of the scale bar, then the right end.")
                            manual_info = gr.Textbox(
                                label="Manual Info",
                                value="Click point 1 of 2...",
                                interactive=False
                            )
                            manual_um_input = gr.Number(
                                label="Scale bar length (µm)",
                                info="Enter the physical length of the scale bar in micrometers",
                                precision=3,
                                minimum=0.001,
                                maximum=100000
                            )
                            manual_apply_btn = gr.Button("Apply Manual Scale", variant="primary")
                            manual_reset_btn = gr.Button("Reset Points", size="sm")

                    with gr.Column():
                        gr.Markdown("### Step 2: Segmentation")
                        min_particle_size_slider = gr.Slider(
                            minimum=5,
                            maximum=100,
                            value=30,
                            step=1,
                            label="Minimum Particle Size (pixels)",
                            info="Particles smaller than this will be filtered out. Lower for small/low-mag images."
                        )
                        segment_btn = gr.Button("🤖 Segment with SAM", variant="primary", size="lg")
                        segment_status = gr.Textbox(label="Status", interactive=False)

                mask_viz = gr.Image(label="")

                with gr.Row():
                    mask_choice = gr.Radio(
                        choices=["Mask 1", "Mask 2", "Mask 3"],
                        value="Mask 1",
                        label="Select Best Mask"
                    )
                    analyze_btn = gr.Button("✓ Select & Analyze", variant="primary", size="lg")

                analysis_status = gr.Textbox(label="Analysis Status", interactive=False)

            # ================================================================
            # TAB 4: Refinement
            # ================================================================
            with gr.Tab("✏️ Refinement", id=3):
                gr.Markdown("## Interactive Particle Refinement")
                gr.Markdown("**Select a mode and click on the image to refine particle segmentation**")

                with gr.Row():
                    with gr.Column(scale=1):
                        click_mode_radio = gr.Radio(
                            choices=["delete", "add", "merge", "point_refine"],
                            value="delete",
                            label="Refinement Mode",
                            info="Select how you want to interact with particles"
                        )

                        gr.Markdown("""
**Mode Descriptions:**
- **Delete**: Click particles to remove them
- **Add**: Click empty areas to add new particles
- **Merge**: Click multiple touching particles to merge them
- **Point Refine**: Click to add positive/negative points to refine a selected particle
                        """)

                    with gr.Column(scale=2):
                        click_mode_status = gr.Textbox(
                            label="Current Mode",
                            value="🗑️ DELETE mode: Click particles to remove them",
                            interactive=False
                        )

                        # Point refine controls (only visible in point_refine mode)
                        with gr.Group(visible=False) as point_refine_controls:
                            gr.Markdown("**Point Refinement Controls**")
                            point_type_radio = gr.Radio(
                                choices=["positive", "negative"],
                                value="positive",
                                label="Point Type",
                                info="Positive = include, Negative = exclude"
                            )
                            selected_particle_id = gr.Textbox(
                                label="Selected Particle ID",
                                value="None",
                                interactive=False
                            )
                            reset_points_btn = gr.Button("Reset Points", size="sm")

                # Visualization controls
                show_numbers_checkbox = gr.Checkbox(
                    label="Show Particle Numbers",
                    value=True,
                    info="Uncheck to hide numbers for better visibility of small particles"
                )

                refine_viz = gr.Image(label="", type="numpy")

                with gr.Row():
                    apply_btn = gr.Button("✓ Apply Changes", variant="primary", size="lg")
                    clear_pending_btn = gr.Button("Clear Pending", variant="secondary")
                    undo_btn = gr.Button("↩️ Undo Last Action", variant="secondary")

                with gr.Row():
                    edge_buffer = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Edge Buffer (pixels)"
                    )
                    with gr.Column():
                        clear_edges_btn = gr.Button("🧹 Clear Edge Particles", variant="secondary")
                        clear_all_btn = gr.Button("🗑️ Clear All Particles", variant="stop")

                refine_status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("Particle Measurements", open=True):
                    refine_results = gr.Dataframe(label="Particle Measurements")

                gr.Markdown("---")
                gr.Markdown("### Save Results")
                with gr.Row():
                    save_btn = gr.Button("💾 Save Current Image Results", variant="primary", size="lg")
                    save_status = gr.Textbox(label="Status", interactive=False)

            # ================================================================
            # TAB 5: Results & Export
            # ================================================================
            with gr.Tab("💾 Results & Export", id=4):
                gr.Markdown("## Current Image Analysis Summary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Particle Measurements")
                        current_results = gr.Dataframe(label="All Particles")

                    with gr.Column():
                        gr.Markdown("### Summary Statistics")
                        current_stats = gr.Dataframe(label="Statistics")

                gr.Markdown("---")
                gr.Markdown("## Session Summary (All Images)")

                refresh_btn = gr.Button("🔄 Refresh Summary")

                # Two-column layout for session statistics
                with gr.Row():
                    with gr.Column():
                        summary_progress = gr.Markdown("No results yet")
                    with gr.Column():
                        summary_particle_stats = gr.Markdown("No particle statistics yet")

                session_table = gr.Dataframe(label="All Processed Images")

                with gr.Row():
                    delete_row_dropdown = gr.Dropdown(
                        label="Select Row to Delete",
                        choices=[],
                        value=None,
                        interactive=True,
                        info="Choose a file to delete from results"
                    )
                    delete_row_btn = gr.Button("❌ Delete Selected Row", variant="stop")

                delete_row_status = gr.Textbox(label="Delete Status", interactive=False)

                with gr.Row():
                    remove_duplicates_btn = gr.Button("🧹 Remove Duplicate Entries", variant="secondary")
                    duplicates_status = gr.Textbox(label="Duplicate Removal Status", interactive=False)

                gr.Markdown("---")
                gr.Markdown("## Export Results")

                with gr.Row():
                    export_btn = gr.Button("📥 Export All Results", variant="primary")
                    export_file = gr.File(label="Download CSV")

            # ================================================================
            # TAB 6: Plotting & Graphs
            # ================================================================
            with gr.Tab("📊 Plotting & Graphs", id=5):
                gr.Markdown("## Particle Size Distribution")
                gr.Markdown("Visualize the distribution of particle sizes with histograms")

                with gr.Row():
                    update_plots_btn = gr.Button("🔄 Update Plots", variant="primary", size="lg")
                    plot_status = gr.Textbox(label="Status", interactive=False)

                histogram_plot = gr.Image(label="", type="numpy")

        # ================================================================
        # Event Handlers
        # ================================================================

        # Setup tab
        init_sam_btn.click(
            initialize_sam,
            inputs=[sam_file, model_type],
            outputs=[init_status, load_btn]
        )

        load_btn.click(
            load_images_from_folder,
            inputs=[file_input],
            outputs=[load_status, gallery]
        )

        resume_btn.click(
            resume_session,
            inputs=[resume_csv_input],
            outputs=[resume_status, gallery, load_status]
        )

        # Gallery tab
        gallery.select(
            select_image_from_gallery,
            outputs=[current_image, selected_image_info, tabs, click_mode_radio]
        )

        # Processing tab — scale detection mode
        scale_mode_dropdown.change(
            set_scale_mode,
            inputs=[scale_mode_dropdown],
            outputs=[ocr_controls, manual_controls, scale_status, current_image, ocr_info, manual_info]
        )

        # Click handler on processing image for OCR / Manual modes
        current_image.select(
            handle_scale_click,
            outputs=[current_image, scale_status, ocr_info, manual_info]
        )

        detect_btn.click(
            detect_scale_clicked,
            outputs=[scale_status, current_image]
        )

        # OCR mode reset
        ocr_reset_btn.click(
            reset_scale_clicks,
            outputs=[current_image, ocr_info]
        )

        # Manual mode controls
        manual_apply_btn.click(
            apply_manual_scale,
            inputs=[manual_um_input],
            outputs=[scale_status, current_image, manual_info]
        )

        manual_reset_btn.click(
            reset_scale_clicks,
            outputs=[current_image, manual_info]
        )

        min_particle_size_slider.change(
            set_min_particle_size,
            inputs=[min_particle_size_slider],
            outputs=[segment_status]
        )

        segment_btn.click(
            segment_with_sam,
            outputs=[mask_viz, segment_status]
        )

        analyze_btn.click(
            select_mask_and_analyze,
            inputs=[mask_choice],
            outputs=[refine_viz, refine_results, analysis_status, current_results, current_stats]
        )

        # Refinement tab
        click_mode_radio.change(
            set_click_mode,
            inputs=[click_mode_radio],
            outputs=[click_mode_status, point_refine_controls]
        )

        point_type_radio.change(
            set_point_type,
            inputs=[point_type_radio],
            outputs=[refine_status]
        )

        reset_points_btn.click(
            reset_point_refine,
            outputs=[refine_viz, refine_status]
        )

        show_numbers_checkbox.change(
            toggle_particle_numbers,
            inputs=[show_numbers_checkbox],
            outputs=[refine_viz]
        )

        refine_viz.select(
            handle_image_click,
            outputs=[refine_viz, refine_status]
        )

        apply_btn.click(
            apply_refinement_changes,
            outputs=[refine_viz, refine_results, refine_status, current_results, current_stats]
        )

        clear_pending_btn.click(
            clear_pending_changes,
            outputs=[refine_viz, refine_status]
        )

        undo_btn.click(
            undo_last_action,
            outputs=[refine_viz, refine_results, refine_status]
        )

        clear_edges_btn.click(
            clear_edge_particles,
            inputs=[edge_buffer],
            outputs=[refine_viz, refine_results, refine_status, current_results, current_stats]
        )

        clear_all_btn.click(
            clear_all_particles,
            outputs=[refine_viz, refine_results, refine_status, current_results, current_stats]
        )

        # Plotting tab
        update_plots_btn.click(
            update_histogram_plots,
            outputs=[histogram_plot, plot_status]
        )

        # Results tab
        save_btn.click(
            save_current_results,
            outputs=[save_status, gallery]
        )

        refresh_btn.click(
            get_session_summary,
            outputs=[session_table, summary_progress, summary_particle_stats, delete_row_dropdown]
        )

        delete_row_btn.click(
            delete_result_row,
            inputs=[delete_row_dropdown],
            outputs=[session_table, summary_progress, summary_particle_stats, delete_row_dropdown, delete_row_status]
        )

        remove_duplicates_btn.click(
            check_and_remove_duplicates,
            outputs=[session_table, duplicates_status]
        )

        export_btn.click(
            export_results,
            outputs=[export_file]
        )

    return app


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    app = create_interface()
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860
    )
