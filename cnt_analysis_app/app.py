"""
CNT Particle Analysis Gradio Application

A user-friendly web interface for segmenting and analyzing carbon nanotube
particles in electron microscopy images using Meta's Segment Anything Model.
"""

import os
import sys
import gradio as gr
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path to import sem_particle_analysis package
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
    visualize_three_masks,
    create_results_dataframe,
    create_summary_statistics_table
)


# ============================================================================
# Global State Management
# ============================================================================

class AppState:
    """Manages application state across function calls."""

    def __init__(self):
        self.sam_model = None
        self.scale_detector = None
        self.image_paths = []
        self.current_index = 0
        self.current_image = None
        self.cropped_image = None
        self.scale_info = None
        self.masks = None
        self.scores = None
        self.selected_mask_index = None
        self.segmenter = None
        self.analyzer = None
        self.results_manager = None

    def reset_image_state(self):
        """Reset state for a new image."""
        self.current_image = None
        self.cropped_image = None
        self.scale_info = None
        self.masks = None
        self.scores = None
        self.selected_mask_index = None
        self.analyzer = None


# Create global state
state = AppState()


# ============================================================================
# Configuration Functions
# ============================================================================

def initialize_sam(checkpoint_path, model_type, progress=gr.Progress()):
    """
    Initialize the SAM model.

    Args:
        checkpoint_path: Path to SAM checkpoint file
        model_type: Model variant ('vit_h' or 'vit_b')
        progress: Gradio progress tracker

    Returns:
        str: Status message
    """
    try:
        if not os.path.exists(checkpoint_path):
            return f"❌ Error: SAM checkpoint not found at {checkpoint_path}"

        progress(0.2, desc="Loading SAM model...")

        # Initialize SAM
        state.sam_model = SAMModel(checkpoint_path, model_type=model_type)
        state.segmenter = ParticleSegmenter(state.sam_model)

        progress(0.6, desc="Initializing scale detector...")

        # Initialize scale detector
        state.scale_detector = ScaleDetector(use_gpu=False)

        progress(1.0, desc="Initialization complete!")

        return f"✅ SAM model initialized successfully ({model_type})"

    except Exception as e:
        return f"❌ Error initializing SAM: {str(e)}"


def load_image_folder(folder_path, progress=gr.Progress()):
    """
    Load all images from a folder.

    Args:
        folder_path: Path to folder containing images
        progress: Gradio progress tracker

    Returns:
        tuple: (status message, first image, image info)
    """
    try:
        if not os.path.exists(folder_path):
            return "❌ Error: Folder not found", None, "No images loaded"

        progress(0.3, desc="Scanning for images...")

        # Find all images
        state.image_paths = find_images_in_folder(folder_path)

        if not state.image_paths:
            return "❌ Error: No images found in folder", None, "No images loaded"

        state.current_index = 0

        progress(0.7, desc="Loading first image...")

        # Initialize results manager
        csv_path = os.path.join(folder_path, "analysis_results.csv")
        state.results_manager = ResultsManager(csv_file=csv_path)

        # Load first image
        state.reset_image_state()
        state.current_image = load_image(state.image_paths[0])

        progress(1.0, desc="Loading complete!")

        info = f"📁 Loaded {len(state.image_paths)} images"
        current_info = f"Image 1 / {len(state.image_paths)}: {os.path.basename(state.image_paths[0])}"

        return info, state.current_image, current_info

    except Exception as e:
        return f"❌ Error loading images: {str(e)}", None, "No images loaded"


# ============================================================================
# Image Navigation Functions
# ============================================================================

def navigate_to_image(direction, jump_index=None):
    """
    Navigate to a different image.

    Args:
        direction: 'prev', 'next', or 'jump'
        jump_index: Image index for jump (1-based)

    Returns:
        tuple: (image, image info, empty states for reset)
    """
    try:
        if not state.image_paths:
            return None, "No images loaded", None, "", None, None, None

        # Calculate new index
        if direction == 'prev':
            new_index = max(0, state.current_index - 1)
        elif direction == 'next':
            new_index = min(len(state.image_paths) - 1, state.current_index + 1)
        elif direction == 'jump' and jump_index is not None:
            new_index = max(0, min(len(state.image_paths) - 1, jump_index - 1))
        else:
            new_index = state.current_index

        # Update index and load image
        state.current_index = new_index
        state.reset_image_state()
        state.current_image = load_image(state.image_paths[state.current_index])

        current_info = f"Image {state.current_index + 1} / {len(state.image_paths)}: {os.path.basename(state.image_paths[state.current_index])}"

        # Return image and reset downstream components
        return (
            state.current_image,
            current_info,
            None,  # scale_detection_output
            "",    # scale_text
            None,  # mask_visualization
            None,  # particle_visualization
            None   # results_table
        )

    except Exception as e:
        return None, f"❌ Error: {str(e)}", None, "", None, None, None


# ============================================================================
# Scale Detection Functions
# ============================================================================

def detect_scale_auto(progress=gr.Progress()):
    """
    Automatically detect scale bar in current image.

    Returns:
        tuple: (status message, scale value for manual input)
    """
    try:
        if state.current_image is None:
            return "❌ No image loaded", ""

        if state.scale_detector is None:
            return "❌ Scale detector not initialized", ""

        progress(0.3, desc="Detecting scale bar...")

        # Detect scale
        state.scale_info = state.scale_detector.detect_scale_bar(state.current_image)

        progress(0.7, desc="Cropping scale bar...")

        # Crop scale bar
        state.cropped_image = state.scale_detector.crop_scale_bar(state.current_image)

        progress(1.0, desc="Scale detection complete!")

        scale_nm = state.scale_info['scale_nm']
        conversion = state.scale_info['conversion']

        return (
            f"✅ Detected: {scale_nm:.0f} nm ({conversion:.3f} nm/pixel)",
            f"{scale_nm:.0f}"
        )

    except Exception as e:
        return f"❌ Scale detection failed: {str(e)}", ""


def set_scale_manual(scale_nm_text):
    """
    Manually set the scale.

    Args:
        scale_nm_text: Scale value in nm (as string)

    Returns:
        str: Status message
    """
    try:
        if state.current_image is None:
            return "❌ No image loaded"

        scale_nm = float(scale_nm_text)

        if scale_nm <= 0:
            return "❌ Scale must be positive"

        # Use a default pixel length (e.g., 100 pixels)
        # In a real scenario, user would measure this
        pixel_length = 100

        conversion = state.scale_detector.set_manual_scale(scale_nm, pixel_length)
        state.scale_info = state.scale_detector.last_detection
        state.cropped_image = state.scale_detector.crop_scale_bar(state.current_image)

        return f"✅ Manual scale set: {scale_nm:.0f} nm ({conversion:.3f} nm/pixel)"

    except ValueError:
        return "❌ Invalid scale value"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# Segmentation Functions
# ============================================================================

def segment_with_sam(progress=gr.Progress()):
    """
    Segment the current image using SAM.

    Returns:
        tuple: (mask visualization, status message)
    """
    try:
        if state.cropped_image is None:
            return None, "❌ No image to segment (detect scale first)"

        if state.segmenter is None:
            return None, "❌ SAM not initialized"

        progress(0.3, desc="Running SAM encoder...")

        # Segment image
        state.masks, state.scores = state.segmenter.segment_image(
            state.cropped_image,
            multimask_output=True
        )

        progress(0.8, desc="Generating visualizations...")

        # Create visualization
        mask_viz = visualize_three_masks(state.cropped_image, state.masks, state.scores)

        progress(1.0, desc="Segmentation complete!")

        return (
            mask_viz,
            f"✅ Generated {len(state.masks)} mask candidates"
        )

    except Exception as e:
        return None, f"❌ Segmentation failed: {str(e)}"


def select_mask(mask_choice):
    """
    Select a specific mask for analysis.

    Args:
        mask_choice: "Mask 1", "Mask 2", or "Mask 3"

    Returns:
        str: Status message
    """
    try:
        if state.masks is None:
            return "❌ No masks available (run segmentation first)"

        # Parse mask index from choice
        mask_index = int(mask_choice.split()[1]) - 1

        if mask_index < 0 or mask_index >= len(state.masks):
            return "❌ Invalid mask selection"

        state.selected_mask_index = mask_index
        state.segmenter.select_mask(mask_index)

        return f"✅ Selected {mask_choice} (score: {state.scores[mask_index]:.3f})"

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# Particle Analysis Functions
# ============================================================================

def analyze_particles(progress=gr.Progress()):
    """
    Analyze particles from selected mask.

    Returns:
        tuple: (particle visualization, results table, summary stats, status)
    """
    try:
        if state.selected_mask_index is None:
            return None, None, None, "❌ No mask selected"

        if state.scale_info is None:
            return None, None, None, "❌ Scale information not available"

        progress(0.3, desc="Analyzing particles...")

        # Create analyzer
        state.analyzer = ParticleAnalyzer(
            conversion_factor=state.scale_info['conversion']
        )

        # Get binary mask (inverted for particles)
        binary_mask = state.segmenter.get_binary_mask(invert=True)

        # Analyze
        num_particles, regions = state.analyzer.analyze_mask(
            binary_mask,
            min_area=50,
            min_size=30,
            remove_border=True,
            border_buffer=4
        )

        progress(0.7, desc="Generating visualizations...")

        # Create visualization
        particle_viz = create_particle_visualization(
            state.cropped_image,
            state.analyzer.labeled_mask,
            state.analyzer.regions
        )

        # Get measurements
        measurements = state.analyzer.get_measurements(in_nm=True)

        # Create tables
        results_df = create_results_dataframe(measurements)
        stats_df = create_summary_statistics_table(measurements)

        progress(0.9, desc="Saving results...")

        # Save to results manager
        if state.results_manager is not None:
            filename = os.path.basename(state.image_paths[state.current_index])
            state.results_manager.add_result(filename, measurements)

        progress(1.0, desc="Analysis complete!")

        status = f"✅ Detected {num_particles} particles | Results saved to CSV"

        return particle_viz, results_df, stats_df, status

    except Exception as e:
        return None, None, None, f"❌ Analysis failed: {str(e)}"


# ============================================================================
# Batch Processing Functions
# ============================================================================

def get_session_summary():
    """
    Get summary of all processed images.

    Returns:
        tuple: (summary DataFrame, statistics text)
    """
    try:
        if state.results_manager is None:
            return None, "No results available"

        # Get all results
        results_df = state.results_manager.get_results()

        if len(results_df) == 0:
            return None, "No images processed yet"

        # Get summary stats
        summary = state.results_manager.get_summary()

        stats_text = f"""
📊 **Session Summary**

- **Images Processed:** {summary['total_images']} / {len(state.image_paths)}
- **Total Particles:** {summary['total_particles']}
- **Average Particles/Image:** {summary['avg_particles_per_image']:.1f}
- **Min Particles:** {summary['min_particles']}
- **Max Particles:** {summary['max_particles']}
        """

        # Create display dataframe
        display_df = results_df[['file_name', 'num_particles']].copy()
        display_df.columns = ['Filename', 'Particle Count']

        return display_df, stats_text

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def export_session_results():
    """
    Export all session results to CSV.

    Returns:
        tuple: (file path, status message)
    """
    try:
        if state.results_manager is None or len(state.results_manager.results_df) == 0:
            return None, "❌ No results to export"

        # Results are already being saved incrementally
        csv_path = state.results_manager.csv_file

        return csv_path, f"✅ Results saved to: {csv_path}"

    except Exception as e:
        return None, f"❌ Export failed: {str(e)}"


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create and configure the Gradio interface."""

    # Custom CSS for styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .config-section {
        background-color: #f0f9ff;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .main-section {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
    }
    """

    with gr.Blocks(css=css, title="CNT Particle Analysis", theme=gr.themes.Soft()) as app:

        gr.Markdown(
            """
            # 🔬 CNT Particle Segmentation & Analysis Tool

            Automated analysis of carbon nanotube particles in electron microscopy images using SAM
            """
        )

        # ====================================================================
        # Configuration Section
        # ====================================================================

        with gr.Group():
            gr.Markdown("## 🔧 Configuration")

            with gr.Row():
                sam_checkpoint = gr.Textbox(
                    label="SAM Model Checkpoint Path",
                    placeholder="/path/to/sam_vit_h_4b8939.pth",
                    scale=3
                )
                model_type = gr.Radio(
                    choices=["vit_h", "vit_b"],
                    value="vit_h",
                    label="Model Type",
                    scale=1
                )

            with gr.Row():
                image_folder = gr.Textbox(
                    label="Image Folder Path",
                    placeholder="/path/to/images",
                    scale=3
                )
                load_folder_btn = gr.Button("📁 Load Images", scale=1, variant="primary")

            init_sam_btn = gr.Button("⚡ Initialize SAM Model", variant="primary", size="lg")

            init_status = gr.Textbox(label="Status", interactive=False)

        # ====================================================================
        # Image Navigation Section
        # ====================================================================

        with gr.Group():
            gr.Markdown("## 📸 Image Processing")

            current_image_info = gr.Textbox(
                label="Current Image",
                value="No image loaded",
                interactive=False
            )

            with gr.Row():
                prev_btn = gr.Button("◄ Previous")
                jump_input = gr.Number(label="Jump to Image #", value=1, precision=0)
                jump_btn = gr.Button("Jump")
                next_btn = gr.Button("Next ►")

            original_image = gr.Image(label="Current Image", type="numpy")

        # ====================================================================
        # Scale Detection Section
        # ====================================================================

        with gr.Group():
            gr.Markdown("### 📏 Scale Detection")

            with gr.Row():
                scale_mode = gr.Radio(
                    choices=["Auto", "Manual"],
                    value="Auto",
                    label="Detection Mode"
                )
                detect_scale_btn = gr.Button("🔍 Detect Scale", variant="primary")

            with gr.Row():
                scale_detection_output = gr.Textbox(
                    label="Detection Result",
                    interactive=False
                )
                manual_scale_input = gr.Textbox(
                    label="Manual Scale (nm)",
                    placeholder="500",
                    visible=False
                )
                set_manual_btn = gr.Button("Set Manual Scale", visible=False)

            # Show/hide manual input based on mode
            def toggle_manual(mode):
                return {
                    manual_scale_input: gr.update(visible=(mode == "Manual")),
                    set_manual_btn: gr.update(visible=(mode == "Manual"))
                }

            scale_mode.change(
                toggle_manual,
                inputs=[scale_mode],
                outputs=[manual_scale_input, set_manual_btn]
            )

        # ====================================================================
        # Segmentation Section
        # ====================================================================

        with gr.Group():
            gr.Markdown("### 🎯 Particle Segmentation")

            segment_btn = gr.Button("🤖 Segment with SAM", variant="primary", size="lg")

            segment_status = gr.Textbox(label="Segmentation Status", interactive=False)

            mask_visualization = gr.Image(label="Mask Candidates")

            gr.Markdown("#### Select Best Mask")

            with gr.Row():
                mask_selection = gr.Radio(
                    choices=["Mask 1", "Mask 2", "Mask 3"],
                    value="Mask 1",
                    label="Choose Mask"
                )
                select_mask_btn = gr.Button("✓ Select Mask", variant="primary")

            mask_select_status = gr.Textbox(label="Selection Status", interactive=False)

        # ====================================================================
        # Analysis Section
        # ====================================================================

        with gr.Group():
            gr.Markdown("### 📊 Particle Analysis")

            analyze_btn = gr.Button("🔬 Analyze Particles", variant="primary", size="lg")

            analysis_status = gr.Textbox(label="Analysis Status", interactive=False)

            particle_visualization = gr.Image(label="Detected Particles")

            with gr.Tabs():
                with gr.Tab("Particle Measurements"):
                    results_table = gr.Dataframe(
                        label="Individual Particle Data",
                        wrap=True
                    )

                with gr.Tab("Summary Statistics"):
                    summary_stats_table = gr.Dataframe(
                        label="Statistical Summary"
                    )

        # ====================================================================
        # Session Summary Section
        # ====================================================================

        with gr.Group():
            gr.Markdown("## 📈 Session Summary")

            refresh_summary_btn = gr.Button("🔄 Refresh Summary")

            summary_text = gr.Markdown("No results yet")

            session_results_table = gr.Dataframe(
                label="All Processed Images"
            )

            with gr.Row():
                export_btn = gr.Button("💾 Export Results CSV", variant="primary")
                export_status = gr.Textbox(label="Export Status", interactive=False)
                export_file = gr.File(label="Download Results")

        # ====================================================================
        # Event Handlers
        # ====================================================================

        # Configuration
        init_sam_btn.click(
            initialize_sam,
            inputs=[sam_checkpoint, model_type],
            outputs=[init_status]
        )

        load_folder_btn.click(
            load_image_folder,
            inputs=[image_folder],
            outputs=[init_status, original_image, current_image_info]
        )

        # Navigation
        prev_btn.click(
            lambda: navigate_to_image('prev'),
            outputs=[
                original_image,
                current_image_info,
                scale_detection_output,
                manual_scale_input,
                mask_visualization,
                particle_visualization,
                results_table
            ]
        )

        next_btn.click(
            lambda: navigate_to_image('next'),
            outputs=[
                original_image,
                current_image_info,
                scale_detection_output,
                manual_scale_input,
                mask_visualization,
                particle_visualization,
                results_table
            ]
        )

        jump_btn.click(
            lambda idx: navigate_to_image('jump', idx),
            inputs=[jump_input],
            outputs=[
                original_image,
                current_image_info,
                scale_detection_output,
                manual_scale_input,
                mask_visualization,
                particle_visualization,
                results_table
            ]
        )

        # Scale detection
        detect_scale_btn.click(
            detect_scale_auto,
            outputs=[scale_detection_output, manual_scale_input]
        )

        set_manual_btn.click(
            set_scale_manual,
            inputs=[manual_scale_input],
            outputs=[scale_detection_output]
        )

        # Segmentation
        segment_btn.click(
            segment_with_sam,
            outputs=[mask_visualization, segment_status]
        )

        select_mask_btn.click(
            select_mask,
            inputs=[mask_selection],
            outputs=[mask_select_status]
        )

        # Analysis
        analyze_btn.click(
            analyze_particles,
            outputs=[
                particle_visualization,
                results_table,
                summary_stats_table,
                analysis_status
            ]
        )

        # Session summary
        refresh_summary_btn.click(
            get_session_summary,
            outputs=[session_results_table, summary_text]
        )

        export_btn.click(
            export_session_results,
            outputs=[export_file, export_status]
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
