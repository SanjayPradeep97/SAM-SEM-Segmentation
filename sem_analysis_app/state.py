"""
Application state.

One process-wide instance, shared by every callback. This mirrors the
original single-file design: the app is intended for one analyst working
locally, and two browser tabs pointed at the same server share this object.
"""

import json
import os
from pathlib import Path

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
        # Ranked candidates from ParticleSegmenter.rank_candidates, each carrying
        # the polarity that produced it so selection can't pick the wrong side.
        self.candidates = []
        # (x0, y0, w, h) of a scale bar printed inside the frame, excluded from
        # segmentation so it isn't measured as a particle.
        self.scale_bar_region = None
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
        # Ranked candidates from ParticleSegmenter.rank_candidates, each carrying
        # the polarity that produced it so selection can't pick the wrong side.
        self.candidates = []
        # (x0, y0, w, h) of a scale bar printed inside the frame, excluded from
        # segmentation so it isn't measured as a particle.
        self.scale_bar_region = None
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
