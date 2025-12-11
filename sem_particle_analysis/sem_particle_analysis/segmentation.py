"""
Particle Segmentation using SAM

Handles SAM-based particle segmentation with mask generation and selection.
"""

import numpy as np
from skimage import morphology


class ParticleSegmenter:
    """
    Segments particles in microscopy images using SAM.

    Attributes:
        sam_model: SAM model instance
        masks: Generated mask candidates
        scores: Confidence scores for each mask
        selected_mask: Currently selected mask
    """

    def __init__(self, sam_model):
        """
        Initialize the particle segmenter.

        Args:
            sam_model: SAMModel instance
        """
        self.sam_model = sam_model
        self.masks = None
        self.scores = None
        self.selected_mask = None
        self.selected_mask_index = None
        self.current_encoded_image = None  # Track if image is already encoded

    def encode_image(self, image):
        """
        Encode image features using SAM encoder (expensive operation).
        Call this once per image, then use refine_with_sam with image_already_encoded=True.

        Args:
            image (np.ndarray): RGB image array (H, W, 3)
        """
        print("Encoding image with SAM...")
        self.sam_model.set_image(image)
        self.current_encoded_image = image
        print("Image encoding complete")

    def segment_image(self, image, multimask_output=True):
        """
        Segment particles in an image using SAM.

        This uses a full-image bounding box as the prompt to SAM.

        Args:
            image (np.ndarray): RGB image array (H, W, 3)
            multimask_output (bool): Whether to generate multiple mask candidates

        Returns:
            tuple: (masks, scores)
                - masks: Array of boolean masks (num_masks, H, W)
                - scores: Confidence scores for each mask
        """
        # Set image in SAM predictor if not already encoded
        if self.current_encoded_image is None or not np.array_equal(self.current_encoded_image, image):
            print("Running SAM image encoder...")
            self.sam_model.set_image(image)
            self.current_encoded_image = image
            print("Encoder complete. Generating masks...")
        else:
            print("Using pre-encoded image. Generating masks...")

        # Use full-image bounding box as prompt
        height, width = image.shape[:2]
        full_box = np.array([[0, 0, width, height]])

        # Generate masks
        self.masks, self.scores, _ = self.sam_model.predict(
            box=full_box,
            multimask_output=multimask_output
        )

        # Convert to boolean
        self.masks = self.masks.astype(bool)

        print(f"Generated {len(self.masks)} mask candidates")
        print(f"Confidence scores: {self.scores}")

        return self.masks, self.scores

    def select_mask(self, mask_index=None):
        """
        Select a specific mask or automatically choose the best one.

        Args:
            mask_index (int, optional): Index of mask to select.
                If None, selects the mask with highest confidence.

        Returns:
            np.ndarray: Selected boolean mask

        Raises:
            RuntimeError: If no masks have been generated
            IndexError: If mask_index is out of range
        """
        if self.masks is None:
            raise RuntimeError("No masks available. Run segment_image() first.")

        if mask_index is None:
            # Select best mask by score
            mask_index = int(np.argmax(self.scores))
            print(f"Auto-selected mask {mask_index} (score: {self.scores[mask_index]:.3f})")
        else:
            if mask_index < 0 or mask_index >= len(self.masks):
                raise IndexError(
                    f"Mask index {mask_index} out of range [0, {len(self.masks)-1}]"
                )
            print(f"Selected mask {mask_index} (score: {self.scores[mask_index]:.3f})")

        self.selected_mask_index = mask_index
        self.selected_mask = self.masks[mask_index]

        return self.selected_mask

    def get_binary_mask(self, invert=True):
        """
        Get the selected mask as a binary image.

        Args:
            invert (bool): If True, particles are 1 and background is 0.
                          If False, particles are 0 and background is 1.

        Returns:
            np.ndarray: Binary mask (dtype: uint8)

        Raises:
            RuntimeError: If no mask has been selected
        """
        if self.selected_mask is None:
            raise RuntimeError("No mask selected. Run select_mask() first.")

        if invert:
            return (1 - self.selected_mask.astype(np.uint8))
        else:
            return self.selected_mask.astype(np.uint8)

    def refine_with_sam(self, image, point_coords, point_labels,
                       base_mask=None, multimask_output=True, image_already_encoded=False):
        """
        Refine segmentation using point prompts.

        Args:
            image (np.ndarray): RGB image
            point_coords (list or np.ndarray): Nx2 array of [x, y] coordinates
            point_labels (list or np.ndarray): N array of labels (1=positive, 0=negative)
            base_mask (np.ndarray, optional): Base mask for ROI calculation
            multimask_output (bool): Whether to return multiple candidates
            image_already_encoded (bool): If True, skip image encoding (assumes set_image was already called)

        Returns:
            tuple: (refined_mask, score)
                - refined_mask: Best refined mask
                - score: Confidence score
        """
        # Ensure image is set (only if not already encoded)
        if not image_already_encoded:
            self.sam_model.set_image(image)

        # Calculate ROI box from base mask if provided
        if base_mask is not None and base_mask.any():
            box = self._compute_roi_box(base_mask, pad=10)
        else:
            H, W = image.shape[:2]
            box = np.array([[0, 0, W, H]])

        # Convert points to numpy arrays
        pts = np.array(point_coords, dtype=float)
        labs = np.array(point_labels, dtype=int)

        # Clip coordinates to image bounds
        H, W = image.shape[:2]
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

        # Generate refined masks
        masks_out, scores, _ = self.sam_model.predict(
            point_coords=pts,
            point_labels=labs,
            box=box,
            multimask_output=multimask_output
        )

        # Select best mask (highest IoU with base mask if provided)
        if base_mask is not None:
            ious = [self._mask_iou(m, base_mask) for m in masks_out]
            best_idx = int(np.argmax(ious))
        else:
            best_idx = int(np.argmax(scores))

        refined = masks_out[best_idx].astype(bool)

        # Clean up
        refined = morphology.remove_small_objects(refined, min_size=20)

        return refined, scores[best_idx]

    def _compute_roi_box(self, mask, pad=10):
        """
        Compute a padded bounding box around a mask.

        Args:
            mask (np.ndarray): Boolean mask
            pad (int): Padding in pixels

        Returns:
            np.ndarray: Bounding box [[x0, y0, x1, y1]]
        """
        if not mask.any():
            H, W = mask.shape
            return np.array([[0, 0, W, H]])

        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        H, W = mask.shape
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(W - 1, x1 + pad)
        y1 = min(H - 1, y1 + pad)

        return np.array([[x0, y0, x1, y1]])

    def _mask_iou(self, mask1, mask2):
        """
        Calculate Intersection over Union between two masks.

        Args:
            mask1, mask2 (np.ndarray): Boolean masks

        Returns:
            float: IoU value (0-1)
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0
