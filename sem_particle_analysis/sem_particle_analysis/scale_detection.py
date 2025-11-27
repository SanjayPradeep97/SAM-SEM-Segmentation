"""
Scale Bar Detection and Calibration

Automatically detects scale bars in SEM/TEM images and calculates
pixel-to-nanometer conversion factors.
"""

import re
import cv2
import numpy as np
import easyocr


class ScaleDetector:
    """
    Detects and measures scale bars in microscopy images.

    Attributes:
        reader (easyocr.Reader): OCR reader instance
        last_detection (dict): Results from most recent detection
    """

    def __init__(self, use_gpu=False):
        """
        Initialize the scale bar detector.

        Args:
            use_gpu (bool): Whether to use GPU for OCR (default: False)
        """
        print("Initializing OCR reader...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        self.last_detection = None
        print("OCR reader initialized")

    def detect_scale_bar(self, image, region_width=0.5, region_height=0.06,
                        vertical_offset=0.0, threshold=250):
        """
        Detect and measure the scale bar in an image.

        Args:
            image (np.ndarray): RGB image array
            region_width (float): Width fraction of image to search (0-1)
            region_height (float): Height fraction of image to search (0-1)
            vertical_offset (float): Vertical offset from bottom (0-1)
            threshold (int): Binary threshold value (0-255)

        Returns:
            dict: Detection results containing:
                - 'pixel_length': Length of scale bar in pixels
                - 'scale_nm': Physical length in nanometers
                - 'conversion': Conversion factor (nm/pixel)
                - 'ocr_text': Raw OCR text
                - 'region': Region coordinates (x0, y0, width, height)
                - 'line_coords': Scale bar line coordinates
                - 'binary_image': Thresholded image

        Raises:
            ValueError: If scale bar cannot be detected or OCR fails
        """
        H, W = image.shape[:2]

        # Calculate search region
        box_w = int(W * region_width)
        box_h = int(H * region_height)
        y_offset = int(H * vertical_offset)
        x0 = W - box_w
        y0 = H - box_h - y_offset

        # Ensure coordinates are within bounds
        x0 = max(0, min(x0, W - box_w))
        y0 = max(0, min(y0, H - box_h))

        # Extract region
        crop = image[y0:y0 + box_h, x0:x0 + box_w]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        # Apply threshold
        _, binary255 = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        binary = (binary255 > 0).astype(np.uint8)

        if not binary.any():
            raise ValueError(
                f"No white pixels found with threshold {threshold}. "
                "Try lowering the threshold."
            )

        # Find scale bar line (topmost row of white pixels)
        coords = np.column_stack(np.where(binary))
        top_row = coords[:, 0].min()
        cols_on_top = coords[coords[:, 0] == top_row][:, 1]
        leftmost = int(cols_on_top.min())
        rightmost = int(cols_on_top.max())
        pixel_length = rightmost - leftmost

        # Perform OCR
        ocr_pairs = self.reader.readtext(crop)
        ocr_text = " ".join(txt for _, txt, _ in ocr_pairs)

        # Extract scale value and units
        scale_nm = self._parse_scale_text(ocr_text)

        # Calculate conversion factor
        conversion = scale_nm / pixel_length

        # Store results
        self.last_detection = {
            'region': (x0, y0, box_w, box_h),
            'pixel_length': pixel_length,
            'scale_nm': scale_nm,
            'conversion': conversion,
            'ocr_text': ocr_text,
            'binary_image': binary255,
            'line_coords': (leftmost, rightmost, top_row),
            'threshold': threshold
        }

        return self.last_detection

    def _parse_scale_text(self, ocr_text):
        """
        Extract scale value and unit from OCR text.

        Args:
            ocr_text (str): Raw OCR text

        Returns:
            float: Scale length in nanometers

        Raises:
            ValueError: If scale information cannot be extracted
        """
        # Match patterns like "100 nm", "1 μm", "500nm", etc.
        pattern = r'(\d+(?:\.\d+)?)\s*(nm|um|µm|uum|pm)'
        match = re.search(pattern, ocr_text, flags=re.I)

        if not match:
            raise ValueError(
                f"Could not extract scale information from OCR text: '{ocr_text}'"
            )

        value = float(match.group(1))
        unit = match.group(2).lower()

        # Convert to nanometers
        if unit in {'um', 'µm', 'uum', 'pm'}:
            return value * 1000.0  # micrometers to nanometers
        else:  # nm
            return value

    def crop_scale_bar(self, image, crop_percent=7.0):
        """
        Crop the bottom portion of an image to remove the scale bar.

        Args:
            image (np.ndarray): Input image
            crop_percent (float): Percentage to crop from bottom

        Returns:
            np.ndarray: Cropped image
        """
        height = image.shape[0]
        crop_height = int(height * (1 - crop_percent / 100))
        return image[:crop_height, :].copy()

    def get_conversion_factor(self):
        """
        Get the conversion factor from the last detection.

        Returns:
            float: Conversion factor (nm/pixel)

        Raises:
            RuntimeError: If no detection has been performed
        """
        if self.last_detection is None:
            raise RuntimeError("No scale bar has been detected yet")
        return self.last_detection['conversion']

    def set_manual_scale(self, scale_nm, pixel_length):
        """
        Manually set the scale calibration.

        Args:
            scale_nm (float): Physical length in nanometers
            pixel_length (int): Length in pixels

        Returns:
            float: Calculated conversion factor (nm/pixel)

        Raises:
            ValueError: If values are not positive
        """
        if scale_nm <= 0 or pixel_length <= 0:
            raise ValueError("Scale and pixel length must be positive values")

        conversion = scale_nm / pixel_length

        self.last_detection = {
            'pixel_length': pixel_length,
            'scale_nm': scale_nm,
            'conversion': conversion,
            'ocr_text': 'Manual entry',
            'manual': True
        }

        return conversion
