"""
Scale Bar Detection and Calibration

Automatically detects scale bars in SEM/TEM images and calculates
pixel-to-nanometer conversion factors.

Supports two detection methods:
1. TIFF Metadata Extraction (primary) - Extracts pixel size from microscope metadata
2. OCR-based Scale Bar Detection (fallback) - Reads scale bar text using OCR
"""

import os
import re
import logging
import cv2
import numpy as np
import easyocr

from .utils import extract_tiff_metadata

logger = logging.getLogger(__name__)


class ScaleDetector:
    """
    Detects and measures scale bars in microscopy images.

    Supports two detection methods:
    1. TIFF metadata extraction (primary) - reads pixel size from image metadata
    2. OCR-based scale bar detection (fallback) - reads scale bar text

    Attributes:
        reader (easyocr.Reader): OCR reader instance
        last_detection (dict): Results from most recent detection
    """

    # Unit conversion factors to nanometers
    UNIT_TO_NM = {
        'nm': 1.0,
        'um': 1000.0,
        'µm': 1000.0,
        'μm': 1000.0,  # Different mu character
        'uum': 1000.0,  # Common OCR misread
        'mm': 1_000_000.0,
        'cm': 10_000_000.0,
        'm': 1_000_000_000.0,
        'a': 0.1,  # Angstroms
        'å': 0.1,  # Angstrom symbol
        'angstrom': 0.1,
        'angstroms': 0.1,
        'pm': 0.001,  # Picometers
    }

    # Manufacturer detection patterns for ImageDescription parsing
    MANUFACTURER_PATTERNS = {
        'fei': [
            r'\[User\]',
            r'PixelWidth\s*=',
            r'PixelHeight\s*=',
            r'Thermo\s*Fisher',
            r'FEI\s*Company',
            r'\[Beam\]',
            r'\[Scan\]',
        ],
        'zeiss': [
            r'SmartSEM',
            r'AP_PIXEL_SIZE',
            r'AP_WD',
            r'Zeiss',
            r'\bLEO\b',
            r'SUPRA',
            r'SIGMA',
            r'GeminiSEM',
        ],
        'jeol': [
            r'JEOL',
            r'JEM-\d+',
            r'JSM-\d+',
            r'JIB-\d+',
        ],
        'hitachi': [
            r'Hitachi',
            r'SU\d{4}',
            r'S-\d{4}',
            r'TM\d{4}',
            r'Regulus',
            r'FlexSEM',
        ],
        'tescan': [
            r'TESCAN',
            r'MIRA',
            r'VEGA',
            r'CLARA',
        ],
    }

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

    def detect_scale(self, image, file_path=None, method='auto',
                     region_width=0.5, region_height=0.06,
                     vertical_offset=0.0, threshold=250):
        """
        Detect scale using the best available method.

        This is the main entry point for scale detection. It tries TIFF metadata
        extraction first (if file_path is provided and is a TIFF), then falls
        back to OCR-based scale bar detection.

        Args:
            image (np.ndarray): RGB image array (required for OCR fallback)
            file_path (str, optional): Path to image file for metadata extraction
            method (str): Detection method - 'auto', 'metadata', or 'ocr'
                - 'auto': Try metadata first, fall back to OCR (default)
                - 'metadata': Only use metadata extraction (raises if fails)
                - 'ocr': Only use OCR detection
            region_width (float): Width fraction for OCR search region (0-1)
            region_height (float): Height fraction for OCR search region (0-1)
            vertical_offset (float): Vertical offset from bottom for OCR (0-1)
            threshold (int): Binary threshold for OCR (0-255)

        Returns:
            dict: Detection results containing:
                - 'conversion': Conversion factor (nm/pixel)
                - 'method': Detection method used ('metadata' or 'ocr')
                - 'scale_nm': Physical scale in nanometers
                - 'pixel_length': Pixel length (1 for metadata, detected for OCR)
                - Additional method-specific fields

        Raises:
            ValueError: If scale cannot be detected by specified method
        """
        # Try metadata extraction first (if applicable)
        if method in ('auto', 'metadata') and file_path:
            if self._is_tiff(file_path):
                try:
                    logger.info(f"Attempting TIFF metadata extraction from: {file_path}")
                    result = self.detect_scale_from_metadata(file_path)
                    result['method'] = 'metadata'
                    self.last_detection = result
                    logger.info(f"Metadata extraction successful: {result['conversion']:.4f} nm/pixel")
                    return result
                except ValueError as e:
                    logger.warning(f"Metadata extraction failed: {e}")
                    if method == 'metadata':
                        raise
                    # Fall through to OCR for 'auto' mode

        # OCR-based detection
        if method in ('auto', 'ocr'):
            logger.info("Using OCR-based scale bar detection")
            result = self.detect_scale_bar(
                image,
                region_width=region_width,
                region_height=region_height,
                vertical_offset=vertical_offset,
                threshold=threshold
            )
            result['method'] = 'ocr'
            return result

        raise ValueError(f"Could not detect scale using method '{method}'")

    def _is_tiff(self, file_path):
        """Check if file is a TIFF based on extension."""
        if not file_path:
            return False
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ('.tif', '.tiff')

    def detect_scale_from_metadata(self, file_path):
        """
        Extract scale information from TIFF metadata.

        Supports multiple SEM microscope manufacturers including FEI/Thermo Fisher,
        Zeiss, JEOL, Hitachi, and TESCAN. Also handles generic TIFF resolution tags.

        Args:
            file_path (str): Path to TIFF image file

        Returns:
            dict: Detection results containing:
                - 'pixel_size_nm': Pixel size in nanometers
                - 'conversion': Conversion factor (nm/pixel), same as pixel_size_nm
                - 'scale_nm': Reference scale (same as pixel_size_nm for single pixel)
                - 'pixel_length': Always 1 (representing one pixel)
                - 'manufacturer': Detected manufacturer name
                - 'confidence': 'high', 'medium', or 'low'
                - 'metadata_source': Description of where scale was found
                - 'raw_metadata': Original metadata dict for debugging

        Raises:
            ValueError: If no scale information found in metadata
        """
        # Extract metadata
        metadata = extract_tiff_metadata(file_path)

        if not metadata['is_tiff']:
            raise ValueError("Not a valid TIFF file")

        image_desc = metadata.get('image_description', '') or ''

        # Detect manufacturer
        manufacturer = self._detect_manufacturer(image_desc, metadata.get('software', ''))
        logger.debug(f"Detected manufacturer: {manufacturer}")

        pixel_size_nm = None
        metadata_source = None
        confidence = 'low'

        # Try manufacturer-specific parsing
        if manufacturer == 'fei':
            pixel_size_nm = self._parse_fei_metadata(image_desc, metadata['raw_tags'])
            if pixel_size_nm:
                metadata_source = 'FEI ImageDescription (PixelWidth)'
                confidence = 'high'

        elif manufacturer == 'zeiss':
            pixel_size_nm = self._parse_zeiss_metadata(image_desc, metadata['raw_tags'])
            if pixel_size_nm:
                metadata_source = 'Zeiss SmartSEM (AP_PIXEL_SIZE)'
                confidence = 'high'

        elif manufacturer == 'jeol':
            pixel_size_nm = self._parse_jeol_metadata(image_desc, metadata['raw_tags'])
            if pixel_size_nm:
                metadata_source = 'JEOL metadata'
                confidence = 'medium'

        elif manufacturer == 'hitachi':
            pixel_size_nm = self._parse_hitachi_metadata(image_desc, metadata['raw_tags'])
            if pixel_size_nm:
                metadata_source = 'Hitachi metadata'
                confidence = 'medium'

        elif manufacturer == 'tescan':
            pixel_size_nm = self._parse_tescan_metadata(image_desc, metadata['raw_tags'])
            if pixel_size_nm:
                metadata_source = 'TESCAN metadata'
                confidence = 'medium'

        # Try generic resolution tags as fallback
        if pixel_size_nm is None:
            pixel_size_nm = self._parse_generic_resolution(metadata)
            if pixel_size_nm:
                metadata_source = 'TIFF XResolution/YResolution tags'
                confidence = 'medium'
                if manufacturer == 'unknown':
                    confidence = 'low'

        # Try generic pixel size patterns in ImageDescription
        if pixel_size_nm is None and image_desc:
            pixel_size_nm = self._parse_generic_pixel_size(image_desc)
            if pixel_size_nm:
                metadata_source = 'ImageDescription pixel size pattern'
                confidence = 'medium'

        # Try tag 34118 (Hitachi format) even if manufacturer wasn't detected
        if pixel_size_nm is None and 34118 in metadata['raw_tags']:
            pixel_size_nm = self._parse_hitachi_metadata(None, metadata['raw_tags'])
            if pixel_size_nm:
                metadata_source = 'TIFF tag 34118 (Hitachi format)'
                confidence = 'high'
                manufacturer = 'hitachi'

        if pixel_size_nm is None:
            raise ValueError(
                f"No scale information found in TIFF metadata. "
                f"Detected manufacturer: {manufacturer}"
            )

        result = {
            'pixel_size_nm': pixel_size_nm,
            'conversion': pixel_size_nm,
            'scale_nm': pixel_size_nm,
            'pixel_length': 1,
            'manufacturer': manufacturer,
            'confidence': confidence,
            'metadata_source': metadata_source,
            'raw_metadata': metadata
        }

        self.last_detection = result
        return result

    def _detect_manufacturer(self, image_desc, software):
        """
        Detect microscope manufacturer from metadata strings.

        Args:
            image_desc (str): ImageDescription tag content
            software (str): Software tag content

        Returns:
            str: Manufacturer name ('fei', 'zeiss', 'jeol', 'hitachi', 'tescan', or 'unknown')
        """
        combined = f"{image_desc} {software or ''}"

        for manufacturer, patterns in self.MANUFACTURER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return manufacturer

        return 'unknown'

    def _parse_fei_metadata(self, image_desc, raw_tags):
        """
        Parse FEI/Thermo Fisher SEM metadata.

        FEI instruments store pixel size in ImageDescription as key=value pairs:
            [Scan]
            PixelWidth=9.765625e-09
            PixelHeight=9.765625e-09

        Values are in meters.

        Args:
            image_desc (str): ImageDescription tag content
            raw_tags (dict): All TIFF tags

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        if not image_desc:
            return None

        # Look for PixelWidth or PixelHeight
        patterns = [
            r'PixelWidth\s*=\s*([0-9.eE+-]+)',
            r'PixelHeight\s*=\s*([0-9.eE+-]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, image_desc)
            if match:
                try:
                    # Value is in meters, convert to nanometers
                    value_m = float(match.group(1))
                    return value_m * 1e9
                except ValueError:
                    continue

        return None

    def _parse_zeiss_metadata(self, image_desc, raw_tags):
        """
        Parse Zeiss SEM metadata.

        Zeiss SmartSEM stores pixel size in ImageDescription:
            AP_PIXEL_SIZE = 3.906250e-009

        Values are in meters.

        Args:
            image_desc (str): ImageDescription tag content
            raw_tags (dict): All TIFF tags

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        if not image_desc:
            return None

        # Look for AP_PIXEL_SIZE
        patterns = [
            r'AP_PIXEL_SIZE\s*=\s*([0-9.eE+-]+)',
            r'Pixel\s*Size\s*=\s*([0-9.eE+-]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, image_desc, re.IGNORECASE)
            if match:
                try:
                    # Value is in meters, convert to nanometers
                    value_m = float(match.group(1))
                    return value_m * 1e9
                except ValueError:
                    continue

        return None

    def _parse_jeol_metadata(self, image_desc, raw_tags):
        """
        Parse JEOL SEM/TEM metadata.

        JEOL instruments may store pixel size in various formats.

        Args:
            image_desc (str): ImageDescription tag content
            raw_tags (dict): All TIFF tags

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        if not image_desc:
            return None

        # Look for common JEOL patterns
        patterns = [
            r'PixelSize\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|m)',
            r'Pixel\s*Size\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|m)',
            r'Resolution\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|m)/pixel',
        ]

        for pattern in patterns:
            match = re.search(pattern, image_desc, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).lower()

                    # Convert to nanometers
                    if unit in ('m',):
                        return value * 1e9
                    elif unit in ('um', 'µm'):
                        return value * 1000
                    else:  # nm
                        return value
                except (ValueError, IndexError):
                    continue

        return None

    def _parse_hitachi_metadata(self, image_desc, raw_tags):
        """
        Parse Hitachi SEM metadata.

        Hitachi instruments often store pixel size in TIFF tag 34118 as a dict
        with a tuple containing calibration data. The pixel size (in meters)
        is typically at index 3 of the tuple.

        Format of tag 34118:
            {'': (0, 0, 0, pixel_size_m, magnification, ...)}

        Args:
            image_desc (str): ImageDescription tag content
            raw_tags (dict): All TIFF tags

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        # First try tag 34118 (Hitachi SEM metadata)
        if 34118 in raw_tags:
            tag_value = raw_tags[34118]
            if isinstance(tag_value, dict) and '' in tag_value:
                data_tuple = tag_value['']
                if isinstance(data_tuple, (tuple, list)) and len(data_tuple) >= 4:
                    try:
                        # Pixel size is at index 3, in meters
                        pixel_size_m = float(data_tuple[3])
                        if pixel_size_m > 0:
                            # Convert meters to nanometers
                            return pixel_size_m * 1e9
                    except (ValueError, TypeError, IndexError):
                        pass

        # Fallback to ImageDescription patterns
        if image_desc:
            patterns = [
                r'PixelSize\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|m)',
                r'Pixel\s*Size\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|m)',
            ]

            for pattern in patterns:
                match = re.search(pattern, image_desc, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        unit = match.group(2).lower()

                        if unit in ('m',):
                            return value * 1e9
                        elif unit in ('um', 'µm'):
                            return value * 1000
                        else:
                            return value
                    except (ValueError, IndexError):
                        continue

        return None

    def _parse_tescan_metadata(self, image_desc, raw_tags):
        """
        Parse TESCAN SEM metadata.

        Args:
            image_desc (str): ImageDescription tag content
            raw_tags (dict): All TIFF tags

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        if not image_desc:
            return None

        # Look for TESCAN patterns
        patterns = [
            r'PixelSize\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|m)',
            r'Pixel\s*Width\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|m)',
        ]

        for pattern in patterns:
            match = re.search(pattern, image_desc, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).lower()

                    if unit in ('m',):
                        return value * 1e9
                    elif unit in ('um', 'µm'):
                        return value * 1000
                    else:
                        return value
                except (ValueError, IndexError):
                    continue

        return None

    def _parse_generic_resolution(self, metadata):
        """
        Parse standard TIFF XResolution/YResolution tags.

        These tags specify pixels per unit (where unit is defined by ResolutionUnit).
        ResolutionUnit: 1=none, 2=inch, 3=centimeter

        Args:
            metadata (dict): Metadata dict from extract_tiff_metadata()

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        x_res = metadata.get('x_resolution')
        y_res = metadata.get('y_resolution')
        res_unit = metadata.get('resolution_unit')

        # Need at least one resolution value
        resolution = x_res or y_res
        if resolution is None or resolution <= 0:
            return None

        # ResolutionUnit: 2=inch, 3=centimeter
        # pixels_per_unit is the resolution value
        # pixel_size = 1 / pixels_per_unit (in that unit)

        if res_unit == 2:  # Inch
            # resolution = pixels per inch
            # pixel_size_inch = 1 / resolution
            # pixel_size_nm = pixel_size_inch * 25.4e6 (25.4 mm/inch, 1e6 nm/mm)
            pixel_size_nm = 25.4e6 / resolution
            return pixel_size_nm

        elif res_unit == 3:  # Centimeter
            # resolution = pixels per cm
            # pixel_size_cm = 1 / resolution
            # pixel_size_nm = pixel_size_cm * 1e7 (1e7 nm/cm)
            pixel_size_nm = 1e7 / resolution
            return pixel_size_nm

        # Unit 1 (none) or unknown - can't determine scale
        return None

    def _parse_generic_pixel_size(self, image_desc):
        """
        Try to find pixel size from generic patterns in ImageDescription.

        Looks for patterns like:
        - "pixel size = 10 nm"
        - "pixelsize: 5.5 um"
        - "10 nm/pixel"

        Args:
            image_desc (str): ImageDescription tag content

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        if not image_desc:
            return None

        patterns = [
            # "pixel size = 10 nm" or "pixelsize: 5.5 um"
            r'pixel\s*size\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|mm|m|pm)',
            # "10 nm/pixel" or "5.5 um / pixel"
            r'([0-9.eE+-]+)\s*(nm|um|µm|mm|m|pm)\s*/\s*pixel',
            # "resolution: 10 nm"
            r'resolution\s*[:=]\s*([0-9.eE+-]+)\s*(nm|um|µm|mm|m|pm)',
        ]

        for pattern in patterns:
            match = re.search(pattern, image_desc, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).lower()

                    # Convert to nanometers using our conversion table
                    conversion = self.UNIT_TO_NM.get(unit, 1.0)
                    return value * conversion
                except (ValueError, IndexError):
                    continue

        return None

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

        Supports units: nm, um/µm, mm, cm, m, pm, angstroms (A/Å)

        Args:
            ocr_text (str): Raw OCR text

        Returns:
            float: Scale length in nanometers

        Raises:
            ValueError: If scale information cannot be extracted
        """
        # Match patterns like "100 nm", "1 μm", "500nm", "10 A", etc.
        # Extended pattern to include more units
        pattern = r'(\d+(?:\.\d+)?)\s*(nm|um|µm|μm|uum|mm|cm|m|pm|a|å|angstrom|angstroms)'
        match = re.search(pattern, ocr_text, flags=re.I)

        if not match:
            raise ValueError(
                f"Could not extract scale information from OCR text: '{ocr_text}'"
            )

        value = float(match.group(1))
        unit = match.group(2).lower()

        # Convert to nanometers using class conversion table
        conversion = self.UNIT_TO_NM.get(unit, 1.0)
        return value * conversion

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
