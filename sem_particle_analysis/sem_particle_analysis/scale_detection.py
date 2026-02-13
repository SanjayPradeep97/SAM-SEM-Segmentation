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

    # Valid pixel size range for SEM/TEM images (nm/pixel)
    # 0.01 nm = highest resolution TEM, 200,000 nm = lowest mag SEM
    VALID_PIXEL_SIZE_RANGE = (0.01, 200_000)

    # Registry for custom tag parsers (extensibility for new microscope formats)
    # Format: {tag_code: [(parser_func, manufacturer_name), ...]}
    _TAG_PARSERS = {}

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

    # Confidence scoring for method comparison
    CONFIDENCE_SCORES = {'high': 3, 'medium': 2, 'low': 1}

    @classmethod
    def register_tag_parser(cls, tag_code, manufacturer, parser_func):
        """
        Register a custom parser for a specific TIFF tag code.

        This allows adding support for new microscope formats without
        modifying the core ScaleDetector code. The parser function receives
        the tag value and should return pixel size in nanometers, or None.

        Args:
            tag_code (int): TIFF tag code to handle
            manufacturer (str): Manufacturer name for identification
            parser_func (callable): Function(tag_value) -> float (nm) or None

        Example:
            ScaleDetector.register_tag_parser(
                tag_code=50000,
                manufacturer='oxford',
                parser_func=lambda v: float(v.get('PixelSize', 0)) * 1e9
            )
        """
        if tag_code not in cls._TAG_PARSERS:
            cls._TAG_PARSERS[tag_code] = []
        cls._TAG_PARSERS[tag_code].append((parser_func, manufacturer))

    def __init__(self, use_gpu=False):
        """
        Initialize the scale bar detector.

        Args:
            use_gpu (bool): Whether to use GPU for OCR (default: False)
        """
        print("Initializing OCR reader...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        self.last_detection = None
        self.last_metadata_result = None
        self.last_ocr_result = None
        print("OCR reader initialized")

    def detect_scale(self, image, file_path=None, method='auto',
                     region_x=0.75, region_y=0.95,
                     region_width=0.5, region_height=0.08,
                     polarity='auto', threshold=200):
        """
        Detect scale using the best available method.

        This is the main entry point for scale detection. In 'auto' mode, it
        tries TIFF metadata first, then OCR. When both succeed, it cross-checks
        the results and selects the most reliable one based on confidence scoring.

        Args:
            image (np.ndarray): RGB image array (required for OCR fallback)
            file_path (str, optional): Path to image file for metadata extraction
            method (str): Detection method - 'auto', 'metadata', or 'ocr'
                - 'auto': Try both, compare confidence, cross-check (default)
                - 'metadata': Only use metadata extraction (raises if fails)
                - 'ocr': Only use OCR detection
            region_x (float): Horizontal center of OCR search region (0-1)
            region_y (float): Vertical center of OCR search region (0-1)
            region_width (float): Width fraction for OCR search region (0-1)
            region_height (float): Height fraction for OCR search region (0-1)
            polarity (str): Scale bar polarity - 'auto', 'bright', or 'dark'
            threshold (int): Binary threshold for OCR (0-255)

        Returns:
            dict: Detection results containing:
                - 'conversion': Conversion factor (nm/pixel)
                - 'method': Detection method used ('metadata' or 'ocr')
                - 'scale_nm': Physical scale in nanometers
                - 'pixel_length': Pixel length (1 for metadata, detected for OCR)
                - 'cross_check': Cross-check info if both methods succeeded
                - Additional method-specific fields

        Raises:
            ValueError: If scale cannot be detected by specified method
        """
        metadata_result = None
        ocr_result = None

        # Reset cross-check state
        self.last_metadata_result = None
        self.last_ocr_result = None

        # --- Try metadata extraction ---
        if method in ('auto', 'metadata') and file_path:
            if self._is_tiff(file_path):
                try:
                    logger.info(f"Attempting TIFF metadata extraction from: {file_path}")
                    metadata_result = self.detect_scale_from_metadata(file_path)
                    metadata_result['method'] = 'metadata'
                    self.last_metadata_result = metadata_result
                    logger.info(f"Metadata extraction successful: {metadata_result['conversion']:.4f} nm/pixel")
                except ValueError as e:
                    logger.warning(f"Metadata extraction failed: {e}")
                    if method == 'metadata':
                        raise

        # For 'metadata' only mode, return what we have
        if method == 'metadata':
            if metadata_result:
                self.last_detection = metadata_result
                return metadata_result
            raise ValueError("Metadata extraction failed and no fallback allowed")

        # --- Try OCR ---
        if method in ('auto', 'ocr'):
            try:
                logger.info("Attempting OCR-based scale bar detection")
                ocr_result = self.detect_scale_bar(
                    image,
                    region_x=region_x,
                    region_y=region_y,
                    region_width=region_width,
                    region_height=region_height,
                    polarity=polarity,
                    threshold=threshold
                )
                ocr_result['method'] = 'ocr'
                ocr_result['confidence'] = 'medium'  # OCR is always medium confidence
                self.last_ocr_result = ocr_result
                logger.info(f"OCR detection successful: {ocr_result['conversion']:.4f} nm/pixel")
            except ValueError as e:
                logger.warning(f"OCR detection failed: {e}")
                if method == 'ocr':
                    raise

        # --- Select best result and attach cross-check info ---
        chosen = self._select_best_result(metadata_result, ocr_result)

        if chosen is None:
            raise ValueError("Could not detect scale using any available method")

        self.last_detection = chosen
        return chosen

    def _select_best_result(self, metadata_result, ocr_result):
        """
        Select the best detection result and attach cross-check info.

        When both methods succeed, compares confidence scores and attaches
        cross-check information showing whether the two methods agree.

        Args:
            metadata_result (dict or None): Result from metadata extraction
            ocr_result (dict or None): Result from OCR detection

        Returns:
            dict or None: The best result with cross-check info attached
        """
        if metadata_result and ocr_result:
            # Cross-check: compare the two conversions
            meta_conv = metadata_result['conversion']
            ocr_conv = ocr_result['conversion']
            ratio = meta_conv / ocr_conv if ocr_conv > 0 else float('inf')
            agrees = 0.9 <= ratio <= 1.1  # Within 10%

            cross_check = {
                'metadata_conversion': meta_conv,
                'ocr_conversion': ocr_conv,
                'agrees': agrees,
                'ratio': ratio,
            }

            # Select based on confidence
            meta_score = self.CONFIDENCE_SCORES.get(
                metadata_result.get('confidence', 'low'), 0)
            ocr_score = self.CONFIDENCE_SCORES.get(
                ocr_result.get('confidence', 'low'), 0)

            if meta_score >= ocr_score:
                chosen = metadata_result
            else:
                chosen = ocr_result

            chosen['cross_check'] = cross_check
            return chosen

        # Only one result available
        return metadata_result or ocr_result

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
        manufacturer = self._detect_manufacturer(image_desc, metadata.get('software', ''), metadata['raw_tags'])
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

        # Try tag 34118 — could be Zeiss CZ_SEM or Hitachi
        if pixel_size_nm is None:
            tag_34118 = metadata['raw_tags'].get(34118) or metadata['raw_tags'].get('CZ_SEM')
            if tag_34118 is not None and isinstance(tag_34118, dict):
                # Check if it's Zeiss format (has 'ap_pixel_size' key)
                if 'ap_pixel_size' in tag_34118 or 'dp_sem' in tag_34118:
                    pixel_size_nm = self._parse_zeiss_cz_sem_tag(metadata['raw_tags'])
                    if pixel_size_nm:
                        metadata_source = 'Zeiss CZ_SEM tag (34118)'
                        confidence = 'high'
                        manufacturer = 'zeiss'
                # Check if it's Hitachi format (has '' key with tuple)
                elif '' in tag_34118:
                    pixel_size_nm = self._parse_hitachi_metadata(None, metadata['raw_tags'])
                    if pixel_size_nm:
                        metadata_source = 'Hitachi tag (34118)'
                        confidence = 'high'
                        manufacturer = 'hitachi'

        # Try tag 34682 — FEI HELIOS
        if pixel_size_nm is None:
            tag_34682 = metadata['raw_tags'].get(34682) or metadata['raw_tags'].get('FEI_HELIOS')
            if tag_34682 is not None:
                pixel_size_nm = self._parse_fei_helios_tag(metadata['raw_tags'])
                if pixel_size_nm:
                    metadata_source = 'FEI HELIOS tag (34682)'
                    confidence = 'high'
                    manufacturer = 'fei'

        # Try registered custom tag parsers
        if pixel_size_nm is None and self._TAG_PARSERS:
            for tag_code, parsers in self._TAG_PARSERS.items():
                tag_val = metadata['raw_tags'].get(tag_code)
                if tag_val is not None:
                    for parser_func, mfr_name in parsers:
                        try:
                            parsed = parser_func(tag_val)
                            if parsed and parsed > 0:
                                pixel_size_nm = parsed
                                manufacturer = mfr_name
                                metadata_source = f'Registered parser ({mfr_name}, tag {tag_code})'
                                confidence = 'medium'
                                break
                        except Exception as e:
                            logger.debug(f"Custom parser for {mfr_name} failed: {e}")
                            continue
                if pixel_size_nm:
                    break

        if pixel_size_nm is None:
            raise ValueError(
                f"No scale information found in TIFF metadata. "
                f"Detected manufacturer: {manufacturer}"
            )

        # Validate pixel size is in plausible range for SEM/TEM
        pixel_size_nm, validity, msg = self._validate_pixel_size(pixel_size_nm, metadata_source or 'metadata')
        if validity == 'invalid':
            logger.warning(f"Pixel size validation failed: {msg}")
            raise ValueError(f"Extracted pixel size is out of range: {msg}")

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

    def _detect_manufacturer(self, image_desc, software, raw_tags=None):
        """
        Detect microscope manufacturer from metadata strings and private tags.

        Args:
            image_desc (str): ImageDescription tag content
            software (str): Software tag content
            raw_tags (dict, optional): All TIFF tags for private tag detection

        Returns:
            str: Manufacturer name ('fei', 'zeiss', 'jeol', 'hitachi', 'tescan', or 'unknown')
        """
        combined = f"{image_desc} {software or ''}"

        for manufacturer, patterns in self.MANUFACTURER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return manufacturer

        # Also check for known private tags
        if raw_tags:
            # FEI HELIOS tag (34682)
            if raw_tags.get(34682) or raw_tags.get('FEI_HELIOS'):
                return 'fei'

            # Tag 34118 could be Zeiss CZ_SEM or Hitachi
            tag_34118 = raw_tags.get(34118) or raw_tags.get('CZ_SEM')
            if isinstance(tag_34118, dict):
                if 'ap_pixel_size' in tag_34118 or 'dp_sem' in tag_34118:
                    return 'zeiss'
                elif '' in tag_34118:
                    return 'hitachi'

        return 'unknown'

    def _parse_fei_helios_tag(self, raw_tags):
        """
        Parse FEI/Thermo Fisher HELIOS private tag (34682).

        FEI Teneo/Helios instruments store metadata in TIFF tag 34682 as a
        nested dict with sections like Scan, EBeam, Image. For example:
            {Scan: {PixelWidth: 1.34896e-06, PixelHeight: 1.34896e-06, ...},
             EBeam: {HFW: 0.002072, ...},
             Image: {ResolutionX: 1536, ResolutionY: 1024, ...}}

        Scan.PixelWidth is in meters.

        Args:
            raw_tags (dict): All TIFF tags (indexed by name or code)

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        tag_value = raw_tags.get(34682) or raw_tags.get('FEI_HELIOS')
        if not isinstance(tag_value, dict):
            return None

        # Primary: Scan.PixelWidth (in meters)
        scan = tag_value.get('Scan', {})
        if isinstance(scan, dict):
            for key in ('PixelWidth', 'PixelHeight'):
                pixel_val = scan.get(key)
                if pixel_val is not None:
                    try:
                        value_m = float(pixel_val)
                        if value_m > 0:
                            return value_m * 1e9  # meters to nm
                    except (ValueError, TypeError):
                        continue

        # Also check EScan section (some FEI instruments use this)
        escan = tag_value.get('EScan', {})
        if isinstance(escan, dict):
            for key in ('PixelWidth', 'PixelHeight'):
                pixel_val = escan.get(key)
                if pixel_val is not None:
                    try:
                        value_m = float(pixel_val)
                        if value_m > 0:
                            return value_m * 1e9
                    except (ValueError, TypeError):
                        continue

        # Fallback: compute from EBeam.HFW / Image.ResolutionX
        ebeam = tag_value.get('EBeam', {})
        image_info = tag_value.get('Image', {})
        if isinstance(ebeam, dict) and isinstance(image_info, dict):
            hfw = ebeam.get('HFW')
            res_x = image_info.get('ResolutionX')
            if hfw is not None and res_x is not None:
                try:
                    hfw_m = float(hfw)
                    width_pixels = int(res_x)
                    if hfw_m > 0 and width_pixels > 0:
                        return (hfw_m * 1e9) / width_pixels
                except (ValueError, TypeError):
                    pass

        return None

    def _parse_fei_metadata(self, image_desc, raw_tags):
        """
        Parse FEI/Thermo Fisher SEM metadata.

        Tries FEI_HELIOS private tag (34682) first, then falls back to
        ImageDescription text parsing (older FEI format).

        Args:
            image_desc (str): ImageDescription tag content
            raw_tags (dict): All TIFF tags

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        # First try FEI_HELIOS private tag (tag 34682)
        result = self._parse_fei_helios_tag(raw_tags)
        if result is not None:
            return result

        # Fall back to ImageDescription text parsing
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

    def _parse_zeiss_cz_sem_tag(self, raw_tags):
        """
        Parse Zeiss CZ_SEM private tag (34118).

        Zeiss instruments store metadata in TIFF tag 34118 as a dict with
        tuple values. For example:
            'ap_pixel_size': ('Pixel Size', 26.37, 'nm')
            'ap_image_pixel_size': ('Image Pixel Size', 26.37, 'nm')
            'ap_width': ('Width', 27.0, 'um')
            'dp_image_store': ('Store resolution', '1024 * 768')

        Args:
            raw_tags (dict): All TIFF tags (indexed by name or code)

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        tag_value = raw_tags.get(34118) or raw_tags.get('CZ_SEM')
        if not isinstance(tag_value, dict):
            return None

        # Primary: ap_pixel_size tuple -> ('Pixel Size', 26.37, 'nm')
        for key in ('ap_pixel_size', 'ap_image_pixel_size'):
            entry = tag_value.get(key)
            if isinstance(entry, (tuple, list)) and len(entry) >= 3:
                try:
                    value = float(entry[1])
                    unit = str(entry[2]).lower().strip()
                    conversion = self.UNIT_TO_NM.get(unit, None)
                    if conversion is not None and value > 0:
                        return value * conversion
                except (ValueError, TypeError):
                    continue

        # Fallback: compute from ap_width and image store resolution
        ap_width = tag_value.get('ap_width')
        dp_image_store = tag_value.get('dp_image_store')
        if ap_width and dp_image_store:
            try:
                width_val = float(ap_width[1])
                width_unit = str(ap_width[2]).lower().strip()
                store_str = str(dp_image_store[1])
                image_width = int(store_str.split('*')[0].strip())
                width_nm = width_val * self.UNIT_TO_NM.get(width_unit, 1.0)
                if image_width > 0 and width_nm > 0:
                    return width_nm / image_width
            except (ValueError, TypeError, IndexError):
                pass

        return None

    def _parse_zeiss_metadata(self, image_desc, raw_tags):
        """
        Parse Zeiss SEM metadata.

        Tries CZ_SEM private tag (34118) first, then falls back to
        ImageDescription text parsing (SmartSEM format).

        Args:
            image_desc (str): ImageDescription tag content
            raw_tags (dict): All TIFF tags

        Returns:
            float: Pixel size in nanometers, or None if not found
        """
        # First try CZ_SEM private tag (tag 34118)
        result = self._parse_zeiss_cz_sem_tag(raw_tags)
        if result is not None:
            return result

        # Fall back to ImageDescription text parsing
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

    # Standard display/scanner DPI values that are NOT SEM calibration
    STANDARD_DPI_VALUES = {72, 96, 150, 192, 200, 240, 300, 360, 600}

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

        # Reject standard display/scanner DPI values that are meaningless for SEM
        if res_unit in (2, 3) and resolution in self.STANDARD_DPI_VALUES:
            logger.info(f"Ignoring standard DPI value {resolution} (not SEM calibration)")
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

    def _validate_pixel_size(self, pixel_size_nm, source="unknown"):
        """
        Validate that a pixel size is in the physically plausible range for SEM/TEM.

        Args:
            pixel_size_nm (float): Pixel size in nanometers
            source (str): Description of where the value came from (for logging)

        Returns:
            tuple: (validated_value_or_None, 'valid'|'invalid', message)
        """
        if pixel_size_nm is None:
            return None, 'invalid', f'{source}: no value'
        if pixel_size_nm <= 0:
            return None, 'invalid', f'{source}: non-positive value {pixel_size_nm}'
        if pixel_size_nm < self.VALID_PIXEL_SIZE_RANGE[0]:
            return None, 'invalid', f'{source}: {pixel_size_nm:.6f} nm/px too small (sub-atomic)'
        if pixel_size_nm > self.VALID_PIXEL_SIZE_RANGE[1]:
            return None, 'invalid', f'{source}: {pixel_size_nm:.1f} nm/px too large (not SEM/TEM scale)'
        return pixel_size_nm, 'valid', f'{source}: {pixel_size_nm:.4f} nm/px within range'

    def detect_scale_bar(self, image, region_x=0.75, region_y=0.95,
                        region_width=0.5, region_height=0.08,
                        polarity='auto', threshold=200):
        """
        Detect and measure the scale bar in an image using adaptive polarity
        and morphological line detection.

        Supports scale bars of any color (bright on dark, dark on bright) at
        any position in the image. The search region is defined by its center
        (region_x, region_y) and size (region_width, region_height), all as
        fractions of image dimensions.

        Args:
            image (np.ndarray): RGB image array
            region_x (float): Horizontal center of search region (0-1, left to right)
            region_y (float): Vertical center of search region (0-1, top to bottom)
            region_width (float): Width of search region as fraction of image (0-1)
            region_height (float): Height of search region as fraction of image (0-1)
            polarity (str): Scale bar polarity:
                - 'auto': Try both bright and dark, pick the one with best line
                - 'bright': Bright bar on dark background (white scale bars)
                - 'dark': Dark bar on bright background (black scale bars)
            threshold (int): Binary threshold sensitivity (0-255)

        Returns:
            dict: Detection results containing:
                - 'pixel_length': Length of scale bar in pixels
                - 'scale_nm': Physical length in nanometers
                - 'conversion': Conversion factor (nm/pixel)
                - 'ocr_text': Raw OCR text
                - 'region': Region coordinates (x0, y0, width, height)
                - 'line_coords': Scale bar line coordinates (leftmost, rightmost, row)
                - 'binary_image': Thresholded image used for detection
                - 'polarity_used': Which polarity succeeded ('bright' or 'dark')

        Raises:
            ValueError: If scale bar cannot be detected or OCR fails
        """
        H, W = image.shape[:2]

        # Calculate search region from center-based coordinates
        box_w = max(1, int(W * region_width))
        box_h = max(1, int(H * region_height))
        cx = int(W * region_x)
        cy = int(H * region_y)
        x0 = max(0, min(cx - box_w // 2, W - box_w))
        y0 = max(0, min(cy - box_h // 2, H - box_h))

        # Extract region
        crop = image[y0:y0 + box_h, x0:x0 + box_w]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        # Determine which polarities to try
        if polarity == 'auto':
            polarities_to_try = ['bright', 'dark']
        else:
            polarities_to_try = [polarity]

        best_line = None
        best_binary = None
        best_polarity = None

        for pol in polarities_to_try:
            if pol == 'bright':
                _, binary255 = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            else:
                _, binary255 = cv2.threshold(gray, 255 - threshold, 255, cv2.THRESH_BINARY_INV)

            line = self._find_scale_line_morphological(binary255)
            if line is not None:
                leftmost, rightmost, row = line
                length = rightmost - leftmost
                if best_line is None or length > (best_line[1] - best_line[0]):
                    best_line = line
                    best_binary = binary255
                    best_polarity = pol

        if best_line is None:
            raise ValueError(
                f"No scale bar line found in region. "
                f"Try adjusting the region position, polarity, or threshold."
            )

        leftmost, rightmost, top_row = best_line
        pixel_length = rightmost - leftmost

        if pixel_length < 5:
            raise ValueError(
                f"Detected line too short ({pixel_length}px). "
                f"Adjust region to cover the scale bar."
            )

        # Perform OCR on the crop region
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
            'binary_image': best_binary,
            'line_coords': (leftmost, rightmost, top_row),
            'threshold': threshold,
            'polarity_used': best_polarity
        }

        return self.last_detection

    def _find_scale_line_morphological(self, binary255):
        """
        Find a horizontal scale bar line using morphological operations.

        Uses a wide horizontal kernel to isolate long horizontal features,
        then finds the longest connected horizontal segment.

        Args:
            binary255 (np.ndarray): Binary image (0 or 255), uint8

        Returns:
            tuple: (leftmost, rightmost, row) of the best line, or None
        """
        if not binary255.any():
            return None

        h, w = binary255.shape

        # Create a horizontal morphological kernel — width scaled to region
        # Minimum kernel width: 10% of crop width, ensures we only find long bars
        kernel_w = max(15, w // 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))

        # Opening: removes small noise, keeps long horizontal structures
        opened = cv2.morphologyEx(binary255, cv2.MORPH_OPEN, kernel)

        if not opened.any():
            return None

        # Find connected components in the morphologically opened image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)

        if num_labels <= 1:
            return None

        # Find the widest component (skip background label 0)
        best_label = None
        best_width = 0
        for label_id in range(1, num_labels):
            comp_w = stats[label_id, cv2.CC_STAT_WIDTH]
            comp_h = stats[label_id, cv2.CC_STAT_HEIGHT]
            # Scale bar lines are wide and thin (aspect ratio > 5:1)
            if comp_w > best_width and comp_w > comp_h * 3:
                best_width = comp_w
                best_label = label_id

        if best_label is None:
            return None

        # Get the row and column extents of the best component
        component_mask = (labels == best_label)
        rows, cols = np.where(component_mask)
        # Use the median row (most representative row of the bar)
        bar_row = int(np.median(rows))
        leftmost = int(cols.min())
        rightmost = int(cols.max())

        return (leftmost, rightmost, bar_row)

    # Known SEM instrument parameter patterns that should NOT be treated as scale
    # These are values like "8.8 mm" (working distance), "10.00 kV", etc.
    _INSTRUMENT_PARAM_PATTERNS = [
        r'WD\s*$',           # Working distance label right before the value
        r'^\s*\d+\.\d+\s*kV',  # Accelerating voltage
        r'^\s*\d+\s*pA',     # Beam current
        r'^\s*\d+\s*nA',     # Beam current
        r'^\s*\d+\s*ps',     # Dwell time
        r'^\s*\d+\s*us',     # Dwell time
        r'PM\b',             # AM/PM time marker
        r'AM\b',             # AM/PM time marker
    ]

    def _parse_scale_text(self, ocr_text):
        """
        Extract scale value and unit from OCR text.

        Supports units: nm, um/µm, mm, cm, m, pm, angstroms (A/Å)
        Rejects matches that are clearly SEM instrument parameters
        (working distance, voltage, beam current, etc.).

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

        # Find ALL matches, not just the first — then filter out instrument params
        candidates = []
        for match in re.finditer(pattern, ocr_text, flags=re.I):
            value = float(match.group(1))
            unit = match.group(2).lower()
            start_pos = match.start()

            # Check if this match is preceded by or part of an instrument parameter
            # Look at the text context around the match
            context_before = ocr_text[max(0, start_pos - 15):start_pos].strip()
            context_after = ocr_text[match.end():match.end() + 10].strip()
            full_context = context_before + " " + match.group(0) + " " + context_after

            is_instrument = False

            # Reject if preceded by known parameter labels
            if re.search(r'(WD|HV|spot|dwell|curr|det|mag)\s*$', context_before, re.I):
                is_instrument = True
            # Reject voltage values (kV)
            if re.search(r'kV', context_after, re.I):
                is_instrument = True
            # Reject if it looks like a working distance (X.X mm with decimal)
            if unit == 'mm' and '.' in match.group(1):
                is_instrument = True
            # Reject if unit is 'pm' and could be AM/PM time
            if unit == 'pm' and re.search(r'\d{1,2}[:.]\d{2}', context_before):
                is_instrument = True
            # Reject 'm' if it's part of "mm" already matched or time
            if unit == 'm' and re.search(r'(mm|PM|AM)', full_context, re.I):
                is_instrument = True

            if not is_instrument:
                conversion = self.UNIT_TO_NM.get(unit, 1.0)
                candidates.append({
                    'value': value,
                    'unit': unit,
                    'nm': value * conversion,
                    'text': match.group(0),
                    'position': start_pos,
                })

        if not candidates:
            raise ValueError(
                f"Could not extract scale information from OCR text: '{ocr_text}'"
            )

        # If multiple candidates, prefer the last one (scale bars are usually
        # at the end/right of the databar) and prefer round numbers
        def score_candidate(c):
            """Higher = better candidate for a scale bar label."""
            score = 0
            # Round numbers are more likely scale labels
            if c['value'] == int(c['value']):
                score += 10
            # Common scale bar values
            if c['value'] in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
                score += 20
            # Later position = more likely the scale label (right side of bar)
            score += c['position'] / 100.0
            return score

        best = max(candidates, key=score_candidate)
        return best['nm']

    def detect_databar(self, image, metadata=None):
        """
        Detect whether the image has a databar/info bar at the bottom.

        Many SEM images (especially FEI, Hitachi, JEOL) have an information
        bar at the bottom showing acquisition parameters. This method detects
        it using metadata (when available) or image analysis.

        Args:
            image (np.ndarray): RGB or grayscale image array
            metadata (dict, optional): Raw metadata dict from extract_tiff_metadata()

        Returns:
            dict: {
                'has_databar': bool,
                'databar_height': int (pixels),
                'databar_fraction': float (0-1)
            }
        """
        H, W = image.shape[:2]
        result = {'has_databar': False, 'databar_height': 0, 'databar_fraction': 0.0}

        # Method 1: FEI metadata tells us the scan resolution vs image size
        if metadata:
            raw = metadata.get('raw_tags', {})

            # FEI HELIOS tag: Image.ResolutionY < actual image height = databar
            tag_34682 = raw.get(34682) or raw.get('FEI_HELIOS')
            if isinstance(tag_34682, dict):
                image_info = tag_34682.get('Image', {})
                res_y = image_info.get('ResolutionY')
                if res_y is not None:
                    try:
                        scan_height = int(res_y)
                        if 0 < scan_height < H:
                            result['has_databar'] = True
                            result['databar_height'] = H - scan_height
                            result['databar_fraction'] = result['databar_height'] / H
                            return result
                    except (ValueError, TypeError):
                        pass

        # Method 2: Detect a uniform-color strip at the bottom
        # SEM databars are typically uniform black or dark gray
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Scan from largest to smallest candidate bar height (up to 15% of image)
        max_bar_height = int(H * 0.15)
        for test_height in range(max_bar_height, 10, -2):
            bottom_strip = gray[H - test_height:, :]
            row_stds = np.std(bottom_strip, axis=1)
            # A databar has very low per-row variation (uniform color per row)
            if np.mean(row_stds) < 35:
                result['has_databar'] = True
                result['databar_height'] = test_height
                result['databar_fraction'] = test_height / H
                return result

        return result

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
