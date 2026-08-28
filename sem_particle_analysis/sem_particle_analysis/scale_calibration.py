"""
Scale calibration: establishing how many nanometres one pixel represents.

Every physical measurement the toolkit reports is a pixel count multiplied by
this one number, so a wrong scale rescales an entire dataset silently. The
calibration is therefore treated as a first-class object with an explicit method
and confidence, rather than as a bare float passed around.

Three tiers, in decreasing order of trustworthiness:

1. ``from_metadata``   — pixel size written by the instrument. Exact.
2. ``from_box_ocr``    — read the printed scale bar inside a region the user
                         drew. Good, but depends on OCR.
3. ``from_two_points`` — the user clicks both ends of the bar and types its
                         value. Slow, but cannot be fooled by a misread glyph.
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

# Unit spellings offered in the manual entry UI.
UNIT_TO_NM = {
    "nm": 1.0,
    "µm": 1000.0,
    "um": 1000.0,
    "mm": 1_000_000.0,
    "Å": 0.1,
    "A": 0.1,
    "pm": 0.001,
}

# Physically sensible pixel sizes for electron microscopy, in nm/px.
VALID_NM_PER_PX = (0.005, 200_000.0)

METHOD_LABELS = {
    "metadata": "file metadata",
    "box_ocr": "scale bar read in box",
    "two_points": "two points clicked on the bar",
    "manual": "entered by hand",
}


class ScaleError(ValueError):
    """Raised when a calibration cannot be established or is implausible."""


@dataclass
class ScaleCalibration:
    """
    A resolved image scale, with enough provenance to justify it later.

    Attributes:
        nm_per_px: The conversion factor. This is the number that matters.
        method: Which tier produced it — a key of ``METHOD_LABELS``.
        scale_nm: Physical length of the reference, when there was one.
        pixel_length: Its measured length in pixels, when there was one.
        detail: Free text describing where the value came from.
        warning: Set when the value is usable but should be eyeballed.
        confirmed: True once a human has explicitly accepted it.
    """

    nm_per_px: float
    method: str
    scale_nm: Optional[float] = None
    pixel_length: Optional[float] = None
    detail: str = ""
    warning: Optional[str] = None
    confirmed: bool = False
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        validate_nm_per_px(self.nm_per_px)

    @property
    def method_label(self):
        return METHOD_LABELS.get(self.method, self.method)

    @property
    def trustworthy(self):
        """Metadata is exact; anything else benefits from a human glance."""
        return self.method == "metadata" or self.confirmed

    def summary(self):
        """One-line description for the UI."""
        text = f"{self.nm_per_px:.6g} nm/px  ·  {self.method_label}"
        if self.scale_nm and self.pixel_length:
            text += f"  ({format_length(self.scale_nm)} over {self.pixel_length:.0f} px)"
        return text

    def to_dict(self):
        data = asdict(self)
        data["method_label"] = self.method_label
        # 'conversion' is the key the rest of the codebase already reads.
        data["conversion"] = self.nm_per_px
        return data


def validate_nm_per_px(value):
    """Reject a conversion factor no electron micrograph could have."""
    if value is None or not math.isfinite(value) or value <= 0:
        raise ScaleError(f"Scale must be a positive number, got {value!r}")
    low, high = VALID_NM_PER_PX
    if not low <= value <= high:
        raise ScaleError(
            f"{value:g} nm/px is outside the plausible range for electron "
            f"microscopy ({low:g}–{high:g} nm/px). Check the value and units."
        )
    return value


def to_nanometres(value, unit):
    """Convert a length in ``unit`` to nanometres."""
    if value is None or not math.isfinite(value) or value <= 0:
        raise ScaleError("Scale bar length must be a positive number")
    key = (unit or "").strip()
    if key not in UNIT_TO_NM:
        raise ScaleError(f"Unknown unit {unit!r}. Use one of: "
                         + ", ".join(sorted(set(UNIT_TO_NM))))
    return value * UNIT_TO_NM[key]


def format_length(nanometres):
    """Human-readable length, picking the unit a microscopist would use."""
    for limit, unit, factor in ((1e6, "mm", 1e6), (1e3, "µm", 1e3), (0, "nm", 1.0)):
        if nanometres >= limit and limit:
            return f"{nanometres / factor:g} {unit}"
    return f"{nanometres:g} nm"


def from_metadata(detector, image, file_path):
    """
    Tier 1: read the pixel size the instrument recorded in the file.

    Raises:
        ScaleError: If the file carries no usable pixel size.
    """
    try:
        result = detector.detect_scale(image, file_path=str(file_path), method="metadata")
    except Exception as exc:
        raise ScaleError(str(exc)) from exc

    conversion = result.get("conversion")
    if not conversion:
        raise ScaleError("No pixel size in the file metadata")

    return ScaleCalibration(
        nm_per_px=float(conversion),
        method="metadata",
        detail=result.get("manufacturer") or "instrument metadata",
        extra={"raw": {k: v for k, v in result.items()
                       if k in ("manufacturer", "source", "scale_nm")}},
    )


def from_box_ocr(detector, image, box):
    """
    Tier 2: measure and read the scale bar inside a user-drawn rectangle.

    Args:
        box: (x0, y0, x1, y1) in image pixel coordinates.

    Raises:
        ScaleError: If no bar or no readable label is found in the region.
    """
    height, width = image.shape[:2]
    x0, y0, x1, y1 = _normalise_box(box, width, height)
    if (x1 - x0) < 12 or (y1 - y0) < 6:
        raise ScaleError("The box is too small — draw it around the whole scale bar "
                         "and its label.")

    region = dict(
        region_x=((x0 + x1) / 2) / width,
        region_y=((y0 + y1) / 2) / height,
        region_width=(x1 - x0) / width,
        region_height=(y1 - y0) / height,
    )
    try:
        result = detector.detect_scale_bar(image, **region)
    except Exception as exc:
        raise ScaleError(str(exc)) from exc

    return ScaleCalibration(
        nm_per_px=float(result["conversion"]),
        method="box_ocr",
        scale_nm=float(result["scale_nm"]),
        pixel_length=float(result["pixel_length"]),
        detail=f"read “{result.get('ocr_text', '').strip()}” in the box",
        warning=result.get("warning"),
        extra={"box": [x0, y0, x1, y1],
               "line_coords": result.get("line_coords"),
               "polarity": result.get("polarity_used")},
    )


def from_two_points(point_a, point_b, value, unit):
    """
    Tier 3: the user clicks both ends of the bar and types its printed length.

    Nothing here can be misread, so the result is trusted without a warning.

    Raises:
        ScaleError: If the points coincide or the length is unusable.
    """
    (xa, ya), (xb, yb) = point_a, point_b
    pixel_length = math.dist((xa, ya), (xb, yb))
    if pixel_length < 2:
        raise ScaleError("Those two points are in the same place — click each end "
                         "of the scale bar.")

    scale_nm = to_nanometres(value, unit)
    return ScaleCalibration(
        nm_per_px=scale_nm / pixel_length,
        method="two_points",
        scale_nm=scale_nm,
        pixel_length=pixel_length,
        detail=f"{format_length(scale_nm)} measured across {pixel_length:.1f} px",
        confirmed=True,
        extra={"points": [[xa, ya], [xb, yb]]},
    )


def _normalise_box(box, width, height):
    """Order corners and clamp the rectangle to the image."""
    x0, y0, x1, y1 = box
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    x0 = int(max(0, min(x0, width - 1)))
    y0 = int(max(0, min(y0, height - 1)))
    x1 = int(max(0, min(x1, width)))
    y1 = int(max(0, min(y1, height)))
    return x0, y0, x1, y1
