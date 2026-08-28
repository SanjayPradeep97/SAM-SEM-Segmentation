"""
Scale tab: establish nm/pixel before anything is measured.

Three tiers, tried in order of trustworthiness. Tier 1 runs by itself when an
image is opened; tiers 2 and 3 are driven from the interactive canvas and are
only needed when the one above them fails.
"""

import base64
import io
import json
import os

import gradio as gr
import numpy as np
from PIL import Image

from sem_particle_analysis import scale_calibration as sc
from ..state import state

UNITS = ["nm", "µm", "mm", "Å"]

# Anything larger is downscaled for transport to the browser. Well above the
# 2048px micrographs in use, so the magnifier still shows real pixels.
MAX_CANVAS_PX = 2600


def _detector():
    """
    The scale detector, created on first use.

    Calibration needs OCR, not segmentation, so it must not depend on a 2.4 GB
    SAM checkpoint having been loaded first — an analyst should be able to sort
    out scale before choosing a model.
    """
    if state.scale_detector is None:
        from sem_particle_analysis import ScaleDetector

        state.scale_detector = ScaleDetector(use_gpu=False)
    return state.scale_detector


def _image_payload(image, box=None):
    """
    Encode the current image as a data URL for the canvas.

    A ``box`` is included when automatic detection already found the bar, so the
    canvas opens with the region it used and the analyst can see what was read
    rather than having to guess where to look.
    """
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3]
    pil = Image.fromarray(array.astype(np.uint8))
    if max(pil.size) > MAX_CANVAS_PX:
        ratio = MAX_CANVAS_PX / max(pil.size)
        pil = pil.resize((int(pil.width * ratio), int(pil.height * ratio)),
                         Image.LANCZOS)
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG", optimize=False, compress_level=1)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = {"img": f"data:image/png;base64,{encoded}",
               "w": pil.width, "h": pil.height}
    if box:
        payload["box"] = {"x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]}
    return json.dumps(payload)


def _adopt(calibration):
    """
    Make a calibration the authoritative scale for the current image.

    The rest of the app reads state.scale_info['conversion'], so keep that in
    step rather than teaching every caller about the new object.
    """
    state.scale_calibration = calibration
    state.scale_info = calibration.to_dict()
    if state.cropped_image is None and state.current_image is not None:
        # TEM frames have no databar to trim; the printed bar is inside the
        # image and is excluded from segmentation instead of cropped away.
        state.cropped_image = state.current_image
    return calibration


def _status_lines():
    """Current calibration rendered for the summary box."""
    cal = getattr(state, "scale_calibration", None)
    if cal is None:
        return ("⚠️ No scale yet — the measurements will be in pixels.\n"
                "Try tier 2 (draw a box) or tier 3 (click both ends of the bar).")
    lines = [f"✅ {cal.summary()}"]
    if cal.detail:
        lines.append(f"   {cal.detail}")
    if cal.warning:
        lines.append(f"⚠️ {cal.warning}")
    if not cal.trustworthy:
        lines.append("   Not yet confirmed — check the overlay, then press "
                     "“Confirm scale”.")
    return "\n".join(lines)


def prepare_scale_tab():
    """
    Open an image on the Scale tab and run tier 1.

    Returns:
        tuple: (canvas_payload, tier1_status, summary, points_hint)
    """
    if state.current_image is None:
        return "", "No image loaded", _status_lines(), ""

    state.scale_calibration = None
    state.scale_info = None

    payload = _image_payload(state.current_image)
    path = (state.image_paths[state.current_index]
            if state.image_paths and state.current_index < len(state.image_paths)
            else None)

    try:
        calibration = sc.from_metadata(_detector(), state.current_image, path)
    except sc.ScaleError as exc:
        tier1 = f"❌ Tier 1 — no pixel size in the file ({exc})."
        return _try_automatic_bar(payload, tier1)

    _adopt(calibration)
    name = os.path.basename(str(path)) if path else "image"
    return (payload,
            f"✅ Tier 1 — pixel size read from {name}'s metadata. Nothing else to do.",
            _status_lines(), "")


def _try_automatic_bar(payload, tier1_message):
    """
    Attempt tier 2 without being asked, by looking where bars usually are.

    This is the same measurement the analyst would get by drawing the box by
    hand, so offering it saves a step on the majority of images — but it is
    reported as unconfirmed, and the box it used is shown on the canvas so it can
    be checked or corrected rather than taken on trust.
    """
    try:
        found = _detector().detect_scale_bar_anywhere(state.current_image)
    except Exception:
        return (payload,
                tier1_message + " Draw a box below (tier 2), or click both ends of "
                "the bar (tier 3).",
                _status_lines(), "")

    calibration = sc.ScaleCalibration(
        nm_per_px=float(found["conversion"]),
        method="box_ocr",
        scale_nm=float(found["scale_nm"]),
        pixel_length=float(found["pixel_length"]),
        detail=f"found automatically in the {found.get('region_name', 'image')} "
               f"corner: “{found.get('ocr_text', '').strip()}”",
        warning=found.get("warning"),
    )
    _adopt(calibration)

    box = found.get("region")
    if box:
        x0, y0, w, h = box
        payload = _image_payload(state.current_image, (x0, y0, x0 + w, y0 + h))

    return (payload,
            tier1_message + " Tier 2 ran automatically — check the box on the image.",
            _status_lines(), "")


def read_box_scale(box_json, progress=gr.Progress()):
    """Tier 2: OCR the scale bar inside the drawn rectangle."""
    if state.current_image is None:
        return "❌ No image loaded", _status_lines()

    box = _parse_box(box_json)
    if box is None:
        return ("❌ Draw a box around the scale bar and its label first — drag on "
                "the image, then drag the corners to adjust."), _status_lines()

    progress(0.4, desc="Reading the scale bar...")
    try:
        calibration = sc.from_box_ocr(_detector(), state.current_image, box)
    except sc.ScaleError as exc:
        return (f"❌ Tier 2 — {exc}\nAdjust the box so it contains the whole bar "
                f"and its label, or use tier 3."), _status_lines()

    _adopt(calibration)
    note = f"  ⚠️ {calibration.warning}" if calibration.warning else ""
    return (f"✅ Tier 2 — {calibration.summary()}{note}", _status_lines())


def apply_two_points(points_json, value, unit):
    """Tier 3: the user clicked both ends of the bar and typed its length."""
    points = _parse_points(points_json)
    if len(points) != 2:
        return ("❌ Click both ends of the scale bar first — the magnifier shows "
                "exactly which pixel you are on."), _status_lines()
    if value is None:
        return "❌ Type the length printed next to the bar.", _status_lines()

    try:
        calibration = sc.from_two_points(points[0], points[1], float(value), unit)
    except sc.ScaleError as exc:
        return f"❌ Tier 3 — {exc}", _status_lines()

    _adopt(calibration)
    return f"✅ Tier 3 — {calibration.summary()}", _status_lines()


def confirm_scale():
    """Mark the current calibration as checked by a human."""
    cal = getattr(state, "scale_calibration", None)
    if cal is None:
        return "❌ Nothing to confirm yet", _status_lines()
    cal.confirmed = True
    cal.warning = None
    state.scale_info = cal.to_dict()
    return "✅ Scale confirmed", _status_lines()


def clear_scale():
    """Discard the calibration and start again."""
    state.scale_calibration = None
    state.scale_info = None
    return "Scale cleared", _status_lines()


def set_canvas_mode(mode):
    """Show the controls belonging to the selected tier."""
    points = mode.startswith("Tier 3")
    return (gr.update(visible=not points),   # tier 2 controls
            gr.update(visible=points))       # tier 3 controls


def live_point_readout(points_json):
    """Report the pixel distance between the two clicked points."""
    points = _parse_points(points_json)
    if len(points) == 0:
        return "Click the left end of the scale bar."
    if len(points) == 1:
        return f"Point 1 at ({points[0][0]:.0f}, {points[0][1]:.0f}). Now click the other end."
    length = ((points[1][0] - points[0][0]) ** 2
              + (points[1][1] - points[0][1]) ** 2) ** 0.5
    return (f"Bar spans {length:.1f} px. Enter its printed length below, "
            f"then press Apply.")


def _parse_box(raw):
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        box = (data["x0"], data["y0"], data["x1"], data["y1"])
    except (TypeError, ValueError, KeyError):
        return None
    return box if (box[2] - box[0]) > 2 and (box[3] - box[1]) > 2 else None


def _parse_points(raw):
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return [tuple(p) for p in data.get("points", [])][:2]
    except (TypeError, ValueError, AttributeError):
        return []
