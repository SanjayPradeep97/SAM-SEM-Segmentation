"""
The scale panel: check the bar, confirm it, or measure it again.

A batch run reads every bar by OCR and nothing confirms it, so every row comes
out of the analysis saying "+unconfirmed". Two of the frames in this set are
worse than unconfirmed: OCR read a "1 um" bar as "7 um", and every size on those
frames is seven times too large. Correcting a mask cannot fix that.

So the panel does three things, in the order they are needed:

  check    re-read the bar and draw the span that was measured across it, so a
           misread label is visible rather than merely possible,
  confirm  record that a person looked, which is what stops the row saying
           nobody has,
  redo     click the two ends of the bar and type its printed length, which
           cannot be fooled by a misread glyph.
"""

from pathlib import Path

import gradio as gr
import numpy as np

from sem_analysis_app.callbacks.refinement import get_current_visualization
from sem_analysis_app.callbacks.scale_tab import confirm_scale
from sem_analysis_app.state import state
from sem_analysis_app.visualization import (create_particle_visualization,
                                            create_results_dataframe,
                                            render_scale_check)
from sem_particle_analysis import scale_calibration as sc

UNITS = ["nm", "µm", "mm", "Å"]


def summary():
    """One line saying what the scale is and how much it should be trusted."""
    calibration = getattr(state, "scale_calibration", None)
    if calibration is None:
        return ("**No scale for this frame** — its measurements are in pixels.")
    text = (f"**{calibration.nm_per_px:.4g} nm/px** · {calibration.method_label}")
    if calibration.trustworthy:
        text += " · ✅ confirmed"
    else:
        text += " · ⚠️ nobody has checked this"
    if calibration.warning:
        text += f"\n\n⚠️ {calibration.warning}"
    return text


def _detector():
    """The OCR detector, built on first use — it costs seconds to start."""
    if state.scale_detector is None:
        from sem_particle_analysis import ScaleDetector

        state.scale_detector = ScaleDetector(use_gpu=False)
    return state.scale_detector


def check_bar():
    """
    Read the bar again and draw what was measured.

    Re-detected rather than restored: the analysis recorded the number it
    arrived at, not the pixels it measured to get there, and the pixels are the
    half of it that cannot be checked by reading.

    Returns:
        tuple: (picture of the bar or None, what happened, summary)
    """
    if state.current_image is None:
        return None, "Open a frame first", summary()
    try:
        found = _detector().detect_scale_bar_anywhere(state.current_image)
    except Exception as problem:
        return (None, f"❌ No bar found — {problem}. Click its two ends instead.",
                summary())

    reading = sc.ScaleCalibration(
        nm_per_px=float(found["conversion"]), method="box_ocr",
        scale_nm=float(found["scale_nm"]), pixel_length=float(found["pixel_length"]),
        detail=f"read in the {found.get('region_name', 'image')} corner",
        warning=found.get("warning"),
        extra={"region": found.get("region"),
               "line_coords": found.get("line_coords")})

    picture = render_scale_check(state.current_image, reading)
    recorded = getattr(state, "scale_calibration", None)
    note = ""
    if recorded is not None and abs(reading.nm_per_px - recorded.nm_per_px) > \
            0.02 * recorded.nm_per_px:
        note = (f" ⚠️ This differs from the {recorded.nm_per_px:.4g} nm/px the "
                f"analysis recorded — one of the two is wrong.")
    warning = f" ⚠️ {reading.warning}" if reading.warning else ""
    return (picture,
            f"Read {sc.format_length(reading.scale_nm)} over "
            f"{reading.pixel_length:.0f} px = {reading.nm_per_px:.4g} nm/px."
            f"{warning}{note}",
            summary())


def confirm():
    """Record that a person has looked at this scale."""
    status, _lines = confirm_scale()
    return status, summary(), _header()


def start_points():
    """Take the next two clicks on the frame as the ends of the bar."""
    state.scale_click_mode = True
    state.scale_points = []
    return ("📏 Click one end of the bar, then the other. The frame is shown at "
            "full size, so click as precisely as you can."), _frame_with_points()


def clear_points():
    """Forget the clicked ends and go back to editing the mask."""
    state.scale_click_mode = False
    state.scale_points = []
    return "Back to editing the mask.", get_current_visualization()


def clicking():
    """True while clicks on the frame mean bar ends rather than particles."""
    return bool(getattr(state, "scale_click_mode", False))


def add_point(x, y):
    """
    Record one end of the bar.

    Returns:
        tuple: (picture, what happened)
    """
    points = getattr(state, "scale_points", None) or []
    if len(points) >= 2:
        points = []
    points.append((float(x), float(y)))
    state.scale_points = points

    if len(points) == 1:
        return _frame_with_points(), f"One end at ({x}, {y}). Now the other."
    span = ((points[1][0] - points[0][0]) ** 2
            + (points[1][1] - points[0][1]) ** 2) ** 0.5
    return (_frame_with_points(),
            f"Bar spans {span:.1f} px. Type its printed length, then Apply.")


def apply_points(value, unit):
    """
    Use the two clicked ends and the typed length as the scale.

    Nothing here can be misread, so the result is confirmed as it is made — this
    is the answer for a frame whose printed label OCR got wrong.

    Returns:
        tuple: (picture, status, summary, header, measurements)
    """
    points = getattr(state, "scale_points", None) or []
    if len(points) != 2:
        return (gr.update(), "❌ Click both ends of the bar first", summary(),
                _header(), gr.update())
    if value in (None, ""):
        return (gr.update(), "❌ Type the length printed beside the bar",
                summary(), _header(), gr.update())

    try:
        calibration = sc.from_two_points(points[0], points[1], float(value), unit)
    except sc.ScaleError as problem:
        return gr.update(), f"❌ {problem}", summary(), _header(), gr.update()

    was = getattr(state, "scale_calibration", None)
    state.scale_calibration = calibration
    state.scale_info = calibration.to_dict()
    state.accept_scale(calibration)
    # The measurements are pixel counts times this number, so they all change.
    if state.analyzer is not None:
        state.analyzer.set_conversion_factor(calibration.nm_per_px)
    # Remember it for this frame, so leaving and coming back keeps the fix.
    stem = Path(state.image_paths[state.current_index]).stem if state.image_paths else None
    if stem:
        getattr(state, "review_scales", {})[stem] = (calibration.nm_per_px,
                                                     calibration.provenance)

    state.scale_click_mode = False
    state.scale_points = []

    change = ""
    if was is not None and was.nm_per_px:
        factor = calibration.nm_per_px / was.nm_per_px
        if abs(factor - 1) > 0.02:
            change = f" Every size on this frame changes by {factor:.3g}x."
    measurements = state.analyzer.get_measurements(in_nm=True) if state.analyzer else None
    return (get_current_visualization(),
            f"✅ {calibration.summary()}.{change}", summary(), _header(),
            create_results_dataframe(measurements) if measurements else gr.update())


def _header():
    from sem_analysis_app.callbacks.scale_tab import frame_header

    return frame_header()


def _frame_with_points():
    """The frame with the clicked ends marked, so they can be judged."""
    if state.cropped_image is None:
        return None
    import cv2

    picture = state.cropped_image.copy()
    for index, (x, y) in enumerate(getattr(state, "scale_points", []) or []):
        cv2.drawMarker(picture, (int(x), int(y)), (255, 210, 0),
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=28,
                       thickness=3)
        cv2.putText(picture, str(index + 1), (int(x) + 14, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 210, 0), 2)
    points = getattr(state, "scale_points", []) or []
    if len(points) == 2:
        cv2.line(picture, (int(points[0][0]), int(points[0][1])),
                 (int(points[1][0]), int(points[1][1])), (255, 64, 96), 2)
    return picture
