"""
Semi-automatic workflow: auto-process on load, then save-and-advance.

The intended loop is: pick an image once, let the app detect scale and produce
ranked mask candidates unattended, choose the best candidate, refine it with
positive/negative clicks, then save and move to the next image without going back
to the gallery.
"""
import os

import gradio as gr

from ..state import state
from .scale import detect_scale_clicked
from .segmentation import segment_with_sam
from .results import save_current_results
from .scale_tab import prepare_scale_tab


def _load_index(index):
    """Load image at ``index`` into state. Returns (image, info) or (None, error)."""
    from sem_particle_analysis.utils import load_image

    if not state.image_paths:
        return None, "No images loaded"
    if not 0 <= index < len(state.image_paths):
        return None, "No more images in this folder"

    state.current_index = index
    state.reset_image_state()
    try:
        state.current_image = load_image(state.image_paths[index])
    except Exception as exc:
        return None, f"❌ Error loading image: {exc}"

    filename = os.path.basename(state.image_paths[index])
    return state.current_image, f"Image {index + 1} / {len(state.image_paths)}: {filename}"


def _auto_detect_scale(progress):
    """
    Establish scale without asking, trying metadata first and then the printed bar.

    The Metadata mode the UI defaults to only works when the vendor wrote a pixel
    size into the TIFF; JEOL TEM exports here do not, and their bar sits in the
    bottom-left rather than the bottom-right a databar search assumes. So fall
    back to sweeping the usual bar positions.

    Returns:
        tuple: (status_text, image_view, crop_slider_update)
    """
    # The Scale tab owns calibration now. If it already established one, this is
    # not the place to second-guess it — re-running detection here would quietly
    # replace a value the analyst confirmed by hand.
    existing = getattr(state, "scale_calibration", None)
    if existing is not None:
        if state.cropped_image is None and state.current_image is not None:
            state.cropped_image = state.current_image
        return (f"✅ Scale ({existing.method_label}): {existing.nm_per_px:.6g} nm/px",
                state.current_image, gr.update())

    # Honour an explicit choice: if the analyst has switched off Metadata mode
    # they are driving scale by hand, and clicks shouldn't be overridden.
    if state.scale_mode != "Metadata":
        try:
            return detect_scale_clicked(progress)
        except Exception as exc:
            return f"❌ Scale error: {exc}", state.current_image, gr.update()

    try:
        status, image_view, crop_update = detect_scale_clicked(progress)
        if state.scale_info is not None:
            return status, image_view, crop_update
    except Exception:
        pass

    if state.scale_detector is None or state.current_image is None:
        return "⚠️ No scale — measurements will be in pixels", state.current_image, gr.update()

    try:
        result = state.scale_detector.detect_scale_bar_anywhere(state.current_image)
    except Exception as exc:
        return (f"⚠️ No scale found ({str(exc)[:80]}). Measurements will be in "
                f"pixels — use OCR or Manual mode to set it."), state.current_image, gr.update()

    state.scale_info = result
    # The bar is printed inside the frame, not in a databar below it, so there is
    # nothing to crop off — cropping would discard real image. Record where the
    # bar sits instead, so segmentation can ignore that patch rather than
    # reporting the bar and its label as particles.
    state.cropped_image = state.current_image
    state.scale_bar_region = result.get("region")

    note = f" ⚠️ {result['warning']}" if result.get("warning") else ""
    return (
        f"✅ Scale from bar ({result['region_name']}): {result['scale_nm']:g} nm / "
        f"{result['pixel_length']}px = {result['conversion']:.4g} nm/px{note}",
        state.current_image,
        gr.update(),
    )


def auto_process_current_image(progress=gr.Progress()):
    """
    Detect scale and generate ranked mask candidates for the loaded image.

    Runs unattended so the analyst lands on a set of candidates rather than on an
    empty tab with two buttons to press. Scale failure is not fatal — segmentation
    still runs and measurements fall back to pixels — because a missing scale is
    worth seeing alongside the masks rather than instead of them.

    Returns:
        tuple: (scale_status, image_view, crop_slider, mask_viz, segment_status,
                mask_choice)
    """
    if state.current_image is None:
        return "No image loaded", None, gr.update(), None, "", gr.update()
    if state.segmenter is None:
        return ("⚠️ SAM is not initialised — load a checkpoint on the Setup tab",
                state.current_image, gr.update(), None, "", gr.update())

    progress(0.15, desc="Detecting scale...")
    scale_status, image_view, crop_update = _auto_detect_scale(progress)

    progress(0.5, desc="Segmenting...")
    try:
        mask_viz, segment_status, choice_update = segment_with_sam(progress)
    except Exception as exc:
        mask_viz, segment_status, choice_update = None, f"❌ Segmentation error: {exc}", gr.update()

    progress(1.0, desc="Ready")
    return (scale_status, image_view, crop_update, mask_viz, segment_status,
            choice_update)


def save_and_next(progress=gr.Progress()):
    """
    Save the current image's measurements, then load and pre-process the next.

    Landing tab is Scale, so the calibration for the new image is verified before
    anything is measured with it.

    Returns:
        tuple: (save_status, gallery, image_info, image_view, scale_status,
                crop_slider, mask_viz, segment_status, mask_choice, tabs,
                canvas_payload, tier1_status, scale_summary, point_readout)
    """
    blank = (gr.update(),) * 13
    save_status, gallery_data = save_current_results()
    if gallery_data is None:
        # Nothing was saved — stay put rather than advancing past unsaved work.
        return (save_status,) + blank

    next_index = state.current_index + 1
    image, info = _load_index(next_index)
    if image is None:
        return ((f"{save_status} — {info}. All done.", gallery_data, info)
                + (gr.update(),) * 11)

    # Establish scale for the new image before segmenting it.
    payload, tier1, summary, hint = prepare_scale_tab()

    scale_status, image_view, crop_update, mask_viz, segment_status, choice_update = \
        auto_process_current_image(progress)

    return (save_status, gallery_data, info, image_view, scale_status,
            crop_update, mask_viz, segment_status, choice_update,
            gr.Tabs(selected=2), payload, tier1, summary, hint)


def skip_to_next():
    """
    Advance without saving, for images that aren't worth measuring.

    Returns:
        tuple: (image_info, image_view, tabs, canvas_payload, tier1_status,
                scale_summary, point_readout)
    """
    image, info = _load_index(state.current_index + 1)
    if image is None:
        return (info,) + (gr.update(),) * 6
    payload, tier1, summary, hint = prepare_scale_tab()
    return info, image, gr.Tabs(selected=2), payload, tier1, summary, hint
