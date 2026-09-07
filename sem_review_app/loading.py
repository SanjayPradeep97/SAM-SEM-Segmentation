"""
Opening a pre-analysed folder, and moving through it.

The analysis has already happened somewhere else: a folder holding the frames,
a mask per frame and a results CSV. This module's job is to put one of those
masks back into a ParticleAnalyzer so the refinement tools can edit it, and to
put the scale back so the numbers come out in nanometres.

Nothing here segments. That is the point — a review pass should not be able to
change a mask by accident, only by a click that says so.

Corrections belong to the frame, not to the visit. Leaving a frame writes the
mask as the analyst left it, so coming back to it shows their work rather than
the batch's first guess, and closing the app does not throw the difference
away.
"""

import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from sem_analysis_app.callbacks.refinement import (get_current_visualization,
                                                   set_click_mode,
                                                   set_point_type)
from sem_analysis_app.callbacks.scale_tab import frame_header
from sem_analysis_app.state import state
from sem_analysis_app.visualization import (create_particle_visualization,
                                            create_results_dataframe,
                                            create_summary_statistics_table)
from sem_particle_analysis import ParticleAnalyzer, ResultsManager
from sem_particle_analysis import scale_calibration as sc
from sem_particle_analysis.data_manager import parse_measurement_list

# Where a reviewed folder keeps its own answers. Kept apart from
# analysis_results.csv so the automatic pass stays on disk to compare against.
REVIEWED_CSV = "reviewed_results.csv"
# Likewise for the masks: the analyst's version goes beside the batch's, never
# over it. "Reload from the analysis" has to have something to reload.
EDITED_MASKS = "mask_reviewed"
EDITED_OVERLAYS = "overlay_reviewed"
# Objects below this are dropped when a mask is loaded. Matches the batch's own
# floor; raising it here re-filters a mask that was made with a lower one.
DEFAULT_MIN_SIZE = 200


class FolderProblem(Exception):
    """The chosen folder is not a pre-analysed one."""


def analysis_folder(folder):
    """
    Check a folder holds an analysis, and return its parts.

    Returns:
        tuple: (raw_dir, mask_dir, frames) where frames are stems present in
        both, sorted.

    Raises:
        FolderProblem: With a message naming what is missing.
    """
    root = Path(folder or "")
    if not root.is_dir():
        raise FolderProblem(f"{root} is not a folder")
    raw, masks = root / "raw", root / "mask"
    missing = [name for name, path in (("raw", raw), ("mask", masks))
               if not path.is_dir()]
    if missing:
        raise FolderProblem(
            f"{root.name} has no {' or '.join(missing)} folder — pick the "
            f"folder an analysis run wrote, the one holding raw, mask and "
            f"overlay.")

    stems = sorted({p.stem for p in raw.glob("*.png")}
                   & {p.stem for p in masks.glob("*.png")})
    if not stems:
        raise FolderProblem(f"{root.name} has no frame with both a raw image "
                            f"and a mask.")
    return raw, masks, stems


def scale_from(method, nm_per_px):
    """
    Rebuild the calibration a results row describes.

    The provenance is carried back as it was written, so reviewing an image and
    saving it again does not quietly upgrade "nobody checked this scale" into
    something else.
    """
    if not nm_per_px or not np.isfinite(nm_per_px):
        return None
    text = str(method or "")
    unconfirmed = text.endswith("+unconfirmed")
    name = text[: -len("+unconfirmed")] if unconfirmed else text
    return sc.ScaleCalibration(nm_per_px=float(nm_per_px),
                               method=name or "box_ocr",
                               confirmed=not unconfirmed and name != "box_ocr",
                               detail="restored from the analysis")


def load_folder(folder, min_size=DEFAULT_MIN_SIZE):
    """
    Open a pre-analysed folder for review.

    Returns:
        tuple: (status, gallery_items)
    """
    try:
        raw, masks, stems = analysis_folder(folder)
    except FolderProblem as problem:
        return f"❌ {problem}", None

    # Before review_root moves: whatever is on screen belongs to the folder
    # being left, and its corrections have to be written there.
    remember_edit()

    root = Path(folder)
    state.review_root = root
    state.review_masks = masks
    state.image_paths = [str(raw / f"{stem}.png") for stem in stems]
    state.current_index = 0
    state.processed_images = {}
    state.min_particle_size = int(min_size)
    # Re-read from disk rather than trusting what this process happens to
    # remember, so re-opening a folder picks up corrections made since.
    state.review_edits = {}
    state.review_opened = None
    state.reset_image_state()

    # What the analysis said, so an image opens with its own scale.
    state.review_scales = {}
    # What the analysis called each frame — the TIFF it came from, not the PNG
    # copy this app reads — so the two results files line up.
    state.review_names = {}
    automatic = root / "analysis_results.csv"
    if automatic.exists():
        table = ResultsManager(csv_file=str(automatic), auto_create=False).get_results()
        for _, row in table.iterrows():
            stem = Path(str(row["file_name"])).stem
            state.review_scales[stem] = (row.get("nm_per_px"),
                                         row.get("scale_method"))
            state.review_names[stem] = str(row["file_name"])

    state.results_manager = ResultsManager(csv_file=str(root / REVIEWED_CSV))
    reviewed, _unmatched = state.sync_processed_from_csv()

    corrected = sum(1 for stem in stems if edited_paths(stem)[0].exists())
    done = f", {reviewed} already reviewed" if reviewed else ""
    kept = f", {corrected} with corrections on disk" if corrected else ""
    return (f"✅ {len(stems)} frames from {root.name}{done}{kept}. "
            f"Objects under {int(min_size)} px are dropped.",
            gallery_items())


# --- the analyst's own masks ------------------------------------------------

def current_stem():
    """The name of the frame on screen, without its extension, or None."""
    if not state.image_paths or not 0 <= state.current_index < len(state.image_paths):
        return None
    return Path(state.image_paths[state.current_index]).stem


def _edits():
    """The corrected masks held for this folder, keyed by frame."""
    if getattr(state, "review_edits", None) is None:
        state.review_edits = {}
    return state.review_edits


def edited_paths(stem):
    """Where this frame's corrected mask and overlay go, existing or not."""
    root = Path(getattr(state, "review_root", None) or ".")
    return (root / EDITED_MASKS / f"{stem}.png",
            root / EDITED_OVERLAYS / f"{stem}.png")


def remember_edit():
    """
    Keep the mask as the analyst left it, so moving on does not discard it.

    Held in memory and written beside the folder's own masks. The disk copy is
    what makes the work survive the app closing, which on a folder of a hundred
    frames is not a remote possibility.

    Writes nothing when the frame was not changed during the visit: a folder
    walked through without an edit should not fill with copies of masks that
    are already on disk.
    """
    stem = current_stem()
    if not stem or state.analyzer is None or state.analyzer.mask is None:
        return
    opened = getattr(state, "review_opened", None)
    mask = state.analyzer.mask
    if opened is not None and opened[0] == stem and np.array_equal(opened[1], mask):
        return

    mask = mask.copy()
    _edits()[stem] = mask
    state.review_opened = (stem, mask)
    mask_path, overlay_path = edited_paths(stem)
    try:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
        overlay = draw_overlay(mask)
        if overlay is not None:
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay).save(overlay_path)
    except OSError:
        # The copy in memory still holds — the folder may simply be read-only.
        pass


def forget_edit(stem):
    """Drop the corrected mask, so the analysis's own comes back."""
    _edits().pop(stem, None)
    if getattr(state, "review_opened", None) is not None \
            and state.review_opened[0] == stem:
        state.review_opened = None
    for path in edited_paths(stem):
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


def mask_for(stem):
    """
    The mask to open a frame with, and where it came from.

    The analyst's own if there is one. An edit made and then navigated away
    from is still their answer, and re-opening the batch's version silently
    threw it away.

    Returns:
        tuple: (mask, note) — note names the source, for the frame's info line.
    """
    edit = _edits().get(stem)
    if edit is not None:
        return edit, "after your corrections"
    saved, _overlay = edited_paths(stem)
    if saved.exists():
        mask = np.array(Image.open(saved).convert("L")) > 127
        _edits()[stem] = mask
        return mask, "after your corrections"
    batch = np.array(Image.open(state.review_masks / f"{stem}.png").convert("L"))
    return batch > 127, "from the analysis"


def draw_overlay(mask):
    """
    The frame with the mask over it, drawn the way the batch draws it.

    Red fill, yellow outline, so a corrected overlay and an automatic one can
    sit in the same contact sheet without one looking like a different kind of
    result.
    """
    if state.cropped_image is None:
        return None
    from skimage import morphology, segmentation as skseg

    picture = state.cropped_image.astype(np.float32).copy()
    picture[mask] = 0.55 * picture[mask] + 0.45 * np.array([255.0, 45.0, 70.0])
    edge = morphology.dilation(skseg.find_boundaries(mask, mode="outer"),
                               morphology.disk(1))
    picture[edge] = np.array([255.0, 210.0, 0.0])
    return picture.astype(np.uint8)


def gallery_items():
    """Thumbnails of the overlays, ticked where a frame has been reviewed."""
    if not state.image_paths:
        return []
    overlays = Path(getattr(state, "review_root", ".")) / "overlay"
    items = []
    for index, path in enumerate(state.image_paths):
        stem = Path(path).stem
        # The corrected overlay first: a thumbnail showing the batch's guess
        # for a frame that has been fixed is the same lie as opening it.
        _mask, corrected = edited_paths(stem)
        for source in (corrected, overlays / f"{stem}.png", Path(path)):
            if not source.exists():
                continue
            try:
                thumbnail = Image.open(source).convert("RGB")
                thumbnail.thumbnail((320, 320), Image.LANCZOS)
                break
            except Exception:
                continue
        else:
            continue
        if state.is_processed(index):
            count = state.processed_images[index].get("num_particles", "?")
            items.append((thumbnail, f"✅ {stem} ({count})"))
        else:
            items.append((thumbnail, f"⚪ {stem}"))
    return items


# --- the controls that belong to one frame ----------------------------------

def fresh_controls():
    """
    The per-frame controls, put back to how a frame opens.

    Everything here belongs to one frame. reset_image_state already puts the
    app's own copy back to Remove, with no points and nothing measured — but
    the widgets went on showing the frame before, so the rail could say Redraw
    while a click meant Remove, and the scale panel could show a bar crop from
    a frame that was no longer open. The state was right and the screen was
    wrong, which is the worse way round.

    Returns:
        tuple: (tool, redraw controls, point type, status, bar picture, scale
        status, printed length)
    """
    set_point_type("positive")
    message, redraw_controls = set_click_mode("delete")
    return ("delete", redraw_controls, "positive", message, None, "", None)


def keep_controls():
    """Leave the controls alone — the frame did not change."""
    return (gr.update(),) * 7


def controls_now():
    """
    The controls as the app's own state has them.

    For a browser that has just connected: the widgets come up at the defaults
    written into the interface, which need not be what the analyst had chosen
    before the page was refreshed. This sends them the truth rather than a
    reset, so a refresh does not silently change the tool under the cursor.
    """
    mode = getattr(state, "click_mode", "delete") or "delete"
    return (mode, gr.update(visible=mode == "point_refine"),
            getattr(state, "point_type", "positive") or "positive",
            "", None, "", None)


def min_size_control():
    """
    The Review tab's slider, showing the floor the folder was opened with.

    Two controls set one number: the box on the Open tab and this slider. The
    slider kept its own default, so opening a folder at 500 px left it reading
    200 — and the first nudge of it silently re-filtered every later frame at
    the wrong floor.
    """
    return gr.update(value=int(state.min_particle_size))


# --- moving between frames --------------------------------------------------

def open_index(index, remember=True):
    """
    Put the frame at ``index`` and its mask on screen.

    Args:
        index (int): Which frame.
        remember (bool): Keep the outgoing frame's corrections. False only for
            a deliberate reload, which exists to throw them away.

    Returns:
        tuple: (visualisation, header, info, measurements, scale summary, this
        frame's particles, this frame's summary). The visualisation is None
        when there is no such frame, and nothing on screen changes.
    """
    from . import scale as scale_panel

    if remember:
        remember_edit()

    if not state.image_paths or not 0 <= index < len(state.image_paths):
        return (None, frame_header(), "No more frames", None,
                scale_panel.summary(), None, None)

    state.current_index = index
    state.reset_image_state()

    path = Path(state.image_paths[index])
    stem = path.stem
    grey = np.array(Image.open(path).convert("L"))
    state.current_image = np.stack([grey] * 3, axis=-1)
    state.cropped_image = state.current_image

    nm_per_px, method = getattr(state, "review_scales", {}).get(stem, (None, None))
    calibration = scale_from(method, nm_per_px)
    if calibration is not None:
        state.scale_calibration = calibration
        state.scale_info = calibration.to_dict()

    mask, note = mask_for(stem)
    analyzer = ParticleAnalyzer(conversion_factor=nm_per_px,
                                min_size=state.min_particle_size)
    # remove_border=False: the batch already decided what to do at the frame
    # edge, and re-applying it here would quietly drop objects the analysis kept.
    analyzer.analyze_mask(mask, min_size=state.min_particle_size,
                          remove_border=False)
    state.analyzer = analyzer
    # What the frame looked like on arrival, so leaving it can tell whether
    # anything was actually done to it.
    state.review_opened = (stem, analyzer.mask.copy())
    # Where this frame's mask came from, for the info line — kept so that
    # redrawing the frame later can say the same thing.
    state.review_note = note

    measurements = analyzer.get_measurements(in_nm=nm_per_px is not None)
    info = (f"{stem} — {index + 1} of {len(state.image_paths)}, "
            f"{measurements['num_particles']} particles {note}")
    # A frame opens with the mask editable and the scale panel showing what it
    # will be measured with, which is the thing most easily taken on trust.
    state.scale_click_mode = False
    state.scale_points = []
    table = create_results_dataframe(measurements)
    return (create_particle_visualization(state.cropped_image,
                                          analyzer.labeled_mask, analyzer.regions,
                                          show_labels=state.show_particle_numbers),
            frame_header(), info, table, scale_panel.summary(),
            table, create_summary_statistics_table(measurements))


def _stayed(frame, info):
    """There was no frame to move to: change the words, not the picture."""
    return (gr.update(), frame[1], info, gr.update(), frame[4],
            gr.update(), gr.update())


def first_unreviewed():
    """Where to resume a folder: the first frame with no reviewed row."""
    for index in range(len(state.image_paths)):
        if not state.is_processed(index):
            return index
    return 0


def current_view():
    """
    Draw the frame that is already open, without disturbing it.

    Unlike open_index this resets nothing and re-reads nothing: clicks queued
    but not yet applied are still queued, and are drawn as such. That is what
    makes it safe to call every time a browser connects.

    Returns:
        tuple: the same seven values open_index returns.
    """
    from . import scale as scale_panel

    stem = current_stem()
    measurements = state.analyzer.get_measurements(in_nm=True)
    table = create_results_dataframe(measurements)
    info = (f"{stem} — {state.current_index + 1} of {len(state.image_paths)}, "
            f"{measurements['num_particles']} particles "
            f"{getattr(state, 'review_note', '') or 'as you left them'}")
    return (get_current_visualization(), frame_header(), info, table,
            scale_panel.summary(), table,
            create_summary_statistics_table(measurements))


def resume_view():
    """
    Put a frame on the Review tab, for a browser that has just connected.

    The Review tab is per-connection and came up saying "Nothing open" every
    time the page was loaded — after a restart, after a refresh, after opening
    a folder — even with the folder open and half of it already reviewed. The
    only way through was to go to Frames and click a thumbnail, which is not
    something the tab said to do in any way an analyst mid-folder would read as
    "your work is still here".

    A frame the app already has open is redrawn as it stands. Otherwise the
    first frame still to be reviewed is opened, which is where somebody
    resuming a folder was going anyway.

    Returns:
        tuple: the frame outputs, followed by the per-frame controls.
    """
    from . import scale as scale_panel

    if not state.image_paths:
        return (None, frame_header(), "No folder open", None,
                scale_panel.summary(), None, None) + controls_now()
    if state.analyzer is None or state.cropped_image is None:
        return open_index(first_unreviewed()) + fresh_controls()
    return current_view() + controls_now()


def select_from_gallery(evt: gr.SelectData):
    """Open the clicked frame, and go to the tab that edits it."""
    return open_index(evt.index) + fresh_controls() + (gr.Tabs(selected=2),)


def save_and_next():
    """
    Record this frame as reviewed, then open the next one.

    Returns:
        tuple: (status, gallery) followed by the frame outputs.
    """
    from sem_analysis_app.callbacks.results import save_current_results

    remember_edit()
    status, _gallery = save_current_results(recorded_name())
    if not status.startswith("✅"):
        return (status,) + (gr.update(),) * 15

    frame = open_index(state.current_index + 1)
    if frame[0] is None:
        return ((f"{status} — that was the last frame.", gallery_items())
                + _stayed(frame, "No more frames") + keep_controls())
    return (status, gallery_items()) + frame + fresh_controls()


def skip_to_next():
    """Move on without recording anything."""
    frame = open_index(state.current_index + 1)
    if frame[0] is None:
        return _stayed(frame, "No more frames") + keep_controls()
    return frame + fresh_controls()


def go_back():
    """Open the previous frame."""
    frame = open_index(state.current_index - 1)
    if frame[0] is None:
        return _stayed(frame, "Already at the first frame") + keep_controls()
    return frame + fresh_controls()


def reload_frame():
    """Throw away every correction and put the analysis's own mask back."""
    stem = current_stem()
    if stem:
        forget_edit(stem)
    frame = open_index(state.current_index, remember=False)
    if frame[0] is None:
        return _stayed(frame, "Nothing open") + keep_controls()
    return ((frame[0], frame[1], f"Reloaded — {frame[2]}") + frame[3:]
            + fresh_controls())


def current_overlay_path():
    """Where the analysis's overlay for the current frame lives, or None."""
    if not state.image_paths:
        return None
    stem = Path(state.image_paths[state.current_index]).stem
    picture = Path(getattr(state, "review_root", ".")) / "overlay" / f"{stem}.png"
    return str(picture) if picture.exists() else None


def recorded_name():
    """The name to save the open frame under: the one the analysis used."""
    if not state.image_paths:
        return None
    stem = Path(state.image_paths[state.current_index]).stem
    return getattr(state, "review_names", {}).get(stem)


def current_header():
    """The header line, after an edit changed the count."""
    return frame_header()


def current_tables():
    """
    This frame's particles and summary, as the Results tab shows them.

    Chained onto the edits that do not carry those two outputs themselves —
    Undo, and a scale measured by hand, which changes every size on the frame.
    Without it the Results tab kept showing figures the frame no longer had.

    Returns:
        tuple: (particles, summary)
    """
    if state.analyzer is None:
        return None, None
    measurements = state.analyzer.get_measurements(in_nm=True)
    return (create_results_dataframe(measurements),
            create_summary_statistics_table(measurements))


def save_here():
    """
    Record this frame as reviewed and stay on it.

    Returns:
        tuple: (status, gallery)
    """
    from sem_analysis_app.callbacks.results import save_current_results

    remember_edit()
    status, _gallery = save_current_results(recorded_name())
    return status, gallery_items()


def restore():
    """
    Put the open folder back on screen when a browser connects.

    The app keeps one process-wide state, so a folder opened by --folder, or
    before a refresh, is still open — but the gallery is per-connection and comes
    up empty, which looks like nothing was loaded.

    Returns:
        tuple: (status, gallery, folder_path)
    """
    root = getattr(state, "review_root", None)
    if root is None or not state.image_paths:
        return "", [], ""
    done = sum(1 for i in range(len(state.image_paths)) if state.is_processed(i))
    return (f"✅ {len(state.image_paths)} frames from {Path(root).name}"
            + (f", {done} already reviewed" if done else ""),
            gallery_items(), str(root))


def review_click(evt: gr.SelectData):
    """
    One click on the frame, routed by what the analyst is doing.

    While the scale panel is asking for the ends of the bar, a click means an
    end; otherwise it means a particle. One image, one click stream, so the
    refinement tools and the scale tool cannot both think a click was theirs.
    """
    from sem_analysis_app.callbacks import handle_image_click
    from . import scale

    if scale.clicking():
        return scale.add_point(evt.index[0], evt.index[1])
    return handle_image_click(evt)
