"""
Opening a pre-analysed folder, and moving through it.

The analysis has already happened somewhere else: a folder holding the frames,
a mask per frame and a results CSV. This module's job is to put one of those
masks back into a ParticleAnalyzer so the refinement tools can edit it, and to
put the scale back so the numbers come out in nanometres.

Nothing here segments. That is the point — a review pass should not be able to
change a mask by accident, only by a click that says so.
"""

import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from sem_analysis_app.callbacks.refinement import get_current_visualization
from sem_analysis_app.callbacks.scale_tab import frame_header
from sem_analysis_app.state import state
from sem_analysis_app.visualization import (create_particle_visualization,
                                            create_results_dataframe)
from sem_particle_analysis import ParticleAnalyzer, ResultsManager
from sem_particle_analysis import scale_calibration as sc
from sem_particle_analysis.data_manager import parse_measurement_list

# Where a reviewed folder keeps its own answers. Kept apart from
# analysis_results.csv so the automatic pass stays on disk to compare against.
REVIEWED_CSV = "reviewed_results.csv"
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

    root = Path(folder)
    state.review_root = root
    state.review_masks = masks
    state.image_paths = [str(raw / f"{stem}.png") for stem in stems]
    state.current_index = 0
    state.processed_images = {}
    state.min_particle_size = int(min_size)
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

    done = f", {reviewed} already reviewed" if reviewed else ""
    return (f"✅ {len(stems)} frames from {root.name}{done}. "
            f"Objects under {int(min_size)} px are dropped.",
            gallery_items())


def gallery_items():
    """Thumbnails of the overlays, ticked where a frame has been reviewed."""
    if not state.image_paths:
        return []
    overlays = Path(getattr(state, "review_root", ".")) / "overlay"
    items = []
    for index, path in enumerate(state.image_paths):
        stem = Path(path).stem
        picture = overlays / f"{stem}.png"
        source = picture if picture.exists() else Path(path)
        try:
            thumbnail = Image.open(source).convert("RGB")
            thumbnail.thumbnail((320, 320), Image.LANCZOS)
        except Exception:
            continue
        if state.is_processed(index):
            count = state.processed_images[index].get("num_particles", "?")
            items.append((thumbnail, f"✅ {stem} ({count})"))
        else:
            items.append((thumbnail, f"⚪ {stem}"))
    return items


def open_index(index):
    """
    Put the frame at ``index`` and its saved mask on screen.

    Returns:
        tuple: (visualisation, header, info, measurements)
    """
    if not state.image_paths or not 0 <= index < len(state.image_paths):
        return None, frame_header(), "No more frames", None

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

    mask = np.array(Image.open(state.review_masks / f"{stem}.png").convert("L")) > 127
    analyzer = ParticleAnalyzer(conversion_factor=nm_per_px,
                                min_size=state.min_particle_size)
    # remove_border=False: the batch already decided what to do at the frame
    # edge, and re-applying it here would quietly drop objects the analysis kept.
    analyzer.analyze_mask(mask, min_size=state.min_particle_size,
                          remove_border=False)
    state.analyzer = analyzer

    measurements = analyzer.get_measurements(in_nm=nm_per_px is not None)
    info = (f"{stem} — {index + 1} of {len(state.image_paths)}, "
            f"{measurements['num_particles']} from the analysis")
    return (create_particle_visualization(state.cropped_image,
                                          analyzer.labeled_mask, analyzer.regions,
                                          show_labels=state.show_particle_numbers),
            frame_header(), info, create_results_dataframe(measurements))


def select_from_gallery(evt: gr.SelectData):
    """Open the clicked frame, and go to the tab that edits it."""
    viz, header, info, table = open_index(evt.index)
    return viz, header, info, table, gr.Tabs(selected=2), "delete"


def save_and_next():
    """
    Record this frame as reviewed, then open the next one.

    Returns:
        tuple: (status, gallery, viz, header, info, measurements)
    """
    from sem_analysis_app.callbacks.results import save_current_results

    status, _gallery = save_current_results(recorded_name())
    if not status.startswith("✅"):
        return (status, gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update())

    viz, header, info, table = open_index(state.current_index + 1)
    if viz is None:
        return (f"{status} — that was the last frame.", gallery_items(),
                gr.update(), header, info, gr.update())
    return status, gallery_items(), viz, header, info, table


def skip_to_next():
    """Move on without recording anything."""
    viz, header, info, table = open_index(state.current_index + 1)
    if viz is None:
        return gr.update(), header, "No more frames", gr.update()
    return viz, header, info, table


def go_back():
    """Open the previous frame."""
    viz, header, info, table = open_index(state.current_index - 1)
    if viz is None:
        return gr.update(), header, "Already at the first frame", gr.update()
    return viz, header, info, table


def reload_frame():
    """Throw away every edit and put the analysis's own mask back."""
    viz, header, info, table = open_index(state.current_index)
    return viz, header, f"Reloaded — {info}", table


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


def save_here():
    """
    Record this frame as reviewed and stay on it.

    Returns:
        tuple: (status, gallery)
    """
    from sem_analysis_app.callbacks.results import save_current_results

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
