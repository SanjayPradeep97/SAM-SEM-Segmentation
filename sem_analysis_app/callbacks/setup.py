"""
Setup tab: SAM initialisation, image loading, session resume.
"""
import os
import gradio as gr

from sem_particle_analysis import (SAMModel, ScaleDetector, ParticleSegmenter,
                                   ResultsManager, discover_checkpoints,
                                   infer_model_type)
from ..state import state
from .gallery import create_image_gallery

def resolve_checkpoint(value):
    """
    Turn whatever the checkpoint dropdown hands back into a real path.

    Depending on version and settings, a Gradio dropdown may return the value, a
    single-item list, or the displayed label. Rather than depend on which, fall
    back to matching the text against the checkpoints actually on disk.

    Returns:
        str or None: An existing path, or None if nothing matches.
    """
    if isinstance(value, list):
        value = value[0] if value else None
    if not value:
        return None

    text = os.path.expanduser(str(value)).strip()
    if os.path.exists(text):
        return text
    return next((str(p) for p in discover_checkpoints() if p.name in text), None)


def initialize_sam(sam_path, progress=gr.Progress()):
    """Load the SAM checkpoint chosen on the Setup tab."""
    try:
        sam_path = resolve_checkpoint(sam_path)
        if sam_path is None:
            return ("❌ No checkpoint at that path. Pick one from the list, or run "
                    "python download_sam_weights.py"), gr.update(interactive=False)

        # Derived from the filename rather than chosen separately, so the
        # architecture can never disagree with the weights being loaded.
        model_type = infer_model_type(sam_path)

        progress(0.3, desc="Loading SAM model...")
        state.sam_model = SAMModel(sam_path, model_type=model_type)
        state.segmenter = ParticleSegmenter(state.sam_model)

        progress(0.7, desc="Initializing scale detector...")
        state.scale_detector = ScaleDetector(use_gpu=False)

        progress(1.0, desc="Complete!")

        return f"✅ SAM model loaded successfully ({model_type})", gr.update(interactive=True)

    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(interactive=False)


def _as_path(item):
    """
    Absolute path for one file-picker entry.

    Gradio has handed back plain strings and File-like objects with a ``.name``
    across versions, so accept either. Paths are checked before ``.name`` because
    pathlib.Path also has that attribute — and there it means the bare filename.
    """
    if isinstance(item, (str, os.PathLike)):
        return os.path.abspath(os.fspath(item))
    return os.path.abspath(getattr(item, "name", str(item)))


def load_images_from_folder(file_input, progress=gr.Progress()):
    """Load images from file browser."""
    try:
        if not file_input:
            return "❌ Please select one or more images", None

        progress(0.3, desc="Loading selected files...")
        items = file_input if isinstance(file_input, list) else [file_input]
        selected = sorted({_as_path(item) for item in items})

        state.image_paths = [p for p in selected if os.path.isfile(p)]
        for missing in (p for p in selected if not os.path.isfile(p)):
            print(f"⚠️  Skipping non-existent file: {missing}")

        skipped = len(selected) - len(state.image_paths)
        if not state.image_paths:
            return f"❌ No valid images found ({skipped} files skipped)", None

        source = "selected files"
        folder_for_csv = os.path.dirname(state.image_paths[0])
        state.current_index = 0
        state.processed_images = {}

        # Initialize results manager — reuse resumed CSV if available
        if state.resumed_csv_path is not None and state.results_manager is not None:
            # User already resumed a session CSV, keep using it
            pass
        else:
            csv_path = os.path.join(folder_for_csv, "analysis_results.csv")
            state.results_manager = ResultsManager(csv_file=csv_path)

        # Load saved state if exists
        state.load_state()

        # Cross-reference images with results CSV to set checkmarks
        matched, unmatched = state.sync_processed_from_csv()

        progress(0.7, desc="Creating gallery...")
        gallery_data = create_image_gallery()

        progress(1.0, desc="Complete!")

        status_parts = [f"✅ Loaded {len(state.image_paths)} images from {source}. Ready to process!"]
        if matched > 0:
            status_parts.append(f"{matched} already analyzed (from CSV)")
        if unmatched > 0:
            status_parts.append(f"⚠️ {unmatched} CSV entries have no matching image")

        status = " | ".join(status_parts)

        return status, gallery_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", None


def resume_session(csv_file):
    """Resume a previous analysis session from a CSV results file.

    Loads the CSV as the active results file and cross-references filenames
    with uploaded images to mark already-analyzed images with checkmarks.
    """
    try:
        if csv_file is None:
            return "❌ No CSV file selected", None, ""

        # Handle Gradio File object or string path
        if hasattr(csv_file, 'name'):
            csv_path = os.path.abspath(csv_file.name)
        else:
            csv_path = os.path.abspath(str(csv_file))

        if not os.path.exists(csv_path):
            return "❌ CSV file not found", None, ""

        # Load the CSV via ResultsManager
        state.results_manager = ResultsManager(csv_file=csv_path, auto_create=False)
        state.resumed_csv_path = csv_path

        results_df = state.results_manager.get_results()
        csv_count = len(results_df)

        if csv_count == 0:
            return "⚠️ CSV loaded but contains no results", None, ""

        # If images are already loaded, cross-reference
        if state.image_paths:
            matched, unmatched = state.sync_processed_from_csv()
            state.save_state()

            # Regenerate gallery with updated checkmarks
            gallery_data = create_image_gallery()

            # Build status
            status_parts = [f"✅ Resumed: {csv_count} results in CSV, {matched} matched to loaded images"]
            if unmatched > 0:
                status_parts.append(f"⚠️ {unmatched} images in results CSV don't have a matching image")

            load_info = f"✅ {len(state.image_paths)} images loaded, {matched} already analyzed"

            return "\n".join(status_parts), gallery_data, load_info
        else:
            # No images loaded yet — just store the CSV for when images are loaded
            return f"✅ CSV loaded with {csv_count} results. Now load your images to see checkmarks.", None, ""

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", None, ""
