"""Gradio callbacks, grouped by the tab they serve."""

from .setup import (
    initialize_sam,
    load_images_from_folder,
    resume_session,
)
from .gallery import (
    create_image_gallery,
    restore_session,
    select_image_from_gallery,
)
from .scale import (
    detect_scale_metadata,
    detect_scale_ocr_in_box,
    apply_manual_scale,
    handle_scale_click,
    set_scale_mode,
    reset_scale_clicks,
    adjust_crop,
    detect_scale_clicked,
)
from .segmentation import (
    segment_with_sam,
    select_mask_and_analyze,
)
from .refinement import (
    get_current_visualization,
    handle_image_click,
    set_min_particle_size,
    toggle_particle_numbers,
    set_point_type,
    reset_point_refine,
    set_click_mode,
    apply_refinement_changes,
    undo_last_action,
    clear_edge_particles,
    clear_all_particles,
    clear_pending_changes,
)
from .plots import (
    update_histogram_plots,
)
from .workflow import (
    auto_process_current_image,
    save_and_next,
    skip_to_next,
)
from .scale_tab import (
    apply_two_points,
    clear_scale,
    confirm_scale,
    live_point_readout,
    prepare_scale_tab,
    read_box_scale,
    set_canvas_mode,
)
from .results import (
    save_current_results,
    get_session_summary,
    delete_result_row,
    export_results,
    check_and_remove_duplicates,
)

__all__ = [
    "adjust_crop",
    "apply_manual_scale",
    "apply_refinement_changes",
    "check_and_remove_duplicates",
    "clear_all_particles",
    "clear_edge_particles",
    "clear_pending_changes",
    "create_image_gallery",
    "delete_result_row",
    "detect_scale_clicked",
    "detect_scale_metadata",
    "detect_scale_ocr_in_box",
    "export_results",
    "get_current_visualization",
    "get_session_summary",
    "handle_image_click",
    "handle_scale_click",
    "initialize_sam",
    "load_images_from_folder",
    "reset_point_refine",
    "reset_scale_clicks",
    "resume_session",
    "save_current_results",
    "segment_with_sam",
    "select_image_from_gallery",
    "select_mask_and_analyze",
    "set_click_mode",
    "set_min_particle_size",
    "set_point_type",
    "set_scale_mode",
    "toggle_particle_numbers",
    "undo_last_action",
    "update_histogram_plots",
]
