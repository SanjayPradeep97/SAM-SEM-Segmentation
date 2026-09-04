"""
Review interface: correct a pre-analysed folder and finalise it.

Four tabs, in the order the work happens: open a folder, pick a frame, fix its
mask, read the numbers. There is no segmentation step — the masks already exist,
and the tools only change one where a click says so.

The refinement tools, the results table and the plots are the main app's own
callbacks, imported rather than reimplemented, so a fix in one is a fix in both.
"""
import gradio as gr

from sem_particle_analysis import discover_checkpoints

from sem_analysis_app.callbacks import (
    apply_refinement_changes,
    check_and_remove_duplicates,
    clear_all_particles,
    clear_edge_particles,
    clear_pending_changes,
    delete_result_row,
    export_results,
    get_session_summary,
    handle_image_click,
    initialize_sam,
    reset_point_refine,
    save_current_results,
    set_click_mode,
    set_min_particle_size,
    set_point_type,
    toggle_particle_numbers,
    undo_last_action,
    update_histogram_plots,
)
from . import loading, scale

# Same tools, same values, as the main app's rail.
REFINE_MODES = [
    ("delete", "🚫  Remove"),
    ("add", "➕  Add"),
    ("merge", "🔗  Merge"),
    ("point_refine", "🎯  Redraw"),
]

APP_CSS = """
.tabs {font-size: 15px;}
.tab-nav button {padding: 10px 20px; font-weight: 500;}
.frame-header {
    padding: 10px 14px; border-radius: 8px;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    font-size: 14px; line-height: 1.5;
}
.frame-header p {margin: 0;}
.tool-rail .gr-form, .tool-rail .gr-box {border: none; background: transparent;}
.tool-rail label {font-size: 13px;}
.tool-rail .gr-button {width: 100%;}
.work-image img {border-radius: 8px; background: #0d0d0d;}
.gr-dataframe table {font-variant-numeric: tabular-nums; font-size: 13px;}
.step-title {
    font-weight: 600; font-size: 13px; letter-spacing: .04em;
    text-transform: uppercase; opacity: .65; margin: 2px 0 6px;
}
"""


def create_interface():
    """Build the review interface."""
    with gr.Blocks(title="Review — SEM & TEM particle analysis") as app:

        gr.Markdown("# 🔍 Review a pre-analysed folder")
        gr.Markdown(
            "Masks made by a batch run, opened one at a time so they can be "
            "corrected and signed off. Remove, Merge and the measurements need "
            "no model; Add and Redraw ask SAM what is under the click."
        )

        with gr.Tabs() as tabs:

            # ---------------------------------------------------------- Setup
            with gr.Tab("⚙️ Open", id=0):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 1 · The analysis", elem_classes=["step-title"])
                        folder_input = gr.Textbox(
                            label="Analysis folder",
                            placeholder=r"D:\...\C1_analysis",
                            info="The folder a run wrote: raw, mask, overlay and "
                                 "analysis_results.csv",
                        )
                        min_size_input = gr.Number(
                            label="Drop objects under (px)",
                            value=loading.DEFAULT_MIN_SIZE, minimum=1, maximum=100000,
                        )
                        load_btn = gr.Button("📂 Open folder", variant="primary",
                                             size="lg")
                        load_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=1):
                        gr.Markdown("### 2 · Model (optional)",
                                    elem_classes=["step-title"])
                        gr.Markdown(
                            "Only Add and Redraw use it. Reviewing, removing and "
                            "merging work without one, and load instantly."
                        )
                        _checkpoints = discover_checkpoints()
                        sam_file = gr.Dropdown(
                            label="SAM checkpoint",
                            choices=[str(p) for p in _checkpoints],
                            value=str(_checkpoints[0]) if _checkpoints else None,
                            allow_custom_value=True,
                        )
                        init_sam_btn = gr.Button("⚡ Load model")
                        init_status = gr.Textbox(label="Status", interactive=False)

            # -------------------------------------------------------- Gallery
            with gr.Tab("🖼️ Frames", id=1):
                gr.Markdown("Click a frame to review it. A tick means it has "
                            "been saved to the reviewed results.")
                gallery = gr.Gallery(label="", show_label=False, columns=6,
                                     height=620, object_fit="contain")

            # --------------------------------------------------------- Review
            with gr.Tab("✏️ Review", id=2):
                header = gr.Markdown("### Nothing open\nOpen a folder, then pick "
                                     "a frame.", elem_classes=["frame-header"])
                frame_info = gr.Textbox(label="", interactive=False,
                                        show_label=False)

                with gr.Accordion("📏 Scale — check it before trusting the sizes",
                                  open=False):
                    scale_summary = gr.Markdown("Open a frame to see its scale.")
                    with gr.Row():
                        with gr.Column(scale=1):
                            check_bar_btn = gr.Button("🔍 Check the bar")
                            confirm_scale_btn = gr.Button("✔️ Confirm scale",
                                                          variant="primary")
                            gr.Markdown(
                                "If the reading is wrong, measure it by hand: "
                                "click one end of the bar, then the other, then "
                                "type the length printed beside it.")
                            points_btn = gr.Button("📏 Click the bar's ends")
                            with gr.Row():
                                bar_value = gr.Number(label="Printed length",
                                                      value=None)
                                bar_unit = gr.Dropdown(choices=scale.UNITS,
                                                       value="µm", label="Unit")
                            apply_points_btn = gr.Button("Apply the two points",
                                                         variant="primary")
                            cancel_points_btn = gr.Button("Cancel", size="sm")
                        with gr.Column(scale=2):
                            scale_check_img = gr.Image(
                                label="What the bar measures", height=210,
                                interactive=False)
                    scale_status = gr.Textbox(label="", interactive=False,
                                              show_label=False)

                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["tool-rail"]):
                        gr.Markdown("### Tool", elem_classes=["step-title"])
                        click_mode_radio = gr.Radio(
                            choices=[(label, value) for value, label in REFINE_MODES],
                            value="delete", label="", show_label=False,
                        )
                        with gr.Group(visible=False) as point_controls:
                            point_type_radio = gr.Radio(
                                choices=[("✓ include", "positive"),
                                         ("✗ exclude", "negative")],
                                value="positive", label="Redraw clicks",
                            )
                            reset_points_btn = gr.Button("Reset points", size="sm")

                        gr.Markdown("### Commit", elem_classes=["step-title"])
                        with gr.Row():
                            apply_btn = gr.Button("✓ Apply", variant="primary")
                            undo_btn = gr.Button("↩️ Undo", variant="secondary")
                        discard_btn = gr.Button("Discard pending clicks", size="sm")
                        reload_btn = gr.Button("↺ Reload from analysis", size="sm")

                        with gr.Accordion("Whole frame", open=False):
                            edge_buffer = gr.Number(label="Edge buffer (px)",
                                                    value=10, minimum=0)
                            clear_edges_btn = gr.Button("🧹 Drop edge particles",
                                                        size="sm")
                            clear_all_btn = gr.Button("🗑️ Clear all", size="sm")
                            min_size_slider = gr.Slider(
                                label="Minimum particle size (px)", minimum=1,
                                maximum=5000, step=1,
                                value=loading.DEFAULT_MIN_SIZE,
                            )
                            show_numbers = gr.Checkbox(label="Number the particles",
                                                       value=True)

                    with gr.Column(scale=4):
                        review_viz = gr.Image(label="", type="numpy",
                                              show_label=False, height=760,
                                              elem_classes=["work-image"])
                        review_status = gr.Textbox(label="", interactive=False,
                                                   show_label=False, lines=1)

                gr.Markdown("---")
                with gr.Row():
                    back_btn = gr.Button("⬅️ Previous", variant="secondary")
                    save_next_btn = gr.Button("✅ Save & next", variant="primary",
                                              size="lg", scale=2)
                    save_btn = gr.Button("💾 Save, stay here", variant="secondary")
                    skip_btn = gr.Button("⏭️ Skip", variant="secondary")
                save_status = gr.Textbox(label="", interactive=False,
                                         show_label=False)

                with gr.Accordion("Measurements for this frame", open=False):
                    frame_results = gr.Dataframe(label="", show_label=False)

            # -------------------------------------------------------- Results
            with gr.Tab("💾 Results", id=3):
                gr.Markdown("### This frame", elem_classes=["step-title"])
                with gr.Row():
                    current_results = gr.Dataframe(label="Particles",
                                                   show_label=True)
                    current_stats = gr.Dataframe(label="Summary",
                                                 show_label=True)

                gr.Markdown("### Reviewed frames", elem_classes=["step-title"])
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh", variant="primary")
                    dedupe_btn = gr.Button("🧹 Remove duplicates")
                    export_btn = gr.Button("📤 Export")
                summary_progress = gr.Markdown("")
                session_table = gr.Dataframe(label="", show_label=False)
                summary_stats = gr.Markdown("")
                with gr.Row():
                    row_picker = gr.Dropdown(label="Delete a row", choices=[],
                                             value=None)
                    delete_row_btn = gr.Button("🗑️ Delete")
                export_status = gr.Textbox(label="", interactive=False,
                                           show_label=False)
                export_file = gr.File(label="Exported CSV", interactive=False)

            # ---------------------------------------------------------- Plots
            with gr.Tab("📊 Plots", id=4):
                plots_btn = gr.Button("📊 Draw size distributions",
                                      variant="primary")
                plots_image = gr.Image(label="", show_label=False)
                plots_status = gr.Textbox(label="", interactive=False,
                                          show_label=False)

        # ----------------------------------------------------------- wiring
        REVIEW_OUTPUTS = [review_viz, header, frame_info, frame_results,
                          scale_summary]

        # A folder opened by --folder, or before a refresh, is still open in the
        # process; the gallery is per-connection and would come up empty.
        app.load(loading.restore,
                 outputs=[load_status, gallery, folder_input])

        load_btn.click(loading.load_folder,
                       inputs=[folder_input, min_size_input],
                       outputs=[load_status, gallery])
        init_sam_btn.click(initialize_sam, inputs=[sam_file],
                           outputs=[init_status, load_btn])

        gallery.select(loading.select_from_gallery,
                       outputs=REVIEW_OUTPUTS + [tabs, click_mode_radio])

        click_mode_radio.change(set_click_mode, inputs=[click_mode_radio],
                                outputs=[review_status, point_controls])
        point_type_radio.change(set_point_type, inputs=[point_type_radio],
                                outputs=[review_status])
        reset_points_btn.click(reset_point_refine,
                               outputs=[review_viz, review_status])

        # Routed: while the scale panel is collecting the bar's ends, a click
        # on the frame is an end rather than a particle.
        review_viz.select(loading.review_click, outputs=[review_viz, review_status])

        check_bar_btn.click(scale.check_bar,
                            outputs=[scale_check_img, scale_status, scale_summary])
        confirm_scale_btn.click(scale.confirm,
                                outputs=[scale_status, scale_summary, header])
        points_btn.click(scale.start_points, outputs=[scale_status, review_viz])
        cancel_points_btn.click(scale.clear_points,
                                outputs=[scale_status, review_viz])
        apply_points_btn.click(
            scale.apply_points, inputs=[bar_value, bar_unit],
            outputs=[review_viz, scale_status, scale_summary, header,
                     frame_results])

        apply_btn.click(apply_refinement_changes,
                        outputs=[review_viz, frame_results, review_status,
                                 current_results, current_stats]).then(
            loading.current_header, outputs=[header])
        undo_btn.click(undo_last_action,
                       outputs=[review_viz, frame_results, review_status]).then(
            loading.current_header, outputs=[header])
        discard_btn.click(clear_pending_changes,
                          outputs=[review_viz, review_status])
        reload_btn.click(loading.reload_frame, outputs=REVIEW_OUTPUTS)

        clear_edges_btn.click(clear_edge_particles, inputs=[edge_buffer],
                              outputs=[review_viz, frame_results, review_status,
                                       current_results, current_stats]).then(
            loading.current_header, outputs=[header])
        clear_all_btn.click(clear_all_particles,
                            outputs=[review_viz, frame_results, review_status,
                                     current_results, current_stats]).then(
            loading.current_header, outputs=[header])
        min_size_slider.change(set_min_particle_size, inputs=[min_size_slider],
                               outputs=[review_status])
        show_numbers.change(toggle_particle_numbers, inputs=[show_numbers],
                            outputs=[review_viz])

        back_btn.click(loading.go_back, outputs=REVIEW_OUTPUTS)
        skip_btn.click(loading.skip_to_next, outputs=REVIEW_OUTPUTS)
        save_btn.click(loading.save_here, outputs=[save_status, gallery])
        save_next_btn.click(
            loading.save_and_next,
            outputs=[save_status, gallery, review_viz, header, frame_info,
                     frame_results, scale_summary])

        refresh_btn.click(get_session_summary,
                          outputs=[session_table, summary_progress,
                                   summary_stats, row_picker])
        dedupe_btn.click(check_and_remove_duplicates,
                         outputs=[session_table, export_status])
        delete_row_btn.click(
            delete_result_row, inputs=[row_picker],
            outputs=[session_table, summary_progress, summary_stats, row_picker,
                     export_status])
        export_btn.click(export_results, outputs=[export_file, export_status])
        plots_btn.click(update_histogram_plots, outputs=[plots_image, plots_status])

    return app
