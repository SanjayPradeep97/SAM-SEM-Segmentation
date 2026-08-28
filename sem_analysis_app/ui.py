"""
Gradio interface definition: tabs, controls and event wiring.
"""
import gradio as gr

from sem_particle_analysis import discover_checkpoints
from .callbacks.scale_tab import UNITS


from .callbacks import (
    auto_process_current_image,
    restore_session,
    apply_two_points,
    clear_scale,
    confirm_scale,
    live_point_readout,
    prepare_scale_tab,
    read_box_scale,
    set_canvas_mode,

    save_and_next,
    skip_to_next,
    adjust_crop,
    apply_manual_scale,
    apply_refinement_changes,
    check_and_remove_duplicates,
    clear_all_particles,
    clear_edge_particles,
    clear_pending_changes,
    delete_result_row,
    detect_scale_clicked,
    export_results,
    get_session_summary,
    handle_image_click,
    handle_scale_click,
    initialize_sam,
    load_images_from_folder,
    reset_point_refine,
    reset_scale_clicks,
    resume_session,
    save_current_results,
    segment_with_sam,
    select_image_from_gallery,
    select_mask_and_analyze,
    set_click_mode,
    set_min_particle_size,
    set_point_type,
    set_scale_mode,
    toggle_particle_numbers,
    undo_last_action,
    update_histogram_plots,
)

APP_CSS = """
.tabs {font-size: 16px; font-weight: 500;}
.tab-nav button {padding: 12px 24px;}
/* Data channels between the scale canvas and Python. They must exist in the
   DOM for client code to write to them, so they are hidden rather than absent. */
.scale-channel {display: none !important;}
"""


def create_interface():
    """
    Build the tabbed Gradio interface.

    Theme and CSS are applied at launch (see __main__), which is where Gradio 6
    expects them; passing them to Blocks is deprecated.
    """
    with gr.Blocks(title="SAM-based SEM Particle Analysis") as app:

        gr.Markdown("# 🔬 SAM-based SEM Particle Analysis")
        gr.Markdown("Automated particle segmentation and analysis for scanning electron microscopy images using Segment Anything Model")

        with gr.Tabs() as tabs:

            # ================================================================
            # TAB 1: Setup
            # ================================================================
            with gr.Tab("⚙️ Setup", id=0):
                gr.Markdown("## Initialize SAM Model and Load Images")

                with gr.Row():
                    with gr.Column(scale=2):
                        # A dropdown of checkpoints already on disk, rather than an
                        # upload widget: Gradio copies uploaded files into its
                        # cache, which meant shuffling 2.4 GB on every start.
                        # The architecture is implied by the checkpoint, so there
                        # is no separate Model Type control to contradict it —
                        # a mismatched pair fails deep inside torch with an
                        # unhelpful shape error.
                        _checkpoints = discover_checkpoints()
                        sam_file = gr.Dropdown(
                            label="SAM Checkpoint",
                            # Plain path strings, not (label, value) pairs: with
                            # allow_custom_value the dropdown hands back what is
                            # displayed, so a pretty label arrives at the callback
                            # instead of the path and the load fails.
                            choices=[str(p) for p in _checkpoints],
                            value=str(_checkpoints[0]) if _checkpoints else None,
                            allow_custom_value=True,
                            info="Found in sam_weights/. Paste a path to use one "
                                 "from elsewhere." if _checkpoints else
                                 "None found — run python download_sam_weights.py, "
                                 "or paste a path.",
                        )
                        init_sam_btn = gr.Button("⚡ Initialize SAM Model", variant="primary", size="lg")
                        init_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=2):
                        gr.Markdown("**Load Images**")
                        gr.Markdown("Select one or more images. If you select multiple images from the same folder, all images in that folder will be loaded.")
                        file_input = gr.File(
                            label="Select Image Files",
                            file_count="multiple",
                            file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                            type="filepath"
                        )

                        load_btn = gr.Button("📁 Load Images", variant="primary", size="lg", interactive=False)
                        load_status = gr.Textbox(label="Status", interactive=False)

                        gr.Markdown("---")
                        gr.Markdown("**Resume Previous Session**")
                        gr.Markdown("Upload a previous results CSV to continue adding to it.")
                        resume_csv_input = gr.File(
                            label="Select Results CSV",
                            file_count="single",
                            file_types=[".csv"],
                            type="filepath"
                        )
                        resume_btn = gr.Button("📂 Resume Session", variant="secondary")
                        resume_status = gr.Textbox(label="Resume Status", interactive=False)

            # ================================================================
            # TAB 2: Image Gallery
            # ================================================================
            with gr.Tab("🖼️ Image Gallery", id=1):
                gr.Markdown("## Image Gallery - Click to Select")
                gr.Markdown("✅ = Processed | ⚪ = Not Processed")

                gallery = gr.Gallery(
                    label="All Images",
                    columns=5,
                    rows=4,
                    height=800,
                    object_fit="contain",
                    show_label=True,
                    # The preview overlay has no reliable way out and blocks
                    # the workflow; clicking a thumbnail should select it.
                    allow_preview=False,
                    type="pil"  # Explicitly set to PIL Image type for cross-platform compatibility
                )

                selected_image_info = gr.Textbox(label="Selected Image", interactive=False)


            # ================================================================
            # TAB 3: Scale
            # ================================================================
            with gr.Tab("📏 Scale", id=2):
                gr.Markdown("## Establish the image scale")
                gr.Markdown(
                    "Every measurement is a pixel count times this one number, so "
                    "it is worth getting right. Tier 1 runs automatically; only "
                    "fall through to tier 2 or 3 if it fails."
                )

                tier1_status = gr.Textbox(
                    label="Tier 1 — file metadata (automatic)", interactive=False, lines=2
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        canvas_mode = gr.Radio(
                            choices=["Tier 2 — draw a box and read it",
                                     "Tier 3 — click both ends of the bar"],
                            value="Tier 2 — draw a box and read it",
                            label="If tier 1 failed, pick a method",
                        )
                        # Channels the canvas JS reads from and writes into.
                        # Hidden with CSS rather than visible=False: Gradio 6
                        # omits invisible components from the DOM entirely, and
                        # client code cannot write to an element that isn't there.
                        _chan = dict(elem_classes=["scale-channel"], show_label=False,
                                     container=False)
                        scale_image_in = gr.Textbox(elem_id="scale_image_in", **_chan)
                        scale_box_out = gr.Textbox(elem_id="scale_box_out", **_chan)
                        scale_points_out = gr.Textbox(elem_id="scale_points_out", **_chan)

                        gr.HTML(
                            '<div style="width:100%">'
                            '<canvas id="scale_canvas" '
                            'style="max-width:100%;border-radius:8px;'
                            'border:1px solid rgba(128,128,128,0.35)"></canvas>'
                            '</div>'
                        )

                    with gr.Column(scale=2):
                        with gr.Group(visible=True) as tier2_controls:
                            gr.Markdown(
                                "**Tier 2.** Drag on the image to draw a box around "
                                "the scale bar *and* its label. Drag a corner to "
                                "resize, drag the middle to move."
                            )
                            read_box_btn = gr.Button("🔍 Read scale in box",
                                                     variant="primary")

                        with gr.Group(visible=False) as tier3_controls:
                            gr.Markdown(
                                "**Tier 3.** Click each end of the bar. A magnifier "
                                "follows the cursor so you can land on the exact "
                                "pixel, then type the printed length."
                            )
                            point_readout = gr.Textbox(label="Points", interactive=False)
                            with gr.Row():
                                bar_value = gr.Number(label="Printed length", value=None)
                                bar_unit = gr.Dropdown(choices=UNITS, value="µm",
                                                       label="Unit")
                            apply_points_btn = gr.Button("📏 Apply two-point scale",
                                                         variant="primary")

                        reset_canvas_btn = gr.Button("↺ Reset box / points", size="sm")
                        tier_status = gr.Textbox(label="Result", interactive=False, lines=3)

                gr.Markdown("---")
                scale_summary = gr.Textbox(
                    label="Scale in force for this image", interactive=False, lines=4
                )
                with gr.Row():
                    confirm_scale_btn = gr.Button("✔️ Confirm scale", variant="primary")
                    clear_scale_btn = gr.Button("✖️ Clear scale", variant="secondary")

            # ================================================================
            # TAB 3: Processing
            # ================================================================
            with gr.Tab("🔍 Processing", id=3):
                gr.Markdown("## Scale Detection and Segmentation")

                current_image = gr.Image(
                    label="",
                    type="numpy"
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Step 1: Scale Detection")
                        scale_mode_dropdown = gr.Dropdown(
                            choices=["Metadata", "OCR", "Manual"],
                            value="Metadata",
                            label="Detection Mode",
                            info="Metadata: TIFF tags | OCR: click box on image | Manual: click endpoints"
                        )
                        detect_btn = gr.Button("🔍 Detect Scale", variant="primary")
                        crop_percent_slider = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=7.0,
                            step=0.5,
                            label="Bottom Crop (%)",
                            info="Adjust how much of the bottom to crop (removes scale bar / databar)"
                        )
                        scale_status = gr.Textbox(label="Status", interactive=False)

                        # OCR mode controls (hidden by default)
                        with gr.Group(visible=False) as ocr_controls:
                            gr.Markdown("**OCR Mode:** Click two opposite corners on the image to define the search box.")
                            ocr_info = gr.Textbox(
                                label="OCR Info",
                                value="Click point 1 of 2...",
                                interactive=False
                            )
                            ocr_reset_btn = gr.Button("Reset OCR Box", size="sm")

                        # Manual mode controls (hidden by default)
                        with gr.Group(visible=False) as manual_controls:
                            gr.Markdown("**Manual Mode:** Click the left end of the scale bar, then the right end.")
                            manual_info = gr.Textbox(
                                label="Manual Info",
                                value="Click point 1 of 2...",
                                interactive=False
                            )
                            manual_um_input = gr.Number(
                                label="Scale bar length (µm)",
                                info="Enter the physical length of the scale bar in micrometers",
                                precision=3,
                                minimum=0.001,
                                maximum=100000
                            )
                            manual_apply_btn = gr.Button("Apply Manual Scale", variant="primary")
                            manual_reset_btn = gr.Button("Reset Points", size="sm")

                    with gr.Column():
                        gr.Markdown("### Step 2: Segmentation")
                        min_particle_size_slider = gr.Slider(
                            minimum=5,
                            maximum=100,
                            value=30,
                            step=1,
                            label="Minimum Particle Size (pixels)",
                            info="Particles smaller than this will be filtered out. Lower for small/low-mag images."
                        )
                        segment_btn = gr.Button("🤖 Segment with SAM", variant="primary", size="lg")
                        segment_status = gr.Textbox(label="Status", interactive=False)

                mask_viz = gr.Image(label="")

                with gr.Row():
                    mask_choice = gr.Radio(
                        choices=["Option 1", "Option 2", "Option 3"],
                        value="Option 1",
                        label="Select Best Mask",
                        info="Ranked best-first by contrast; see captions above"
                    )
                    analyze_btn = gr.Button("✓ Select & Analyze", variant="primary", size="lg")

                analysis_status = gr.Textbox(label="Analysis Status", interactive=False)

            # ================================================================
            # TAB 4: Refinement
            # ================================================================
            with gr.Tab("✏️ Refinement", id=4):
                gr.Markdown("## Interactive Particle Refinement")
                gr.Markdown("**Select a mode and click on the image to refine particle segmentation**")

                with gr.Row():
                    with gr.Column(scale=1):
                        click_mode_radio = gr.Radio(
                            choices=["delete", "add", "merge", "point_refine"],
                            value="delete",
                            label="Refinement Mode",
                            info="Select how you want to interact with particles"
                        )

                        gr.Markdown("""
**Mode Descriptions:**
- **Delete**: Click particles to remove them
- **Add**: Click empty areas to add new particles
- **Merge**: Click multiple touching particles to merge them
- **Point Refine**: Click to add positive/negative points to refine a selected particle
                        """)

                    with gr.Column(scale=2):
                        click_mode_status = gr.Textbox(
                            label="Current Mode",
                            value="🗑️ DELETE mode: Click particles to remove them",
                            interactive=False
                        )

                        # Point refine controls (only visible in point_refine mode)
                        with gr.Group(visible=False) as point_refine_controls:
                            gr.Markdown("**Point Refinement Controls**")
                            point_type_radio = gr.Radio(
                                choices=["positive", "negative"],
                                value="positive",
                                label="Point Type",
                                info="Positive = include, Negative = exclude"
                            )
                            reset_points_btn = gr.Button("Reset Points", size="sm")

                # Visualization controls
                show_numbers_checkbox = gr.Checkbox(
                    label="Show Particle Numbers",
                    value=True,
                    info="Uncheck to hide numbers for better visibility of small particles"
                )

                refine_viz = gr.Image(label="", type="numpy")

                with gr.Row():
                    apply_btn = gr.Button("✓ Apply Changes", variant="primary", size="lg")
                    clear_pending_btn = gr.Button("Clear Pending", variant="secondary")
                    undo_btn = gr.Button("↩️ Undo Last Action", variant="secondary")

                with gr.Row():
                    edge_buffer = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Edge Buffer (pixels)"
                    )
                    with gr.Column():
                        clear_edges_btn = gr.Button("🧹 Clear Edge Particles", variant="secondary")
                        clear_all_btn = gr.Button("🗑️ Clear All Particles", variant="stop")

                refine_status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("Particle Measurements", open=True):
                    refine_results = gr.Dataframe(label="Particle Measurements")

                gr.Markdown("---")
                gr.Markdown("### Save Results")
                gr.Markdown(
                    "**Save & Next** records this image and immediately loads and "
                    "pre-processes the following one — the normal way to work "
                    "through a folder."
                )
                with gr.Row():
                    save_next_btn = gr.Button("✅ Save & Next Image", variant="primary", size="lg")
                    save_btn = gr.Button("💾 Save (stay here)", variant="secondary", size="lg")
                    skip_btn = gr.Button("⏭️ Skip (don't save)", variant="secondary")
                save_status = gr.Textbox(label="Status", interactive=False)

            # ================================================================
            # TAB 5: Results & Export
            # ================================================================
            with gr.Tab("💾 Results & Export", id=5):
                gr.Markdown("## Current Image Analysis Summary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Particle Measurements")
                        current_results = gr.Dataframe(label="All Particles")

                    with gr.Column():
                        gr.Markdown("### Summary Statistics")
                        current_stats = gr.Dataframe(label="Statistics")

                gr.Markdown("---")
                gr.Markdown("## Session Summary (All Images)")

                refresh_btn = gr.Button("🔄 Refresh Summary")

                # Two-column layout for session statistics
                with gr.Row():
                    with gr.Column():
                        summary_progress = gr.Markdown("No results yet")
                    with gr.Column():
                        summary_particle_stats = gr.Markdown("No particle statistics yet")

                session_table = gr.Dataframe(label="All Processed Images")

                with gr.Row():
                    delete_row_dropdown = gr.Dropdown(
                        label="Select Row to Delete",
                        choices=[],
                        value=None,
                        interactive=True,
                        info="Choose a file to delete from results"
                    )
                    delete_row_btn = gr.Button("❌ Delete Selected Row", variant="stop")

                delete_row_status = gr.Textbox(label="Delete Status", interactive=False)

                with gr.Row():
                    remove_duplicates_btn = gr.Button("🧹 Remove Duplicate Entries", variant="secondary")
                    duplicates_status = gr.Textbox(label="Duplicate Removal Status", interactive=False)

                gr.Markdown("---")
                gr.Markdown("## Export Results")

                with gr.Row():
                    export_btn = gr.Button("📥 Export All Results", variant="primary")
                    export_file = gr.File(label="Download CSV")

            # ================================================================
            # TAB 6: Plotting & Graphs
            # ================================================================
            with gr.Tab("📊 Plotting & Graphs", id=6):
                gr.Markdown("## Particle Size Distribution")
                gr.Markdown("Visualize the distribution of particle sizes with histograms")

                with gr.Row():
                    update_plots_btn = gr.Button("🔄 Update Plots", variant="primary", size="lg")
                    plot_status = gr.Textbox(label="Status", interactive=False)

                histogram_plot = gr.Image(label="", type="numpy")

        # ================================================================
        # Event Handlers
        # ================================================================

        # Setup tab
        init_sam_btn.click(
            initialize_sam,
            inputs=[sam_file],
            outputs=[init_status, load_btn]
        )

        load_btn.click(
            load_images_from_folder,
            inputs=[file_input],
            outputs=[load_status, gallery]
        )

        resume_btn.click(
            resume_session,
            inputs=[resume_csv_input],
            outputs=[resume_status, gallery, load_status]
        )

        # Gallery tab. Selecting an image immediately detects scale and produces
        # ranked candidates, so the analyst lands on masks to judge rather than
        # on buttons to press.
        gallery.select(
            select_image_from_gallery,
            outputs=[current_image, selected_image_info, tabs, click_mode_radio]
        ).then(
            prepare_scale_tab,
            outputs=[scale_image_in, tier1_status, scale_summary, point_readout]
        ).then(
            None, js="() => { window.SCALE && window.SCALE.load(); }"
        ).then(
            auto_process_current_image,
            outputs=[scale_status, current_image, crop_percent_slider,
                     mask_viz, segment_status, mask_choice]
        )


        # A refresh reconnects to the same process-wide session, so bring the
        # gallery back rather than showing an empty one.
        app.load(restore_session, outputs=[gallery, load_status])

        # --- Scale tab ---
        canvas_mode.change(
            set_canvas_mode,
            inputs=[canvas_mode],
            outputs=[tier2_controls, tier3_controls]
        ).then(
            None, inputs=[canvas_mode],
            js="(m) => { window.SCALE && window.SCALE.setMode("
               "m.startsWith('Tier 3') ? 'points' : 'box'); }"
        )

        read_box_btn.click(
            read_box_scale,
            inputs=[scale_box_out],
            outputs=[tier_status, scale_summary]
        )

        # Live feedback as the two points are placed.
        scale_points_out.change(
            live_point_readout,
            inputs=[scale_points_out],
            outputs=[point_readout]
        )

        apply_points_btn.click(
            apply_two_points,
            inputs=[scale_points_out, bar_value, bar_unit],
            outputs=[tier_status, scale_summary]
        )

        reset_canvas_btn.click(
            None, js="() => { window.SCALE && window.SCALE.reset(); }"
        )

        confirm_scale_btn.click(
            confirm_scale, outputs=[tier_status, scale_summary]
        )

        clear_scale_btn.click(
            clear_scale, outputs=[tier_status, scale_summary]
        )

        # Processing tab — scale detection mode
        scale_mode_dropdown.change(
            set_scale_mode,
            inputs=[scale_mode_dropdown],
            outputs=[ocr_controls, manual_controls, scale_status, current_image, ocr_info, manual_info]
        )

        # Click handler on processing image for OCR / Manual modes
        current_image.select(
            handle_scale_click,
            outputs=[current_image, scale_status, ocr_info, manual_info]
        )

        detect_btn.click(
            detect_scale_clicked,
            outputs=[scale_status, current_image, crop_percent_slider]
        )

        crop_percent_slider.change(
            adjust_crop,
            inputs=[crop_percent_slider],
            outputs=[scale_status, current_image]
        )

        # OCR mode reset
        ocr_reset_btn.click(
            reset_scale_clicks,
            outputs=[current_image, ocr_info]
        )

        # Manual mode controls
        manual_apply_btn.click(
            apply_manual_scale,
            inputs=[manual_um_input],
            outputs=[scale_status, current_image, manual_info]
        )

        manual_reset_btn.click(
            reset_scale_clicks,
            outputs=[current_image, manual_info]
        )

        min_particle_size_slider.change(
            set_min_particle_size,
            inputs=[min_particle_size_slider],
            outputs=[segment_status]
        )

        segment_btn.click(
            segment_with_sam,
            outputs=[mask_viz, segment_status, mask_choice]
        )

        analyze_btn.click(
            select_mask_and_analyze,
            inputs=[mask_choice],
            outputs=[refine_viz, refine_results, analysis_status, current_results, current_stats]
        )

        # Refinement tab
        click_mode_radio.change(
            set_click_mode,
            inputs=[click_mode_radio],
            outputs=[click_mode_status, point_refine_controls]
        )

        point_type_radio.change(
            set_point_type,
            inputs=[point_type_radio],
            outputs=[refine_status]
        )

        reset_points_btn.click(
            reset_point_refine,
            outputs=[refine_viz, refine_status]
        )

        show_numbers_checkbox.change(
            toggle_particle_numbers,
            inputs=[show_numbers_checkbox],
            outputs=[refine_viz]
        )

        refine_viz.select(
            handle_image_click,
            outputs=[refine_viz, refine_status]
        )

        apply_btn.click(
            apply_refinement_changes,
            outputs=[refine_viz, refine_results, refine_status, current_results, current_stats]
        )

        clear_pending_btn.click(
            clear_pending_changes,
            outputs=[refine_viz, refine_status]
        )

        undo_btn.click(
            undo_last_action,
            outputs=[refine_viz, refine_results, refine_status]
        )

        clear_edges_btn.click(
            clear_edge_particles,
            inputs=[edge_buffer],
            outputs=[refine_viz, refine_results, refine_status, current_results, current_stats]
        )

        clear_all_btn.click(
            clear_all_particles,
            outputs=[refine_viz, refine_results, refine_status, current_results, current_stats]
        )

        # Plotting tab
        update_plots_btn.click(
            update_histogram_plots,
            outputs=[histogram_plot, plot_status]
        )

        # Results tab
        save_btn.click(
            save_current_results,
            outputs=[save_status, gallery]
        )

        save_next_btn.click(
            save_and_next,
            outputs=[save_status, gallery, selected_image_info, current_image,
                     scale_status, crop_percent_slider, mask_viz, segment_status,
                     mask_choice, tabs, scale_image_in, tier1_status,
                     scale_summary, point_readout]
        ).then(
            None, js="() => { window.SCALE && window.SCALE.load(); }"
        )

        skip_btn.click(
            skip_to_next,
            outputs=[selected_image_info, current_image, tabs, scale_image_in,
                     tier1_status, scale_summary, point_readout]
        ).then(
            None, js="() => { window.SCALE && window.SCALE.load(); }"
        ).then(
            auto_process_current_image,
            outputs=[scale_status, current_image, crop_percent_slider,
                     mask_viz, segment_status, mask_choice]
        )

        refresh_btn.click(
            get_session_summary,
            outputs=[session_table, summary_progress, summary_particle_stats, delete_row_dropdown]
        )

        delete_row_btn.click(
            delete_result_row,
            inputs=[delete_row_dropdown],
            outputs=[session_table, summary_progress, summary_particle_stats, delete_row_dropdown, delete_row_status]
        )

        remove_duplicates_btn.click(
            check_and_remove_duplicates,
            outputs=[session_table, duplicates_status]
        )

        export_btn.click(
            export_results,
            outputs=[export_file]
        )

    return app
