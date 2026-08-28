"""
Refinement tab: click-driven add/delete/merge/point-refine.
"""
import gradio as gr
import numpy as np

from ..visualization import (
    create_particle_visualization,
    create_point_refine_visualization,
    create_results_dataframe,
    create_summary_statistics_table,
)
from ..state import state

def get_current_visualization():
    """Get current particle visualization with all pending changes."""
    if state.analyzer is None:
        return None
    return create_particle_visualization(
        state.cropped_image,
        state.analyzer.labeled_mask,
        state.analyzer.regions,
        show_labels=state.show_particle_numbers,
        pending_deletes=state.pending_deletes,
        pending_add_masks=state.pending_add_masks,
        pending_merge=state.pending_merge
    )


def handle_image_click(evt: gr.SelectData):
    """Handle clicks on particle visualization for all refinement modes."""
    try:
        if state.analyzer is None:
            return get_current_visualization(), "❌ No analysis available"

        x, y = evt.index[0], evt.index[1]

        if state.click_mode == "delete":
            # DELETE MODE: Click particles to remove them
            region, idx, label = state.analyzer.find_particle_at_point(x, y)
            if region is not None:
                if label not in state.pending_deletes:
                    # Save pending state before modification (for undo of this click)
                    state.save_pending_state()
                    state.pending_deletes.append(label)

                particle_viz = create_particle_visualization(
                    state.cropped_image,
                    state.analyzer.labeled_mask,
                    state.analyzer.regions,
                    show_labels=state.show_particle_numbers,
                    pending_deletes=state.pending_deletes,
                    pending_add_masks=state.pending_add_masks
                )
                return particle_viz, f"🟡 Queued particle #{idx+1} for deletion (yellow outline)"
            else:
                return get_current_visualization(), "No particle found at this location"

        elif state.click_mode == "add":
            # ADD MODE: Click to add a single particle at this location
            try:
                # Save pending state before modification (for undo of this click)
                state.save_pending_state()

                # Use single positive point WITHOUT base_mask to segment just the clicked particle
                refined_mask, score = state.segmenter.refine_with_sam(
                    state.cropped_image,
                    [[x, y]],
                    [1],
                    base_mask=None,  # Don't constrain to existing mask - segment the clicked object
                    multimask_output=True,
                    image_already_encoded=True
                )

                state.pending_add_points.append((x, y))
                state.pending_add_masks.append(refined_mask)

                particle_viz = create_particle_visualization(
                    state.cropped_image,
                    state.analyzer.labeled_mask,
                    state.analyzer.regions,
                    show_labels=state.show_particle_numbers,
                    pending_deletes=state.pending_deletes,
                    pending_add_masks=state.pending_add_masks
                )
                return particle_viz, f"🟢 Added particle preview (green outline) - score: {score:.3f}"
            except Exception as e:
                return get_current_visualization(), f"❌ SAM refinement failed: {str(e)}"

        elif state.click_mode == "merge":
            # MERGE MODE: Click multiple particles to merge them
            region, idx, label = state.analyzer.find_particle_at_point(x, y)
            if region is not None:
                if label not in state.pending_merge:
                    # Save pending state before modification (for undo of this click)
                    state.save_pending_state()
                    state.pending_merge.append(label)

                # Visualization will show selected particles in different color
                particle_viz = create_particle_visualization(
                    state.cropped_image,
                    state.analyzer.labeled_mask,
                    state.analyzer.regions,
                    pending_deletes=state.pending_deletes,
                    pending_add_masks=state.pending_add_masks,
                    pending_merge=state.pending_merge
                )
                return particle_viz, f"🔵 Selected {len(state.pending_merge)} particles for merging"
            else:
                return get_current_visualization(), "No particle found at this location"

        elif state.click_mode == "point_refine":
            # POINT REFINE MODE: Click anywhere to add positive/negative points with live preview
            # Users can refine existing particles OR create new ones from scratch

            # Save pending state before adding point (for undo of this click)
            state.save_pending_state()

            # Check if user clicked on an existing particle (only on first click)
            if len(state.point_refine_points) == 0:
                region, idx, label = state.analyzer.find_particle_at_point(x, y)
                if region is not None:
                    # User clicked on existing particle - use it as base mask for IoU selection
                    state.point_refine_particle = label
                    state.point_refine_base_mask = (state.analyzer.labeled_mask == label).astype(bool)
                # If no particle found, that's OK - user is creating a new particle from scratch

            # Add the point
            point_label = 1 if state.point_type == "positive" else 0
            state.point_refine_points.append((x, y))
            state.point_refine_labels.append(point_label)

            # Generate live preview with SAM using all accumulated points
            try:
                # Compute ROI box from base mask (with padding)
                if state.point_refine_base_mask is not None and state.point_refine_base_mask.any():
                    ys, xs = np.where(state.point_refine_base_mask)
                    y0, y1 = int(ys.min()), int(ys.max())
                    x0, x1 = int(xs.min()), int(xs.max())
                    pad = 10
                    H, W = state.cropped_image.shape[:2]
                    roi_box = np.array([[
                        max(0, x0 - pad),
                        max(0, y0 - pad),
                        min(W - 1, x1 + pad),
                        min(H - 1, y1 + pad)
                    ]])
                else:
                    roi_box = None

                # Call SAM with all points using iterative refinement
                if state.point_refine_logits is not None:
                    # Use previous logits for iterative refinement (like the notebook)
                    masks_out, scores, logits_out = state.segmenter.sam_model.predictor.predict(
                        point_coords=np.array(state.point_refine_points, dtype=float),
                        point_labels=np.array(state.point_refine_labels, dtype=int),
                        box=roi_box,
                        mask_input=state.point_refine_logits[None, ...],  # Use previous mask logits
                        multimask_output=False  # Single mask output for iterative refinement
                    )
                    refined_mask = masks_out[0].astype(bool)
                    state.point_refine_logits = logits_out[0]  # Store for next iteration
                else:
                    # First point: get initial masks and select best one
                    masks_out, scores, logits_out = state.segmenter.sam_model.predictor.predict(
                        point_coords=np.array(state.point_refine_points, dtype=float),
                        point_labels=np.array(state.point_refine_labels, dtype=int),
                        box=roi_box,
                        multimask_output=True  # Multiple masks for initial selection
                    )

                    # Select best mask by IoU with base mask
                    if state.point_refine_base_mask is not None:
                        ious = []
                        for mask in masks_out:
                            intersection = np.logical_and(mask, state.point_refine_base_mask).sum()
                            union = np.logical_or(mask, state.point_refine_base_mask).sum()
                            iou = intersection / union if union > 0 else 0
                            ious.append(iou)
                        best_idx = int(np.argmax(ious))
                    else:
                        best_idx = int(np.argmax(scores))

                    refined_mask = masks_out[best_idx].astype(bool)
                    state.point_refine_logits = logits_out[best_idx]  # Store for next iteration

                # Clean up the refined mask using user-specified minimum size
                from skimage import morphology
                refined_mask = morphology.remove_small_objects(refined_mask, min_size=state.min_particle_size)

                # Store as preview (will be applied when user clicks Apply)
                state.point_refine_preview_mask = refined_mask

                # Create visualization with point markers overlaid
                particle_viz = create_point_refine_visualization(
                    state.cropped_image,
                    refined_mask,
                    state.point_refine_points,
                    state.point_refine_labels
                )

                point_type_str = "positive ✓" if point_label == 1 else "negative ✗"
                return particle_viz, f"➕ Added {point_type_str} point ({len(state.point_refine_points)} total)"

            except Exception as e:
                point_type_str = "positive ✓" if point_label == 1 else "negative ✗"
                return get_current_visualization(), f"➕ Added {point_type_str} point - Preview update failed: {str(e)}"

        return get_current_visualization(), "Click registered"

    except Exception as e:
        return get_current_visualization(), f"❌ Error: {str(e)}"


def set_min_particle_size(size):
    """Set minimum particle size for filtering."""
    state.min_particle_size = int(size)
    return f"✓ Minimum particle size set to {int(size)} pixels"


def toggle_particle_numbers(show_numbers):
    """Toggle visibility of particle numbers in visualization."""
    state.show_particle_numbers = show_numbers
    # Return updated visualization
    return get_current_visualization()


def set_point_type(point_type):
    """Set point type for point refine mode."""
    state.point_type = point_type
    return f"✓ Point type: {point_type.upper()}"


def reset_point_refine():
    """Reset point refine state and redraw visualization."""
    try:
        state.point_refine_particle = None
        state.point_refine_base_mask = None
        state.point_refine_points = []
        state.point_refine_labels = []
        state.point_refine_preview_mask = None
        state.point_refine_logits = None

        # Redraw regular visualization
        if state.analyzer is not None:
            particle_viz = create_particle_visualization(
                state.cropped_image,
                state.analyzer.labeled_mask,
                state.analyzer.regions,
                show_labels=state.show_particle_numbers,
                pending_deletes=state.pending_deletes,
                pending_add_masks=state.pending_add_masks,
                pending_merge=state.pending_merge
            )
            return particle_viz, "✅ Reset point refinement"
        else:
            return None, "✅ Reset point refinement"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def set_click_mode(mode):
    """Set click mode and return status message."""
    state.click_mode = mode

    # Reset mode-specific state
    if mode != "merge":
        state.pending_merge = []
    if mode != "point_refine":
        state.point_refine_particle = None
        state.point_refine_base_mask = None
        state.point_refine_points = []
        state.point_refine_labels = []
        state.point_refine_preview_mask = None
        state.point_refine_logits = None

    # Return appropriate status message and visibility for point refine controls
    mode_messages = {
        "delete": "🗑️ DELETE mode: Click particles to remove them",
        "add": "➕ ADD mode: Click empty areas to add new particles",
        "merge": "🔗 MERGE mode: Click multiple touching particles to merge them",
        "point_refine": "🎯 POINT REFINE mode: Add positive/negative points to refine or create particles"
    }

    # Show point refine controls only in point_refine mode
    show_controls = (mode == "point_refine")

    return mode_messages.get(mode, "Unknown mode"), gr.update(visible=show_controls)


def apply_refinement_changes(progress=gr.Progress()):
    """Apply all pending refinement changes (delete, add, merge, point_refine)."""
    try:
        if state.analyzer is None:
            return gr.update(), gr.update(), "❌ No analysis available", gr.update(), gr.update()

        changes_made = False
        status_messages = []

        # Note: State is already saved on each click, no need to save here

        # Apply deletions
        if state.pending_deletes:
            progress(0.2, desc=f"Deleting {len(state.pending_deletes)} particles...")
            state.analyzer.delete_particles(state.pending_deletes)
            status_messages.append(f"Deleted {len(state.pending_deletes)} particles")
            state.pending_deletes = []
            changes_made = True

        # Apply additions using pre-generated masks
        if state.pending_add_masks:
            progress(0.4, desc=f"Adding {len(state.pending_add_masks)} particles...")
            for add_mask in state.pending_add_masks:
                state.analyzer.add_particle_from_sam(add_mask)
            status_messages.append(f"Added {len(state.pending_add_masks)} particles")
            state.pending_add_points = []
            state.pending_add_masks = []
            changes_made = True

        # Apply merge
        if state.pending_merge and len(state.pending_merge) >= 2:
            progress(0.6, desc=f"Merging {len(state.pending_merge)} particles...")
            state.analyzer.merge_particles(state.pending_merge)
            if getattr(state.analyzer, "last_merge_succeeded", True):
                status_messages.append(f"Merged {len(state.pending_merge)} particles")
            else:
                status_messages.append(
                    f"⚠️ Could not merge {len(state.pending_merge)} particles — they "
                    f"are too far apart to join"
                )
            state.pending_merge = []
            changes_made = True
        elif state.pending_merge:
            status_messages.append("⚠️ Need at least 2 particles to merge")
            state.pending_merge = []

        # Apply point refinement
        if state.point_refine_preview_mask is not None:
            progress(0.7, desc="Applying point refinement...")

            # Use the pre-generated preview mask (already refined during clicking)
            refining_existing = state.point_refine_particle is not None
            if refining_existing:
                # Refining existing particle - delete old and add refined one
                state.analyzer.delete_particles([state.point_refine_particle])
                status_messages.append(f"Refined particle with {len(state.point_refine_points)} points")
            else:
                # Creating new particle from scratch
                status_messages.append(f"Created new particle with {len(state.point_refine_points)} points")

            # Refining one particle must yield one particle. SAM's mask often has
            # stray disconnected blobs, and unioning those in turns a single
            # refinement into several new particles.
            state.analyzer.add_particle_from_sam(
                state.point_refine_preview_mask, largest_only=refining_existing
            )

            state.point_refine_particle = None
            state.point_refine_base_mask = None
            state.point_refine_points = []
            state.point_refine_labels = []
            state.point_refine_preview_mask = None
            state.point_refine_logits = None
            changes_made = True

        if not changes_made:
            return gr.update(), gr.update(), "No changes to apply", gr.update(), gr.update()

        # Clear undo history since changes have been applied
        state.undo_history = []

        progress(0.9, desc="Updating visualization...")

        # Update visualization (no pending changes now)
        particle_viz = create_particle_visualization(
            state.cropped_image,
            state.analyzer.labeled_mask,
            state.analyzer.regions,
            show_labels=state.show_particle_numbers
        )

        # Update measurements
        measurements = state.analyzer.get_measurements(in_nm=True)
        results_df = create_results_dataframe(measurements)
        stats_df = create_summary_statistics_table(measurements)

        progress(1.0, desc="Complete!")

        num_particles = len(state.analyzer.regions)
        status = f"✅ {' | '.join(status_messages)} | Total: {num_particles} particles"

        # Return: refine_viz, refine_results, refine_status, current_results, current_stats
        return (
            particle_viz,
            results_df,
            status,
            results_df,
            stats_df
        )

    except Exception as e:
        return gr.update(), gr.update(), f"❌ Error: {str(e)}", gr.update(), gr.update()


def undo_last_action():
    """Undo the last click (removes last item from pending changes)."""
    try:
        if state.analyzer is None:
            return gr.update(), gr.update(), "❌ No analysis available"

        if not state.undo_history:
            return gr.update(), gr.update(), "❌ No actions to undo"

        # Restore previous pending state (before last click)
        previous_state = state.undo_history.pop()

        # Restore only the pending changes (not the mask itself)
        state.pending_deletes = previous_state['pending_deletes']
        state.pending_add_points = previous_state['pending_add_points']
        state.pending_add_masks = previous_state['pending_add_masks']
        state.pending_merge = previous_state['pending_merge']
        state.point_refine_particle = previous_state['point_refine_particle']
        state.point_refine_base_mask = previous_state['point_refine_base_mask']
        state.point_refine_points = previous_state['point_refine_points']
        state.point_refine_labels = previous_state['point_refine_labels']
        state.point_refine_preview_mask = previous_state['point_refine_preview_mask']
        state.point_refine_logits = previous_state['point_refine_logits']

        # Update visualization based on mode
        if state.click_mode == "point_refine" and state.point_refine_preview_mask is not None:
            # Show point refine visualization with restored points
            particle_viz = create_point_refine_visualization(
                state.cropped_image,
                state.point_refine_preview_mask,
                state.point_refine_points,
                state.point_refine_labels
            )
        else:
            # Show regular visualization with restored pending changes
            particle_viz = create_particle_visualization(
                state.cropped_image,
                state.analyzer.labeled_mask,
                state.analyzer.regions,
                show_labels=state.show_particle_numbers,
                pending_deletes=state.pending_deletes,
                pending_add_masks=state.pending_add_masks,
                pending_merge=state.pending_merge
            )

        # Measurements don't change (we're only undoing pending changes)
        measurements = state.analyzer.get_measurements(in_nm=True)
        results_df = create_results_dataframe(measurements)

        num_particles = len(state.analyzer.regions)
        status = f"↩️ Undone last click! {num_particles} particles | {len(state.undo_history)} undo steps remaining"

        return particle_viz, results_df, status

    except Exception as e:
        return gr.update(), gr.update(), f"❌ Error: {str(e)}"


def clear_edge_particles(buffer_size):
    """Clear particles whose centroid is within buffer distance from edges."""
    try:
        if state.analyzer is None:
            return gr.update(), gr.update(), "❌ No analysis available", gr.update(), gr.update()

        buffer = int(buffer_size)
        H, W = state.analyzer.mask.shape

        # Identify particles to remove based on centroid position
        labels_to_remove = []
        for region in state.analyzer.regions:
            y, x = region.centroid
            # Check if centroid is within buffer distance from any edge
            if (x < buffer or x > W - buffer or
                y < buffer or y > H - buffer):
                labels_to_remove.append(region.label)

        n_removed = len(labels_to_remove)

        if n_removed > 0:
            state.analyzer.delete_particles(labels_to_remove)

        # Update visualization
        particle_viz = create_particle_visualization(
            state.cropped_image,
            state.analyzer.labeled_mask,
            state.analyzer.regions,
            show_labels=state.show_particle_numbers
        )

        # Update measurements
        measurements = state.analyzer.get_measurements(in_nm=True)
        results_df = create_results_dataframe(measurements)
        stats_df = create_summary_statistics_table(measurements)

        num_particles = len(state.analyzer.regions)
        status = f"✅ Removed {n_removed} edge particles (centroid within {buffer}px of edge) - Now {num_particles} total"

        # Return: refine_viz, refine_results, refine_status, current_results, current_stats
        return (
            particle_viz,
            results_df,
            status,
            results_df,
            stats_df
        )

    except Exception as e:
        return gr.update(), gr.update(), f"❌ Error: {str(e)}", gr.update(), gr.update()


def clear_all_particles():
    """Clear ALL particles from the mask, giving user a blank canvas to add particles manually."""
    try:
        if state.analyzer is None:
            return gr.update(), gr.update(), "❌ No analysis available", gr.update(), gr.update()

        # Get all current labels and delete them all
        all_labels = [r.label for r in state.analyzer.regions]
        n_removed = len(all_labels)

        if n_removed > 0:
            state.analyzer.delete_particles(all_labels)

        # Clear all pending state too
        state.pending_deletes = []
        state.pending_add_points = []
        state.pending_add_masks = []
        state.pending_merge = []
        state.point_refine_particle = None
        state.point_refine_base_mask = None
        state.point_refine_points = []
        state.point_refine_labels = []
        state.point_refine_preview_mask = None
        state.point_refine_logits = None
        state.undo_history = []

        # Redraw (empty image, no particles)
        particle_viz = create_particle_visualization(
            state.cropped_image,
            state.analyzer.labeled_mask,
            state.analyzer.regions,
            show_labels=state.show_particle_numbers
        )

        # Empty measurements
        measurements = state.analyzer.get_measurements(in_nm=True)
        results_df = create_results_dataframe(measurements)
        stats_df = create_summary_statistics_table(measurements)

        status = f"🗑️ Cleared all {n_removed} particles. Use 'add' mode to start segmenting."

        return particle_viz, results_df, status, results_df, stats_df

    except Exception as e:
        return gr.update(), gr.update(), f"❌ Error: {str(e)}", gr.update(), gr.update()


def clear_pending_changes():
    """Clear all pending changes and redraw visualization."""
    try:
        state.pending_deletes = []
        state.pending_add_points = []
        state.pending_add_masks = []
        state.pending_merge = []
        state.point_refine_particle = None
        state.point_refine_base_mask = None
        state.point_refine_points = []
        state.point_refine_labels = []
        state.point_refine_preview_mask = None
        state.point_refine_logits = None
        state.undo_history = []  # Clear undo history since all pending changes are cleared

        # Redraw visualization without pending changes
        if state.analyzer is not None:
            particle_viz = create_particle_visualization(
                state.cropped_image,
                state.analyzer.labeled_mask,
                state.analyzer.regions,
                pending_deletes=[],
                pending_add_masks=[],
                pending_merge=[]
            )
            return particle_viz, "✅ Cleared all pending changes"
        else:
            return gr.update(), "✅ Cleared all pending changes"
    except Exception as e:
        return gr.update(), f"❌ Error: {str(e)}"
