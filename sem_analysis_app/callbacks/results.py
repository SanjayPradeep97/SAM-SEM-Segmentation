"""
Results tab: saving, summarising and exporting measurements.
"""
import os
import gradio as gr
import numpy as np

from ..state import state
from .gallery import create_image_gallery

def save_current_results():
    """Save current image results and mark as processed."""
    try:
        if state.analyzer is None:
            return "❌ No analysis to save", None

        if state.results_manager is None:
            return "❌ Results manager not initialized", None

        # Get measurements
        measurements = state.analyzer.get_measurements(in_nm=True)

        # Save to CSV
        filename = os.path.basename(state.image_paths[state.current_index])
        state.results_manager.add_result(filename, measurements)

        # Mark as processed
        state.mark_processed(state.current_index, measurements['num_particles'])
        state.save_state()

        # Update gallery
        gallery_data = create_image_gallery()

        return (
            f"✅ Saved results for {filename}",
            gallery_data
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", None


def get_session_summary():
    """Get summary of all processed images with aggregate particle statistics."""
    try:
        if state.results_manager is None:
            return None, "No results available", "No particle statistics available", gr.update(choices=[], value=None)

        results_df = state.results_manager.get_results()

        if len(results_df) == 0:
            return None, "No images processed yet", "No particle statistics available", gr.update(choices=[], value=None)

        summary = state.results_manager.get_summary()

        # Progress summary (left column)
        progress_text = f"""
### Session Progress

- **Images Processed:** {summary['total_images']} / {len(state.image_paths)}
- **Total Particles:** {summary['total_particles']}
- **Average Particles/Image:** {summary['avg_particles_per_image']:.1f}
- **Min Particles/Image:** {summary['min_particles']}
- **Max Particles/Image:** {summary['max_particles']}
        """

        # Aggregate particle statistics (right column)
        # Collect all particle diameters from all images
        all_diameters = []
        for idx, row in results_df.iterrows():
            diams_nm = eval(row['equiv_diameters_nm']) if row['equiv_diameters_nm'] != '[]' else []
            all_diameters.extend(diams_nm)

        if len(all_diameters) > 0:
            mean_diam = np.mean(all_diameters) / 1000  # Convert to μm
            median_diam = np.median(all_diameters) / 1000
            std_diam = np.std(all_diameters) / 1000
            min_diam = np.min(all_diameters) / 1000
            max_diam = np.max(all_diameters) / 1000

            particle_stats_text = f"""
### Aggregate Particle Statistics

- **Mean Diameter:** {mean_diam:.3f} μm
- **Median Diameter:** {median_diam:.3f} μm
- **Std Deviation:** {std_diam:.3f} μm
- **Min Diameter:** {min_diam:.3f} μm
- **Max Diameter:** {max_diam:.3f} μm
            """
        else:
            particle_stats_text = "### Aggregate Particle Statistics\n\nNo particle data available"

        display_df = results_df[['file_name', 'num_particles']].copy()
        display_df.columns = ['Filename', 'Particle Count']

        # Create dropdown choices for delete functionality
        # Format: "index: filename" so user can see both
        dropdown_choices = [f"{i}: {row['file_name']}" for i, row in results_df.iterrows()]

        return display_df, progress_text, particle_stats_text, gr.update(choices=dropdown_choices, value=None)

    except Exception as e:
        return None, f"❌ Error: {str(e)}", f"❌ Error: {str(e)}", gr.update(choices=[], value=None)


def delete_result_row(selected_item):
    """Delete a specific row from the results table."""
    try:
        if state.results_manager is None:
            return None, None, None, gr.update(), "❌ No results available"

        if selected_item is None:
            return None, None, None, gr.update(), "❌ Please select a row to delete"

        results_df = state.results_manager.get_results()

        if len(results_df) == 0:
            return None, None, None, gr.update(), "❌ No results to delete"

        # Parse the selection format "index: filename"
        idx_str = selected_item.split(":")[0].strip()
        idx = int(idx_str)

        # Get filename for confirmation message
        filename = results_df.loc[idx, 'file_name']

        # Delete the row
        state.results_manager.delete_result(idx)

        # Get updated summary (includes new dropdown choices)
        display_df, progress_text, particle_stats_text, dropdown_update = get_session_summary()

        return display_df, progress_text, particle_stats_text, dropdown_update, f"✅ Deleted: {filename}"

    except IndexError as e:
        return None, None, None, gr.update(), f"❌ Index error: {str(e)}"
    except Exception as e:
        return None, None, None, gr.update(), f"❌ Error: {str(e)}"


def export_results():
    """Export all results to CSV."""
    try:
        if state.results_manager is None or len(state.results_manager.results_df) == 0:
            return None

        csv_path = state.results_manager.csv_file
        return csv_path

    except Exception:
        return None


def check_and_remove_duplicates():
    """Check for duplicate entries and remove them."""
    try:
        if state.results_manager is None:
            return None, "❌ No results manager initialized"

        # Find duplicates
        duplicates = state.results_manager.find_duplicates()

        if len(duplicates) == 0:
            return None, "✅ No duplicate entries found"

        # Delete duplicates (keep last occurrence)
        deleted_count = state.results_manager.delete_duplicates(keep='last')

        # Get updated summary
        results_df = state.results_manager.get_results()
        if len(results_df) == 0:
            display_df = None
        else:
            display_df = results_df[['file_name', 'num_particles']].copy()
            display_df.columns = ['Filename', 'Particle Count']

        return display_df, f"✅ Removed {deleted_count} duplicate entries (kept last occurrence)"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"
