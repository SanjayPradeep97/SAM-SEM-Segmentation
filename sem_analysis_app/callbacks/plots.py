"""
Plotting tab: session-wide histograms.
"""

from ..state import state

def update_histogram_plots():
    """Generate histogram plots for all particles from all images in the session."""
    try:
        if state.results_manager is None:
            return None, "❌ No results manager initialized. Save some image results first."

        # Get all results
        results_df = state.results_manager.get_results()

        if len(results_df) == 0:
            return None, "❌ No saved results to plot. Save at least one image's results first."

        # Collect all particle measurements from all images
        all_areas = []
        all_diameters = []

        for idx, row in results_df.iterrows():
            # Parse the string lists back to arrays
            areas_nm2 = eval(row['particle_areas_nm2']) if row['particle_areas_nm2'] != '[]' else []
            diams_nm = eval(row['equiv_diameters_nm']) if row['equiv_diameters_nm'] != '[]' else []

            all_areas.extend(areas_nm2)
            all_diameters.extend(diams_nm)

        total_particles = len(all_areas)

        if total_particles == 0:
            return None, "❌ No particles found in saved results"

        # Create combined measurements dictionary
        combined_measurements = {
            'num_particles': total_particles,
            'areas': all_areas,
            'diameters': all_diameters,
            'unit': 'nm'
        }

        # Create histograms
        from visualization import create_histogram_plots
        histogram_img = create_histogram_plots(combined_measurements)

        return histogram_img, f"✅ Generated histograms for {total_particles} particles from {len(results_df)} images"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"
