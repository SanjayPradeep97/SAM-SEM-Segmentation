"""
Results Data Management

Handles saving and loading of particle analysis results to CSV files.
"""

import os
import pandas as pd
import random


class ResultsManager:
    """
    Manages storage and retrieval of particle analysis results.

    Attributes:
        csv_file (str): Path to CSV results file
        results_df (pd.DataFrame): Current results dataframe
    """

    def __init__(self, csv_file=None, auto_create=True):
        """
        Initialize the results manager.

        Args:
            csv_file (str, optional): Path to CSV file. If None, creates a unique file.
            auto_create (bool): Whether to auto-create the CSV if it doesn't exist
        """
        if csv_file is None and auto_create:
            # Generate unique filename
            suffix = random.randint(1000, 9999)
            csv_file = f"analysis_results_{suffix}.csv"

        self.csv_file = csv_file

        # Define column structure
        self.columns = [
            "file_name",
            "num_particles",
            "particle_areas_px",
            "equiv_diameters_px",
            "particle_areas_nm2",
            "equiv_diameters_nm"
        ]

        # Create or load CSV
        if auto_create and csv_file is not None:
            if not os.path.exists(csv_file):
                self._create_csv()
                print(f"Created new results file: {csv_file}")
            else:
                print(f"Using existing results file: {csv_file}")

        # Load existing data
        self.results_df = self._load_data()

    def _create_csv(self):
        """Create a new CSV file with headers."""
        pd.DataFrame(columns=self.columns).to_csv(self.csv_file, index=False)

    def _load_data(self):
        """Load data from CSV file."""
        if self.csv_file is None or not os.path.exists(self.csv_file):
            return pd.DataFrame(columns=self.columns)
        return pd.read_csv(self.csv_file)

    def add_result(self, file_name, measurements):
        """
        Add a new analysis result to the CSV.

        Args:
            file_name (str): Name of the analyzed image file
            measurements (dict): Measurements dictionary from ParticleAnalyzer
                Must contain: num_particles, areas_px, diameters_px

        Returns:
            bool: True if successful
        """
        # Extract measurements
        num_particles = measurements.get('num_particles', 0)
        areas_px = measurements.get('areas_px', [])
        diams_px = measurements.get('diameters_px', [])

        # Get nm measurements if available
        if measurements.get('unit') == 'nm':
            areas_nm2 = measurements.get('areas', [])
            diams_nm = measurements.get('diameters', [])
        else:
            areas_nm2 = []
            diams_nm = []

        # Create new row
        new_row = {
            "file_name": file_name,
            "num_particles": num_particles,
            "particle_areas_px": str(areas_px),
            "equiv_diameters_px": str(diams_px),
            "particle_areas_nm2": str(areas_nm2),
            "equiv_diameters_nm": str(diams_nm)
        }

        # Append to CSV
        pd.DataFrame([new_row]).to_csv(
            self.csv_file,
            mode='a',
            index=False,
            header=False
        )

        # Reload data
        self.results_df = self._load_data()

        print(f"Stored results for '{file_name}' - {num_particles} particles detected")
        return True

    def get_results(self):
        """
        Get all results as a DataFrame.

        Returns:
            pd.DataFrame: Results dataframe
        """
        return self.results_df.copy()

    def delete_result(self, index):
        """
        Delete a result by index.

        Args:
            index (int): Row index to delete

        Returns:
            bool: True if successful

        Raises:
            IndexError: If index is out of range
        """
        if index not in self.results_df.index:
            raise IndexError(f"Index {index} not found in results")

        # Remove row and save
        self.results_df = self.results_df.drop(index=index)
        self.results_df.to_csv(self.csv_file, index=False)

        # Reload to reset indices
        self.results_df = self._load_data()

        print(f"Deleted result at index {index}")
        return True

    def find_duplicates(self):
        """
        Find duplicate entries based on file_name.

        Returns:
            list: List of tuples (index, file_name) for duplicate entries
        """
        duplicates = []
        seen = {}

        for idx, row in self.results_df.iterrows():
            file_name = row['file_name']
            if file_name in seen:
                # This is a duplicate - add to list
                duplicates.append((idx, file_name))
            else:
                seen[file_name] = idx

        return duplicates

    def delete_duplicates(self, keep='first'):
        """
        Delete duplicate entries based on file_name.

        Args:
            keep (str): Which duplicates to keep - 'first' or 'last'

        Returns:
            int: Number of duplicates deleted
        """
        initial_count = len(self.results_df)

        # Remove duplicates
        self.results_df = self.results_df.drop_duplicates(subset='file_name', keep=keep)
        self.results_df.to_csv(self.csv_file, index=False)

        # Reload to reset indices
        self.results_df = self._load_data()

        deleted_count = initial_count - len(self.results_df)
        print(f"Deleted {deleted_count} duplicate entries (kept {keep})")
        return deleted_count

    def export_results(self, output_file):
        """
        Export current results to a new CSV file.

        Args:
            output_file (str): Output file path

        Returns:
            bool: True if successful
        """
        if not output_file.endswith('.csv'):
            output_file += '.csv'

        self.results_df.to_csv(output_file, index=False)
        print(f"Exported {len(self.results_df)} results to '{output_file}'")
        return True

    def get_summary(self):
        """
        Get a summary of all stored results.

        Returns:
            dict: Summary statistics
        """
        if len(self.results_df) == 0:
            return {
                'total_images': 0,
                'total_particles': 0
            }

        total_particles = self.results_df['num_particles'].sum()

        return {
            'total_images': len(self.results_df),
            'total_particles': int(total_particles),
            'avg_particles_per_image': float(self.results_df['num_particles'].mean()),
            'min_particles': int(self.results_df['num_particles'].min()),
            'max_particles': int(self.results_df['num_particles'].max())
        }

    def print_summary(self):
        """Print a formatted summary of stored results."""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Total images analyzed: {summary['total_images']}")
        print(f"Total particles detected: {summary['total_particles']}")

        if summary['total_images'] > 0:
            print(f"Average particles per image: {summary['avg_particles_per_image']:.1f}")
            print(f"Min particles in an image: {summary['min_particles']}")
            print(f"Max particles in an image: {summary['max_particles']}")

        print("="*60 + "\n")

    def clear_all(self):
        """
        Clear all results and reset the CSV file.

        Returns:
            bool: True if successful
        """
        self._create_csv()
        self.results_df = self._load_data()
        print("Cleared all results")
        return True

    def __repr__(self):
        """String representation of the manager."""
        summary = self.get_summary()
        return (
            f"ResultsManager(file='{self.csv_file}', "
            f"images={summary['total_images']}, "
            f"particles={summary['total_particles']})"
        )
