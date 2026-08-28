"""
Results Data Management

Handles saving and loading of particle analysis results to CSV files.
"""

import ast
import math
import os
import re
import pandas as pd
import random

# Matches the numpy 2 scalar repr, e.g. "np.float64(896.0)". str() of a list of
# numpy scalars produces these, and neither ast.literal_eval nor a plain eval in
# a namespace without numpy can read them back.
_NUMPY_SCALAR = re.compile(r"np\.\w+\(([^()]*)\)")


def serialise_measurements(values):
    """
    Render a list of measurements for storage in a CSV cell.

    Values coming out of scikit-image are numpy scalars, and since numpy 2 their
    repr is ``np.float64(896.0)`` rather than ``896.0``. Writing ``str(values)``
    therefore produced cells that no reader could parse: ``ast.literal_eval``
    rejects the call syntax, and ``eval`` raises NameError wherever numpy is not
    in scope. Coercing to plain floats keeps the cell readable by anything.

    Args:
        values: Iterable of numbers.

    Returns:
        str: A plain Python list literal, e.g. "[896.0, 1024.0]".
    """
    return str([float(v) for v in (values or [])])


def parse_measurement_list(cell):
    """
    Read a measurement list back out of a CSV cell.

    Tolerates every form the column has held: a plain list literal, an empty
    list, a blank or missing cell, and the ``np.float64(...)`` reprs written by
    older runs under numpy 2. A cell that cannot be parsed at all raises, rather
    than being silently treated as "no particles" — an empty distribution that
    should have held data is worse than an error, because it looks like a result.

    Args:
        cell: The raw CSV cell (str, NaN, or None).

    Returns:
        list[float]: The measurements, empty only if the cell genuinely is.

    Raises:
        ValueError: If the cell holds something unparseable.
    """
    if cell is None:
        return []
    if isinstance(cell, float) and math.isnan(cell):
        return []
    if isinstance(cell, (list, tuple)):
        return [float(v) for v in cell]

    text = str(cell).strip()
    if text in ("", "[]", "nan", "None"):
        return []

    # Unwrap numpy scalar reprs left behind by earlier runs.
    text = _NUMPY_SCALAR.sub(r"\1", text)

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Could not parse measurement list from {cell!r}") from exc

    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expected a list of measurements, got {parsed!r}")
    return [float(v) for v in parsed]


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

        # Define column structure. nm_per_px is recorded so a row measured in
        # pixels — because no scale could be read — still says what it would have
        # been calibrated with, and so any nm value can be traced back.
        self.columns = [
            "file_name",
            "num_particles",
            "nm_per_px",
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

        # Create new row. The measurement lists go through serialise_measurements
        # rather than str(): they hold numpy scalars, whose repr since numpy 2 is
        # "np.float64(896.0)" — a form no reader can parse back.
        new_row = {
            "file_name": file_name,
            "num_particles": num_particles,
            "nm_per_px": measurements.get('nm_per_px'),
            "particle_areas_px": serialise_measurements(areas_px),
            "equiv_diameters_px": serialise_measurements(diams_px),
            "particle_areas_nm2": serialise_measurements(areas_nm2),
            "equiv_diameters_nm": serialise_measurements(diams_nm)
        }

        # Append in the order this file's header actually uses. Writing a fixed
        # field order with header=False silently shifts values into the wrong
        # columns whenever the target CSV has a different or older schema —
        # which is exactly what resuming someone else's results file does.
        if not os.path.exists(self.csv_file):
            self._create_csv()
        columns = list(self.results_df.columns) or self.columns

        pd.DataFrame([new_row]).reindex(columns=columns).to_csv(
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
