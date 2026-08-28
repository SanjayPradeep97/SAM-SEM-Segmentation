"""
Results CSV storage.

The failure this guards against is silent: appending a fixed field order with
header=False into a CSV whose columns are in a different order writes every value
into the wrong column, and nothing complains. Resuming someone else's results
file is exactly that situation.
"""

import ast

import pandas as pd
import pytest

from sem_particle_analysis import ResultsManager
from sem_particle_analysis import data_manager as dm


def measurements(num_particles=3, nm_per_px=2.5):
    """A measurements dict shaped like ParticleAnalyzer.get_measurements()."""
    areas_px = [100.0 * (i + 1) for i in range(num_particles)]
    diams_px = [10.0 * (i + 1) for i in range(num_particles)]
    return {
        "num_particles": num_particles,
        "areas_px": areas_px,
        "diameters_px": diams_px,
        "areas": [a * nm_per_px**2 for a in areas_px],
        "diameters": [d * nm_per_px for d in diams_px],
        "unit": "nm",
        "nm_per_px": nm_per_px,
    }


@pytest.fixture
def manager(tmp_path):
    return ResultsManager(csv_file=str(tmp_path / "results.csv"))


class TestAddResult:
    def test_writes_one_row_per_image(self, manager):
        manager.add_result("a.tif", measurements(3))
        manager.add_result("b.tif", measurements(5))
        results = manager.get_results()
        assert len(results) == 2
        assert list(results["file_name"]) == ["a.tif", "b.tif"]
        assert list(results["num_particles"]) == [3, 5]

    def test_records_the_scale_that_produced_the_nm_values(self, manager):
        manager.add_result("a.tif", measurements(3, nm_per_px=2.5))
        assert manager.get_results()["nm_per_px"].iloc[0] == pytest.approx(2.5)

    def test_stores_both_pixel_and_nanometre_measurements(self, manager):
        data = measurements(3)
        manager.add_result("a.tif", data)
        row = manager.get_results().iloc[0]
        assert ast.literal_eval(row["particle_areas_px"]) == data["areas_px"]
        assert ast.literal_eval(row["equiv_diameters_nm"]) == pytest.approx(data["diameters"])

    def test_leaves_nanometre_columns_empty_when_there_is_no_scale(self, manager):
        data = measurements(2)
        data["unit"] = "pixels"
        manager.add_result("a.tif", data)
        row = manager.get_results().iloc[0]
        assert ast.literal_eval(row["particle_areas_nm2"]) == []
        assert ast.literal_eval(row["equiv_diameters_px"]) == data["diameters_px"]

    def test_handles_an_image_with_no_particles(self, manager):
        manager.add_result("empty.tif", {"num_particles": 0, "areas_px": [],
                                         "diameters_px": [], "unit": "pixels"})
        assert manager.get_results()["num_particles"].iloc[0] == 0


class TestSchemaOrderIsRespected:
    def test_appending_to_a_reordered_csv_keeps_values_in_their_columns(self, tmp_path):
        # Someone else's results file with the columns in a different order.
        path = tmp_path / "foreign.csv"
        reordered = ["num_particles", "file_name", "equiv_diameters_nm",
                     "particle_areas_nm2", "equiv_diameters_px",
                     "particle_areas_px", "nm_per_px"]
        pd.DataFrame(columns=reordered).to_csv(path, index=False)

        manager = ResultsManager(csv_file=str(path))
        manager.add_result("a.tif", measurements(7, nm_per_px=1.25))

        row = pd.read_csv(path).iloc[0]
        assert row["file_name"] == "a.tif"
        assert int(row["num_particles"]) == 7
        assert float(row["nm_per_px"]) == pytest.approx(1.25)

    def test_reading_back_a_written_file_round_trips(self, tmp_path):
        path = tmp_path / "results.csv"
        first = ResultsManager(csv_file=str(path))
        first.add_result("a.tif", measurements(4))

        second = ResultsManager(csv_file=str(path))
        assert len(second.get_results()) == 1
        assert second.get_results()["num_particles"].iloc[0] == 4


class TestDuplicates:
    def test_finds_repeated_filenames(self, manager):
        manager.add_result("a.tif", measurements(3))
        manager.add_result("b.tif", measurements(4))
        manager.add_result("a.tif", measurements(9))
        duplicates = manager.find_duplicates()
        assert [name for _, name in duplicates] == ["a.tif"]

    def test_no_duplicates_in_a_clean_file(self, manager):
        manager.add_result("a.tif", measurements(3))
        manager.add_result("b.tif", measurements(4))
        assert manager.find_duplicates() == []

    def test_delete_duplicates_keeps_the_first_by_default(self, manager):
        manager.add_result("a.tif", measurements(3))
        manager.add_result("a.tif", measurements(9))
        assert manager.delete_duplicates() == 1
        assert manager.get_results()["num_particles"].iloc[0] == 3

    def test_delete_duplicates_can_keep_the_last(self, manager):
        manager.add_result("a.tif", measurements(3))
        manager.add_result("a.tif", measurements(9))
        manager.delete_duplicates(keep="last")
        assert manager.get_results()["num_particles"].iloc[0] == 9


class TestRowOperations:
    def test_delete_removes_a_row(self, manager):
        manager.add_result("a.tif", measurements(3))
        manager.add_result("b.tif", measurements(4))
        manager.delete_result(0)
        assert list(manager.get_results()["file_name"]) == ["b.tif"]

    def test_deleting_a_missing_index_raises(self, manager):
        manager.add_result("a.tif", measurements(3))
        with pytest.raises(IndexError):
            manager.delete_result(99)

    def test_clear_all_empties_the_file(self, manager):
        manager.add_result("a.tif", measurements(3))
        manager.clear_all()
        assert len(manager.get_results()) == 0

    def test_export_writes_a_second_file(self, manager, tmp_path):
        manager.add_result("a.tif", measurements(3))
        destination = tmp_path / "export"
        manager.export_results(str(destination))
        assert (tmp_path / "export.csv").exists()
        assert len(pd.read_csv(tmp_path / "export.csv")) == 1


class TestSummary:
    def test_summary_of_an_empty_file(self, manager):
        assert manager.get_summary() == {"total_images": 0, "total_particles": 0}

    def test_summary_aggregates_across_images(self, manager):
        manager.add_result("a.tif", measurements(3))
        manager.add_result("b.tif", measurements(7))
        summary = manager.get_summary()
        assert summary["total_images"] == 2
        assert summary["total_particles"] == 10
        assert summary["avg_particles_per_image"] == pytest.approx(5.0)
        assert summary["min_particles"] == 3
        assert summary["max_particles"] == 7


class TestSerialisation:
    """
    CSV cells must survive a round trip.

    numpy 2 changed the repr of a scalar from "896.0" to "np.float64(896.0)".
    Since skimage hands back numpy scalars, str(list) started writing cells that
    nothing could read: ast.literal_eval rejects the call syntax and a plain eval
    raises NameError wherever numpy is not imported. The downstream notebook
    swallowed that in a bare except and reported zero particles.
    """

    def test_writes_plain_floats_not_numpy_reprs(self):
        import numpy as np

        cell = dm.serialise_measurements([np.float64(896.0), np.float64(1024.5)])
        assert cell == "[896.0, 1024.5]"
        assert "np.float64" not in cell

    def test_round_trips_through_the_parser(self):
        import numpy as np

        values = [np.float64(1.5), np.float64(2.25)]
        assert dm.parse_measurement_list(dm.serialise_measurements(values)) == [1.5, 2.25]

    def test_a_real_analyzer_result_round_trips(self, tmp_path):
        # End to end: measure, save, read back. This is the path the Plotting tab
        # and analyze_results.ipynb both take.
        import numpy as np
        from sem_particle_analysis import ParticleAnalyzer

        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:80, 50:80] = 1
        mask[120:150, 120:150] = 1
        analyzer = ParticleAnalyzer(conversion_factor=2.5, min_size=30)
        analyzer.analyze_mask(mask, remove_border=False)

        manager = ResultsManager(csv_file=str(tmp_path / "r.csv"))
        manager.add_result("a.tif", analyzer.get_measurements(in_nm=True))

        row = manager.get_results().iloc[0]
        diameters = dm.parse_measurement_list(row["equiv_diameters_nm"])
        assert len(diameters) == 2
        assert all(d > 0 for d in diameters)
        # The cell must also be readable by a plain literal_eval, which is what
        # any external consumer will reach for.
        assert ast.literal_eval(row["equiv_diameters_nm"]) == pytest.approx(diameters)


class TestParseMeasurementList:
    @pytest.mark.parametrize("cell", ["[]", "", None, float("nan"), "nan", "None"])
    def test_empty_forms_give_an_empty_list(self, cell):
        assert dm.parse_measurement_list(cell) == []

    def test_reads_a_plain_list(self):
        assert dm.parse_measurement_list("[1.0, 2.5, 3]") == [1.0, 2.5, 3.0]

    def test_reads_a_legacy_numpy_repr_cell(self):
        # CSVs already written under numpy 2 must stay readable.
        cell = "[np.float64(896.0), np.float64(1024.5)]"
        assert dm.parse_measurement_list(cell) == [896.0, 1024.5]

    def test_reads_a_mixed_legacy_cell(self):
        assert dm.parse_measurement_list("[np.float32(1.5), 2.0]") == [1.5, 2.0]

    def test_accepts_a_list_that_was_never_stringified(self):
        assert dm.parse_measurement_list([1, 2]) == [1.0, 2.0]

    def test_unparseable_content_raises_rather_than_reporting_no_particles(self):
        # Silently returning [] is what made the notebook report zero particles
        # for a whole dataset without any error.
        with pytest.raises(ValueError):
            dm.parse_measurement_list("[1.0, 2.0")
        with pytest.raises(ValueError):
            dm.parse_measurement_list("not a list at all")

    def test_a_non_list_literal_raises(self):
        with pytest.raises(ValueError):
            dm.parse_measurement_list("42")
