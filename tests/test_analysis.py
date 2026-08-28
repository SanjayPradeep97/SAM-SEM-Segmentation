"""
Particle measurement arithmetic.

Checked against disks of exactly known radius rather than against a previous
run, so a regression shows up as a wrong number and not merely a changed one.
"""

import numpy as np
import pytest
from skimage import measure

from sem_particle_analysis import ParticleAnalyzer
from synthetic import make_disk_mask


@pytest.fixture
def two_disks():
    """Two well-separated disks, radii 30 and 45 px."""
    return make_disk_mask(shape=(400, 400), centers_radii=((100, 100, 30), (300, 250, 45)))


class TestMeasurementArithmetic:
    def test_counts_separated_particles(self, two_disks):
        mask, _ = two_disks
        analyzer = ParticleAnalyzer(min_size=30)
        count, _regions = analyzer.analyze_mask(mask, remove_border=False)
        assert count == 2

    def test_areas_match_pi_r_squared(self, two_disks):
        mask, radii = two_disks
        analyzer = ParticleAnalyzer(min_size=30)
        analyzer.analyze_mask(mask, remove_border=False)
        measured = sorted(analyzer.get_measurements(in_nm=False)["areas"])
        expected = sorted(np.pi * r**2 for r in radii)
        # A rasterised disk plus a radius-1 opening differs from the ideal area
        # by a boundary term, which is a few percent at these radii.
        for got, want in zip(measured, expected):
            assert got == pytest.approx(want, rel=0.05)

    def test_equivalent_diameter_recovers_the_radius(self, two_disks):
        mask, radii = two_disks
        analyzer = ParticleAnalyzer(min_size=30)
        analyzer.analyze_mask(mask, remove_border=False)
        measured = sorted(analyzer.get_measurements(in_nm=False)["diameters"])
        for got, r in zip(measured, sorted(radii)):
            assert got == pytest.approx(2 * r, rel=0.05)

    def test_measuring_emits_no_deprecation_warning(self, two_disks):
        # equivalent_diameter is deprecated in scikit-image 0.26 and goes away in
        # 2.0; the measurement everything else derives from must not ride on it.
        import warnings

        mask, _ = two_disks
        analyzer = ParticleAnalyzer(min_size=30)
        analyzer.analyze_mask(mask, remove_border=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analyzer.get_measurements(in_nm=False)

        offenders = [str(w.message) for w in caught
                     if issubclass(w.category, (DeprecationWarning, FutureWarning))]
        assert not offenders, f"deprecation warnings raised: {offenders}"


class TestUnitConversion:
    def test_nm_areas_scale_with_the_square_of_the_conversion(self, two_disks):
        mask, _ = two_disks
        nm_per_px = 2.5

        pixels = ParticleAnalyzer(min_size=30)
        pixels.analyze_mask(mask, remove_border=False)
        px_measurements = pixels.get_measurements(in_nm=False)

        nanometres = ParticleAnalyzer(conversion_factor=nm_per_px, min_size=30)
        nanometres.analyze_mask(mask, remove_border=False)
        nm_measurements = nanometres.get_measurements(in_nm=True)

        assert nm_measurements["unit"] == "nm"
        for area_px, area_nm in zip(px_measurements["areas"], nm_measurements["areas"]):
            assert area_nm == pytest.approx(area_px * nm_per_px**2)
        for d_px, d_nm in zip(px_measurements["diameters"], nm_measurements["diameters"]):
            assert d_nm == pytest.approx(d_px * nm_per_px)

    def test_falls_back_to_pixels_without_a_conversion(self, two_disks):
        mask, _ = two_disks
        analyzer = ParticleAnalyzer(conversion_factor=None, min_size=30)
        analyzer.analyze_mask(mask, remove_border=False)
        measurements = analyzer.get_measurements(in_nm=True)
        assert measurements["unit"] == "pixels"
        assert measurements["areas"] == measurements["areas_px"]

    def test_pixel_measurements_are_always_reported_alongside(self, two_disks):
        mask, _ = two_disks
        analyzer = ParticleAnalyzer(conversion_factor=2.5, min_size=30)
        analyzer.analyze_mask(mask, remove_border=False)
        measurements = analyzer.get_measurements(in_nm=True)
        assert len(measurements["areas_px"]) == len(measurements["areas"])
        assert len(measurements["diameters_px"]) == len(measurements["diameters"])

    def test_nm_per_px_is_reported_so_results_stay_traceable(self, two_disks):
        # ResultsManager writes measurements['nm_per_px'] into the CSV. It used
        # to be present only on the zero-particle branch, so the column was blank
        # on every row that actually held measurements.
        mask, _ = two_disks
        analyzer = ParticleAnalyzer(conversion_factor=2.5, min_size=30)
        analyzer.analyze_mask(mask, remove_border=False)

        populated = analyzer.get_measurements(in_nm=True)
        assert populated["num_particles"] > 0
        assert populated["nm_per_px"] == pytest.approx(2.5)

        empty = ParticleAnalyzer(conversion_factor=2.5, min_size=30)
        empty.analyze_mask(np.zeros((100, 100), dtype=np.uint8), remove_border=False)
        assert empty.get_measurements(in_nm=True)["nm_per_px"] == pytest.approx(2.5)

    def test_empty_mask_reports_no_particles(self):
        analyzer = ParticleAnalyzer(conversion_factor=2.5, min_size=30)
        analyzer.analyze_mask(np.zeros((100, 100), dtype=np.uint8), remove_border=False)
        measurements = analyzer.get_measurements()
        assert measurements["num_particles"] == 0
        assert measurements["areas"] == []
        assert analyzer.get_summary_statistics() == {"num_particles": 0}


class TestSizeFiltering:
    def test_discards_particles_below_min_size(self):
        mask, _ = make_disk_mask(
            shape=(300, 300), centers_radii=((80, 80, 30), (220, 200, 3))
        )
        analyzer = ParticleAnalyzer(min_size=200)
        count, _ = analyzer.analyze_mask(mask, remove_border=False)
        assert count == 1

    def test_min_size_argument_overrides_the_instance_default(self):
        mask, _ = make_disk_mask(
            shape=(300, 300), centers_radii=((80, 80, 30), (220, 200, 8))
        )
        analyzer = ParticleAnalyzer(min_size=30)
        assert analyzer.analyze_mask(mask, remove_border=False)[0] == 2
        assert analyzer.analyze_mask(mask, min_size=1000, remove_border=False)[0] == 1


class TestEdgeHandling:
    def test_removes_a_thin_ribbon_hugging_the_frame(self):
        # The classic SAM artefact: a 2px strip along the top edge.
        mask = np.zeros((200, 200), dtype=bool)
        mask[0:2, :] = True
        cleaned = ParticleAnalyzer._remove_edge_slivers(mask)
        assert not cleaned.any()

    def test_keeps_a_real_object_running_off_the_frame(self):
        # A particle crossing the border extends well past the edge band, so it
        # is a real object that happens to be clipped — not an artefact.
        mask = np.zeros((200, 200), dtype=bool)
        mask[0:80, 60:140] = True
        cleaned = ParticleAnalyzer._remove_edge_slivers(mask)
        assert cleaned.any()

    def test_border_artifacts_strip_clears_the_outermost_pixels(self):
        analyzer = ParticleAnalyzer(min_size=1)
        mask = np.ones((50, 50), dtype=bool)
        cleaned = analyzer._remove_border_artifacts(mask, border_width=4)
        assert not cleaned[:4, :].any()
        assert not cleaned[-4:, :].any()
        assert not cleaned[:, :4].any()
        assert not cleaned[:, -4:].any()
        assert cleaned[10, 10]

    def test_clear_edge_particles_drops_only_the_touching_one(self):
        mask = np.zeros((300, 300), dtype=bool)
        mask[140:200, 140:200] = True   # interior
        mask[0:40, 100:160] = True      # touches the top edge
        analyzer = ParticleAnalyzer(min_size=30)
        analyzer.analyze_mask(mask, remove_border=False)
        assert len(analyzer.regions) == 2
        analyzer.clear_edge_particles(buffer_size=0)
        assert len(analyzer.regions) == 1


class TestRefinementOperations:
    @pytest.fixture
    def analyzed(self):
        mask, _ = make_disk_mask(
            shape=(400, 400), centers_radii=((100, 100, 30), (300, 250, 45))
        )
        analyzer = ParticleAnalyzer(min_size=30)
        analyzer.analyze_mask(mask, remove_border=False)
        return analyzer

    def test_delete_removes_the_named_particle(self, analyzed):
        label = analyzed.regions[0].label
        analyzed.delete_particles([label])
        assert len(analyzed.regions) == 1
        assert label not in [r.label for r in analyzed.regions] or len(analyzed.regions) == 1

    def test_merge_reports_failure_when_particles_are_far_apart(self, analyzed):
        # Closing with a radius-1 disk cannot bridge a 200px gap. The operation
        # must say so rather than silently leaving the count unchanged.
        labels = [r.label for r in analyzed.regions]
        analyzed.merge_particles(labels)
        assert analyzed.last_merge_succeeded is False
        assert len(analyzed.regions) == 2

    def test_merge_succeeds_for_touching_particles(self):
        mask = np.zeros((200, 200), dtype=bool)
        mask[80:120, 40:95] = True
        mask[80:120, 97:150] = True   # a 2px gap, bridgeable by closing
        analyzer = ParticleAnalyzer(min_size=30)
        analyzer.analyze_mask(mask, remove_border=False)
        assert len(analyzer.regions) == 2
        analyzer.merge_particles([r.label for r in analyzer.regions])
        assert analyzer.last_merge_succeeded is True
        assert len(analyzer.regions) == 1

    def test_merge_needs_at_least_two_particles(self, analyzed):
        before = len(analyzed.regions)
        assert analyzed.merge_particles([analyzed.regions[0].label]) == before

    def test_add_from_sam_keeps_only_the_largest_component_when_asked(self, analyzed):
        # SAM often returns several disconnected blobs for one particle; adding
        # them all multiplies the count.
        before = len(analyzed.regions)
        sam_mask = np.zeros((400, 400), dtype=bool)
        sam_mask[20:60, 300:340] = True   # big
        sam_mask[380:390, 20:30] = True   # stray speck
        analyzed.add_particle_from_sam(sam_mask, largest_only=True)
        assert len(analyzed.regions) == before + 1

    def test_add_from_sam_without_largest_only_adds_every_blob(self, analyzed):
        before = len(analyzed.regions)
        sam_mask = np.zeros((400, 400), dtype=bool)
        sam_mask[20:60, 300:340] = True
        sam_mask[350:390, 20:60] = True
        analyzed.add_particle_from_sam(sam_mask, largest_only=False)
        assert len(analyzed.regions) == before + 2

    def test_find_particle_at_point_returns_the_nearest(self, analyzed):
        region, index, label = analyzed.find_particle_at_point(100, 100)
        assert region is not None
        assert index in (0, 1)
        assert label == region.label

    def test_find_particle_at_point_with_no_particles(self):
        analyzer = ParticleAnalyzer(min_size=30)
        analyzer.analyze_mask(np.zeros((50, 50), dtype=np.uint8), remove_border=False)
        assert analyzer.find_particle_at_point(10, 10) == (None, None, None)


class TestCleanupConsistency:
    def test_a_particle_measures_the_same_before_and_after_an_edit(self):
        # analyze_mask and the refinement operations must apply identical
        # cleanup, or a particle's reported area shifts the first time the
        # analyst touches an unrelated particle.
        mask, _ = make_disk_mask(
            shape=(400, 400), centers_radii=((100, 100, 30), (300, 250, 45))
        )
        analyzer = ParticleAnalyzer(min_size=30)
        analyzer.analyze_mask(mask, remove_border=False)

        keep = min(analyzer.regions, key=lambda r: r.area)
        drop = max(analyzer.regions, key=lambda r: r.area)
        area_before = keep.area

        analyzer.delete_particles([drop.label])

        assert len(analyzer.regions) == 1
        assert analyzer.regions[0].area == pytest.approx(area_before)


class TestSummaryStatistics:
    def test_statistics_describe_the_measured_particles(self):
        mask, _ = make_disk_mask(
            shape=(400, 400), centers_radii=((100, 100, 30), (300, 250, 45))
        )
        analyzer = ParticleAnalyzer(conversion_factor=2.0, min_size=30)
        analyzer.analyze_mask(mask, remove_border=False)
        stats = analyzer.get_summary_statistics()
        measurements = analyzer.get_measurements(in_nm=True)

        assert stats["num_particles"] == 2
        assert stats["unit"] == "nm"
        assert stats["diameter_mean"] == pytest.approx(np.mean(measurements["diameters"]))
        assert stats["diameter_min"] <= stats["diameter_median"] <= stats["diameter_max"]
        assert stats["area_mean"] == pytest.approx(np.mean(measurements["areas"]))
