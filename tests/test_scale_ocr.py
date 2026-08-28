"""
Reading a printed scale bar — tier 2.

Checked against synthetic micrographs whose bar length in pixels and printed
label are known exactly, so a misread shows up as a wrong nm/px rather than as a
plausible-looking number.

Marked slow: EasyOCR downloads and loads detection models on first use.
"""

import numpy as np
import pytest

from synthetic import databar_region, make_micrograph

pytestmark = pytest.mark.slow


class TestReadsASyntheticBar:
    def test_recovers_the_known_nm_per_pixel(self, scale_detector):
        # 500 nm printed over a bar drawn exactly 200 px long -> 2.5 nm/px.
        image, truth = make_micrograph(bar_length_px=200, scale_text="500 nm",
                                       scale_nm=500.0)
        result = scale_detector.detect_scale_bar(image, **databar_region(truth))
        assert result["conversion"] == pytest.approx(truth["nm_per_px"], rel=0.05)
        assert result["scale_nm"] == pytest.approx(500.0)
        assert result["pixel_length"] == pytest.approx(200, abs=10)

    def test_reads_a_micron_label(self, scale_detector):
        image, truth = make_micrograph(bar_length_px=250, scale_text="1 um",
                                       scale_nm=1000.0)
        result = scale_detector.detect_scale_bar(image, **databar_region(truth))
        assert result["conversion"] == pytest.approx(truth["nm_per_px"], rel=0.1)

    def test_reads_a_bar_on_a_light_databar(self, scale_detector):
        # Not every instrument draws white-on-black.
        image, truth = make_micrograph(databar_gray=235, bar_gray=0,
                                       bar_length_px=200, scale_text="500 nm",
                                       scale_nm=500.0)
        result = scale_detector.detect_scale_bar(image, **databar_region(truth))
        assert result["conversion"] == pytest.approx(truth["nm_per_px"], rel=0.1)


class TestGuardRails:
    def test_a_non_standard_value_is_flagged_for_confirmation(self, scale_detector):
        # Instruments label bars with a 1/2/5 mantissa. "700 nm" is off that grid
        # and is nearly always a misread digit — usable, but it must warn.
        image, truth = make_micrograph(bar_length_px=200, scale_text="700 nm",
                                       scale_nm=700.0)
        result = scale_detector.detect_scale_bar(image, **databar_region(truth))
        assert result.get("warning")

    def test_a_standard_value_is_not_flagged(self, scale_detector):
        image, truth = make_micrograph(bar_length_px=200, scale_text="500 nm",
                                       scale_nm=500.0)
        result = scale_detector.detect_scale_bar(image, **databar_region(truth))
        assert not result.get("warning")

    def test_a_region_with_no_bar_raises(self, scale_detector):
        image, truth = make_micrograph()
        with pytest.raises(ValueError):
            # The middle of the micrograph, nowhere near the databar.
            scale_detector.detect_scale_bar(
                image, region_x=0.5, region_y=0.3,
                region_width=0.2, region_height=0.1)

    def test_a_blank_image_raises(self, scale_detector):
        blank = np.full((400, 600, 3), 128, dtype=np.uint8)
        with pytest.raises(ValueError):
            scale_detector.detect_scale_bar(blank)


class TestStandardValueClassification:
    @pytest.mark.parametrize("nanometres", [100.0, 200.0, 500.0, 1000.0, 2500.0])
    def test_grid_values_are_standard(self, scale_detector, nanometres):
        assert scale_detector._is_standard_scale_value(nanometres)

    @pytest.mark.parametrize("nanometres", [700.0, 1100.0, 333.0])
    def test_off_grid_values_are_not_standard(self, scale_detector, nanometres):
        assert not scale_detector._is_standard_scale_value(nanometres)


