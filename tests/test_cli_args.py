"""
Batch CLI argument contract.

The defaults here end up in every published number and in the run.json
provenance record, so a silent change to one is a silent change to results.
"""

from pathlib import Path

import pytest

from sem_particle_analysis import cli


@pytest.fixture
def parser():
    return cli.build_parser()


class TestDefaults:
    def test_defaults_match_the_documented_pipeline(self, parser):
        args = parser.parse_args(["images"])
        assert args.input == Path("images")
        assert args.output == Path("analysis_output")
        assert args.model_type == "vit_h"
        assert args.scale_method == "auto"
        assert args.scale_nm_per_px is None
        assert args.min_size == 30
        assert args.border_buffer == 4
        assert args.clear_edges is False
        assert args.no_plots is False

    def test_crop_percent_defaults_to_measuring_the_databar(self, parser):
        # None means "measure it"; a fixed percentage is fragile in both
        # directions, so it must not become the default by accident.
        assert parser.parse_args(["images"]).crop_percent is None

    def test_crop_percent_zero_keeps_the_full_frame(self, parser):
        assert parser.parse_args(["images", "--crop-percent", "0"]).crop_percent == 0.0


class TestOverrides:
    def test_output_directory(self, parser):
        args = parser.parse_args(["images", "-o", "out"])
        assert args.output == Path("out")

    def test_fixed_scale(self, parser):
        args = parser.parse_args(["images", "--scale-nm-per-px", "2.5"])
        assert args.scale_nm_per_px == pytest.approx(2.5)

    def test_clear_edges_is_a_flag(self, parser):
        assert parser.parse_args(["images", "--clear-edges"]).clear_edges is True

    def test_min_size(self, parser):
        assert parser.parse_args(["images", "--min-size", "120"]).min_size == 120

    @pytest.mark.parametrize("model", ["vit_b", "vit_h", "vit_l"])
    def test_accepts_every_sam_variant(self, parser, model):
        assert parser.parse_args(["images", "--model-type", model]).model_type == model

    @pytest.mark.parametrize("method", ["auto", "metadata", "ocr"])
    def test_accepts_every_scale_method(self, parser, method):
        assert parser.parse_args(["images", "--scale-method", method]).scale_method == method


class TestRejections:
    def test_requires_an_input(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_rejects_an_unknown_model(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["images", "--model-type", "vit_xl"])

    def test_rejects_an_unknown_scale_method(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["images", "--scale-method", "guess"])


class TestMainErrorPaths:
    def test_missing_input_exits_non_zero(self, tmp_path, capsys):
        assert cli.main([str(tmp_path / "nope"), "-o", str(tmp_path / "out")]) == 2
        assert "no such file or folder" in capsys.readouterr().err

    def test_empty_folder_exits_non_zero(self, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert cli.main([str(empty), "-o", str(tmp_path / "out")]) == 2
        assert "no images found" in capsys.readouterr().err


class TestProvenanceHelpers:
    def test_sha256_of_a_known_file(self, tmp_path):
        # Hashing the inputs is what makes a published result traceable.
        path = tmp_path / "f.bin"
        path.write_bytes(b"abc")
        expected = ("ba7816bf8f01cfea414140de5dae2223"
                    "b00361a396177a9cb410ff61f20015ad")
        assert cli.sha256_file(path) == expected

    def test_sha256_reads_in_chunks(self, tmp_path):
        # Checkpoints are multi-GB; the hash must not load one into memory.
        path = tmp_path / "big.bin"
        path.write_bytes(b"x" * 100_000)
        assert cli.sha256_file(path, chunk_size=1024) == cli.sha256_file(path)

    def test_versions_reports_the_running_python(self):
        versions = cli._versions()
        assert versions["python"]
        assert "platform" in versions
        for library in ("numpy", "pandas", "skimage"):
            assert library in versions


class TestResolveScale:
    class _Args:
        scale_nm_per_px = None
        scale_method = "auto"

    def test_a_fixed_scale_short_circuits_detection(self):
        args = self._Args()
        args.scale_nm_per_px = 2.5

        class Boom:
            def detect_scale(self, *a, **k):
                raise AssertionError("detection must not run")

        nm_per_px, info = cli.resolve_scale(Boom(), None, "x.tif", args)
        assert nm_per_px == pytest.approx(2.5)
        assert info["method"] == "manual"

    def test_detection_failure_falls_back_to_pixels_rather_than_raising(self):
        # A missing scale must not abort a batch; the image is still measurable
        # in pixels, and run.json records that scale failed.
        class Broken:
            def detect_scale(self, *a, **k):
                raise ValueError("no bar")

        nm_per_px, info = cli.resolve_scale(Broken(), None, "x.tif", self._Args())
        assert nm_per_px is None
        assert info["method"] == "failed"
        assert "no bar" in info["error"]

    def test_a_successful_detection_is_reported_with_provenance(self):
        class Works:
            def detect_scale(self, *a, **k):
                return {"conversion": 2.5, "method": "metadata", "scale_nm": 2.5,
                        "pixel_length": 1, "ocr_text": None, "warning": None}

        nm_per_px, info = cli.resolve_scale(Works(), None, "x.tif", self._Args())
        assert nm_per_px == pytest.approx(2.5)
        assert info["method"] == "metadata"
