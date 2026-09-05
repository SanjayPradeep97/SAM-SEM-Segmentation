"""
The review app: opening a batch's masks and correcting them.

The app never segments. Its whole job is to put a mask somebody else made back
into a ParticleAnalyzer with the right scale, hand it to the refinement tools,
and write out what the analyst settles on. So the things worth pinning down are
the round trip — mask in, same numbers out — and that every control the
interface wires actually works when there is a frame open.

Needs gradio, which the callbacks import. Never touches SAM: the two tools that
need a model are checked for the message they give without one.
"""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("gradio")

from sem_particle_analysis import ResultsManager
from sem_particle_analysis import scale_calibration as sc
from synthetic import make_disk_mask


NM_PER_PX = 2.5


@pytest.fixture
def analysis(tmp_path):
    """A folder shaped like one a batch run wrote: three frames, three masks."""
    root = tmp_path / "X1_analysis"
    for sub in ("raw", "mask", "overlay"):
        (root / sub).mkdir(parents=True)

    frames = {}
    for index, centres in enumerate((
            ((60, 60, 20), (150, 150, 22), (240, 240, 18)),
            ((80, 80, 25),),
            ((120, 120, 30), (200, 60, 20)))):
        stem = f"X1_{index + 1:04d}"
        mask, _ = make_disk_mask(shape=(300, 300), centers_radii=centres)
        grey = np.where(mask, 40, 200).astype(np.uint8)
        Image.fromarray(grey).save(root / "raw" / f"{stem}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            root / "mask" / f"{stem}.png")
        Image.fromarray(np.stack([grey] * 3, -1)).save(
            root / "overlay" / f"{stem}.png")
        frames[stem] = mask

    results = ResultsManager(csv_file=str(root / "analysis_results.csv"))
    from sem_particle_analysis import ParticleAnalyzer

    for stem, mask in frames.items():
        analyzer = ParticleAnalyzer(conversion_factor=NM_PER_PX, min_size=30)
        analyzer.analyze_mask(mask, min_size=30, remove_border=False)
        results.add_result(f"{stem}.png", analyzer.get_measurements(in_nm=True),
                           scale_method="box_ocr+unconfirmed")
    return root, frames


@pytest.fixture
def opened(analysis):
    """The app with that folder open and its first frame on screen."""
    from sem_analysis_app.state import state
    from sem_review_app import loading

    root, frames = analysis
    status, gallery = loading.load_folder(str(root), min_size=30)
    assert status.startswith("✅"), status
    loading.open_index(0)
    yield state, loading, root, frames
    state.analyzer = None
    state.results_manager = None
    state.image_paths = []
    state.segmenter = None


class TestOpeningAFolder:
    def test_it_finds_every_frame_with_both_halves(self, analysis):
        from sem_review_app import loading

        root, _frames = analysis
        status, gallery = loading.load_folder(str(root), min_size=30)
        assert "3 frames" in status
        assert len(gallery) == 3

    def test_a_frame_with_no_mask_is_not_offered(self, analysis):
        # Half a pair is not a reviewable frame, and silently listing it would
        # fail only later, when it was clicked.
        from sem_review_app import loading

        root, _frames = analysis
        (root / "mask" / "X1_0002.png").unlink()
        _status, gallery = loading.load_folder(str(root), min_size=30)
        assert len(gallery) == 2

    def test_a_folder_that_is_not_an_analysis_says_which_part_is_missing(self, tmp_path):
        from sem_review_app import loading

        (tmp_path / "raw").mkdir()
        status, gallery = loading.load_folder(str(tmp_path), min_size=30)
        assert status.startswith("❌") and "mask" in status
        assert gallery is None

    def test_a_missing_folder_is_reported_not_raised(self):
        from sem_review_app import loading

        status, _gallery = loading.load_folder("nowhere at all", min_size=30)
        assert status.startswith("❌")


class TestTheMaskComesBackAsItWas:
    def test_the_particle_count_matches_the_analysis(self, opened):
        state, loading, root, frames = opened
        table = ResultsManager(csv_file=str(root / "analysis_results.csv"),
                               auto_create=False).get_results()
        expected = int(table.loc[table.file_name == "X1_0001.png",
                                 "num_particles"].iloc[0])
        assert len(state.analyzer.regions) == expected == 3

    def test_the_measurements_match_the_analysis(self, opened):
        from sem_particle_analysis.data_manager import parse_measurement_list

        state, loading, root, _frames = opened
        table = ResultsManager(csv_file=str(root / "analysis_results.csv"),
                               auto_create=False).get_results()
        saved = parse_measurement_list(
            table.loc[table.file_name == "X1_0001.png",
                      "equiv_diameters_nm"].iloc[0])
        now = state.analyzer.get_measurements(in_nm=True)["diameters"]
        assert sorted(now) == pytest.approx(sorted(saved), rel=1e-6)

    def test_the_scale_comes_back_with_it(self, opened):
        state, _loading, _root, _frames = opened
        assert state.analyzer.conversion == pytest.approx(NM_PER_PX)
        assert state.scale_calibration.nm_per_px == pytest.approx(NM_PER_PX)

    def test_an_unconfirmed_scale_stays_unconfirmed(self, opened):
        # Reviewing a mask says nothing about whether anyone checked the scale
        # bar, and saving must not claim otherwise.
        state, _loading, _root, _frames = opened
        assert state.scale_calibration.provenance == "box_ocr+unconfirmed"

    def test_a_confirmed_scale_stays_confirmed(self):
        from sem_review_app.loading import scale_from

        assert scale_from("two_points", 2.5).provenance == "two_points"
        assert scale_from("metadata", 2.5).provenance == "metadata"

    def test_no_scale_at_all_is_handled(self):
        from sem_review_app.loading import scale_from

        assert scale_from(None, None) is None
        assert scale_from("box_ocr", float("nan")) is None


class TestMovingThroughTheFolder:
    def test_next_and_previous_walk_the_list(self, opened):
        state, loading, _root, _frames = opened
        loading.skip_to_next()
        assert state.current_index == 1
        loading.go_back()
        assert state.current_index == 0

    def test_it_stops_at_the_ends(self, opened):
        state, loading, _root, _frames = opened
        _viz, _header, info, _table, _scale = loading.go_back()
        assert "first frame" in info
        assert state.current_index == 0

        state.current_index = 2
        loading.open_index(2)
        _viz, _header, info, _table, _scale = loading.skip_to_next()
        assert "No more frames" in info

    def test_saving_records_a_row_and_ticks_the_frame(self, opened):
        state, loading, root, _frames = opened
        status, gallery = loading.save_here()
        assert status.startswith("✅")

        reviewed = ResultsManager(csv_file=str(root / loading.REVIEWED_CSV),
                                  auto_create=False).get_results()
        # Under the name the analysis used, so the two files line up on it.
        assert list(reviewed["file_name"]) == ["X1_0001.png"]
        assert gallery[0][1].startswith("✅")

    def test_save_and_next_moves_on(self, opened):
        state, loading, _root, _frames = opened
        status, _gallery, _viz, _header, info, _table, _scale = loading.save_and_next()
        assert status.startswith("✅")
        assert state.current_index == 1

    def test_reload_throws_away_the_edits(self, opened):
        state, loading, _root, _frames = opened
        before = len(state.analyzer.regions)
        state.analyzer.delete_particles([state.analyzer.regions[0].label])
        assert len(state.analyzer.regions) == before - 1

        _viz, _header, info, _table, _scale = loading.reload_frame()
        assert len(state.analyzer.regions) == before
        assert "Reloaded" in info


class TestEveryControl:
    """
    One test per control the interface wires, with a frame open.

    A Gradio callback that raises shows the user a red box and nothing else, and
    a callback returning the wrong number of values silently updates the wrong
    components. Neither shows up until somebody clicks it.
    """

    def click(self, x, y):
        class Event:
            index = (x, y)

        return Event()

    def test_remove_queues_the_particle_under_the_click(self, opened):
        from sem_analysis_app.callbacks import handle_image_click, set_click_mode

        state, _loading, _root, _frames = opened
        set_click_mode("delete")
        _viz, status = handle_image_click(self.click(60, 60))
        assert "Queued" in status
        assert len(state.pending_deletes) == 1

    def test_clicking_background_removes_nothing(self, opened):
        from sem_analysis_app.callbacks import handle_image_click, set_click_mode

        state, _loading, _root, _frames = opened
        set_click_mode("delete")
        _viz, status = handle_image_click(self.click(5, 290))
        assert "No particle" in status
        assert state.pending_deletes == []

    def test_merge_collects_particles(self, opened):
        from sem_analysis_app.callbacks import handle_image_click, set_click_mode

        state, _loading, _root, _frames = opened
        set_click_mode("merge")
        handle_image_click(self.click(60, 60))
        _viz, status = handle_image_click(self.click(150, 150))
        assert len(state.pending_merge) == 2
        assert "2 particles" in status

    def test_add_says_it_needs_a_model(self, opened):
        from sem_analysis_app.callbacks import handle_image_click, set_click_mode

        state, _loading, _root, _frames = opened
        state.segmenter = None
        set_click_mode("add")
        _viz, status = handle_image_click(self.click(20, 20))
        assert "needs a model" in status

    def test_redraw_says_it_needs_a_model(self, opened):
        from sem_analysis_app.callbacks import handle_image_click, set_click_mode

        state, _loading, _root, _frames = opened
        state.segmenter = None
        set_click_mode("point_refine")
        _viz, status = handle_image_click(self.click(60, 60))
        assert "needs a model" in status

    def test_apply_commits_a_removal(self, opened):
        from sem_analysis_app.callbacks import (apply_refinement_changes,
                                                handle_image_click, set_click_mode)

        state, _loading, _root, _frames = opened
        before = len(state.analyzer.regions)
        set_click_mode("delete")
        handle_image_click(self.click(60, 60))
        outputs = apply_refinement_changes()
        assert len(outputs) == 5
        assert len(state.analyzer.regions) == before - 1

    def test_undo_puts_it_back(self, opened):
        from sem_analysis_app.callbacks import (apply_refinement_changes,
                                                handle_image_click,
                                                set_click_mode, undo_last_action)

        state, _loading, _root, _frames = opened
        before = len(state.analyzer.regions)
        set_click_mode("delete")
        handle_image_click(self.click(60, 60))
        apply_refinement_changes()
        outputs = undo_last_action()
        assert len(outputs) == 3
        assert len(state.analyzer.regions) == before

    def test_discarding_pending_clicks_clears_them(self, opened):
        from sem_analysis_app.callbacks import (clear_pending_changes,
                                                handle_image_click, set_click_mode)

        state, _loading, _root, _frames = opened
        set_click_mode("delete")
        handle_image_click(self.click(60, 60))
        outputs = clear_pending_changes()
        assert len(outputs) == 2
        assert state.pending_deletes == []

    def test_dropping_edge_particles(self, opened):
        from sem_analysis_app.callbacks import clear_edge_particles

        state, _loading, _root, _frames = opened
        outputs = clear_edge_particles(90)
        assert len(outputs) == 5
        assert len(state.analyzer.regions) == 1

    def test_clearing_the_frame(self, opened):
        from sem_analysis_app.callbacks import clear_all_particles

        state, _loading, _root, _frames = opened
        outputs = clear_all_particles()
        assert len(outputs) == 5
        assert len(state.analyzer.regions) == 0

    def test_the_tool_switch_shows_the_redraw_controls(self, opened):
        from sem_analysis_app.callbacks import set_click_mode

        for mode in ("delete", "add", "merge", "point_refine"):
            message, controls = set_click_mode(mode)
            assert message and not message.startswith("Unknown")
            assert controls.get("visible") is (mode == "point_refine")

    def test_the_point_type_switch(self, opened):
        from sem_analysis_app.callbacks import set_point_type

        assert "NEGATIVE" in set_point_type("negative")
        assert "POSITIVE" in set_point_type("positive")

    def test_resetting_the_redraw_points(self, opened):
        from sem_analysis_app.callbacks import reset_point_refine

        state, _loading, _root, _frames = opened
        state.point_refine_points = [(1, 2)]
        outputs = reset_point_refine()
        assert len(outputs) == 2
        assert state.point_refine_points == []

    def test_the_minimum_size_control(self, opened):
        from sem_analysis_app.callbacks import set_min_particle_size

        state, _loading, _root, _frames = opened
        assert "500" in set_min_particle_size(500)
        assert state.min_particle_size == 500

    def test_the_numbering_toggle_redraws(self, opened):
        from sem_analysis_app.callbacks import toggle_particle_numbers

        state, _loading, _root, _frames = opened
        picture = toggle_particle_numbers(False)
        assert picture is not None
        assert state.show_particle_numbers is False

    def test_the_results_tab(self, opened):
        from sem_analysis_app.callbacks import get_session_summary

        _state, loading, _root, _frames = opened
        loading.save_here()
        outputs = get_session_summary()
        assert len(outputs) == 4
        assert outputs[0] is not None

    def test_removing_duplicates(self, opened):
        from sem_analysis_app.callbacks import check_and_remove_duplicates

        _state, loading, _root, _frames = opened
        loading.save_here()
        loading.save_here()
        table, status = check_and_remove_duplicates()
        assert "Removed 1" in status

    def test_deleting_a_row(self, opened):
        from sem_analysis_app.callbacks import delete_result_row

        _state, loading, _root, _frames = opened
        loading.save_here()
        outputs = delete_result_row("0: X1_0001.png")
        assert len(outputs) == 5
        assert "Deleted" in outputs[4]

    def test_exporting_hands_over_a_file_that_exists(self, opened):
        # Gradio serves only what it is allowed to read, and the results file
        # sits beside the images on another drive. Returning that path made the
        # download button do nothing, silently. The export is a copy, in a
        # directory Gradio will serve.
        from pathlib import Path

        from sem_analysis_app.callbacks import export_results

        _state, loading, root, _frames = opened
        loading.save_here()
        path, status = export_results()
        assert path is not None and Path(path).exists()
        assert Path(path).parent != root, "exported from the results folder itself"
        assert "1 rows" in status and str(root) in status

    def test_the_export_holds_the_results(self, opened):
        import pandas as pd
        from sem_analysis_app.callbacks import export_results

        _state, loading, _root, _frames = opened
        loading.save_here()
        loading.save_and_next()
        path, _status = export_results()
        assert list(pd.read_csv(path)["file_name"]) == ["X1_0001.png", "X1_0001.png"]

    def test_exporting_nothing_says_so(self, opened):
        from sem_analysis_app.callbacks import export_results

        path, status = export_results()
        assert path is None
        assert "save a frame first" in status

    def test_the_plots(self, opened):
        from sem_analysis_app.callbacks import update_histogram_plots

        _state, loading, _root, _frames = opened
        loading.save_here()
        picture, status = update_histogram_plots()
        assert picture is not None, status


class TestTheInterfaceBuilds:
    def test_every_wired_callback_exists(self):
        # Building the Blocks resolves every handler and every output component,
        # so a typo in the wiring fails here rather than on the first click.
        from sem_review_app.ui import create_interface

        assert create_interface() is not None


class TestTheShapeOfWhatCallbacksReturn:
    """
    Types, not just counts.

    Building a Blocks checks that every output component exists; it does not
    check that a callback returns something the component can display. A
    DataFrame sent to a Markdown fails at the moment of the click, in the server
    log, with the interface simply not updating — which is how the review app's
    Clear all silently did nothing.
    """

    def test_the_refinement_callbacks_end_with_two_tables(self, opened):
        import pandas as pd
        from sem_analysis_app.callbacks import (apply_refinement_changes,
                                                clear_all_particles,
                                                clear_edge_particles)

        for call in (lambda: clear_edge_particles(5),
                     lambda: clear_all_particles()):
            outputs = call()
            assert len(outputs) == 5
            assert isinstance(outputs[1], pd.DataFrame), "per-frame table"
            assert isinstance(outputs[2], str), "status line"
            assert isinstance(outputs[3], pd.DataFrame), "this frame's particles"
            assert isinstance(outputs[4], pd.DataFrame), "this frame's summary"

    def test_the_session_summary_ends_with_text_and_a_dropdown(self, opened):
        import pandas as pd
        from sem_analysis_app.callbacks import get_session_summary

        _state, loading, _root, _frames = opened
        loading.save_here()
        table, progress, stats, dropdown = get_session_summary()
        assert isinstance(table, pd.DataFrame)
        assert isinstance(progress, str) and isinstance(stats, str)
        assert isinstance(dropdown, dict) and "choices" in dropdown


class TestReconnecting:
    """
    A browser that connects to an already-open folder must see it.

    The state is process-wide, the gallery is per-connection: opening a folder
    with --folder, or refreshing the page, left the gallery empty and the app
    looking like it had loaded nothing.
    """

    def test_a_fresh_connection_gets_the_open_folder(self, opened):
        _state, loading, root, _frames = opened
        status, gallery, folder = loading.restore()
        assert "3 frames" in status
        assert len(gallery) == 3
        assert folder == str(root)

    def test_it_says_how_many_are_already_reviewed(self, opened):
        _state, loading, _root, _frames = opened
        loading.save_here()
        status, _gallery, _folder = loading.restore()
        assert "1 already reviewed" in status

    def test_nothing_open_returns_nothing(self):
        from sem_analysis_app.state import state
        from sem_review_app import loading

        state.image_paths = []
        state.review_root = None
        assert loading.restore() == ("", [], "")


class TestTheLauncher:
    """
    A busy port is ordinary, not an error worth a stack trace.

    Two of these apps and the analysis app may all be running; whichever starts
    second used to die with Gradio's OSError.
    """

    def test_it_moves_to_the_next_port(self, monkeypatch):
        from sem_review_app import __main__ as launcher
        from sem_review_app import ui

        tried = []

        class Blocks:
            def launch(self, **kwargs):
                tried.append(kwargs["server_port"])
                if len(tried) < 3:
                    raise OSError("Cannot find empty port in range: x-x")

        monkeypatch.setattr(ui, "create_interface", lambda: Blocks())
        launcher.main(["--port", "7870"])
        assert tried == [7870, 7871, 7872]

    def test_strict_port_does_not_wander(self, monkeypatch):
        from sem_review_app import __main__ as launcher
        from sem_review_app import ui

        class Blocks:
            def launch(self, **kwargs):
                raise OSError("Cannot find empty port in range: x-x")

        monkeypatch.setattr(ui, "create_interface", lambda: Blocks())
        with pytest.raises(SystemExit):
            launcher.main(["--port", "7870", "--strict-port"])

    def test_another_oserror_is_not_swallowed(self, monkeypatch):
        from sem_review_app import __main__ as launcher
        from sem_review_app import ui

        class Blocks:
            def launch(self, **kwargs):
                raise OSError("the network is on fire")

        monkeypatch.setattr(ui, "create_interface", lambda: Blocks())
        with pytest.raises(OSError, match="on fire"):
            launcher.main(["--port", "7870"])


class TestFramesAreNamedAsTheAnalysisNamedThem:
    """
    The app reads PNG copies; the analysis measured TIFFs.

    Saving under the PNG's own name left reviewed_results.csv keyed
    "O1_0001.png" against analysis_results.csv's "O1_0001.tif", so the automatic
    and the reviewed answer for one frame could not be joined at all.
    """

    def test_the_reviewed_row_uses_the_analysis_name(self, tmp_path):
        import pandas as pd
        from sem_analysis_app.state import state
        from sem_review_app import loading

        # An analysis that measured TIFFs, reviewed from PNG copies.
        root = tmp_path / "T1_analysis"
        for sub in ("raw", "mask"):
            (root / sub).mkdir(parents=True)
        mask, _ = make_disk_mask(shape=(200, 200), centers_radii=((100, 100, 30),))
        Image.fromarray(np.where(mask, 30, 200).astype(np.uint8)).save(
            root / "raw" / "T1_0001.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            root / "mask" / "T1_0001.png")

        from sem_particle_analysis import ParticleAnalyzer

        analyzer = ParticleAnalyzer(conversion_factor=2.0, min_size=30)
        analyzer.analyze_mask(mask, min_size=30, remove_border=False)
        ResultsManager(csv_file=str(root / "analysis_results.csv")).add_result(
            "T1_0001.tif", analyzer.get_measurements(in_nm=True),
            scale_method="box_ocr")

        loading.load_folder(str(root), min_size=30)
        loading.open_index(0)
        assert loading.recorded_name() == "T1_0001.tif"
        loading.save_here()

        reviewed = pd.read_csv(root / loading.REVIEWED_CSV)
        assert list(reviewed["file_name"]) == ["T1_0001.tif"]
        state.analyzer = None
        state.results_manager = None
        state.image_paths = []

    def test_it_falls_back_to_the_file_it_opened(self, opened):
        # No analysis CSV, or a frame missing from it: the PNG's own name is
        # still better than nothing.
        state, loading, _root, _frames = opened
        state.review_names = {}
        assert loading.recorded_name() is None
        status, _gallery = loading.save_here()
        assert status.startswith("✅")


class TestTheScalePanel:
    """
    Correcting a mask cannot fix a misread bar.

    Every row a batch writes says "+unconfirmed", and two frames in the NIOSH
    set read a "1 um" bar as "7 um" — every size on them seven times too large.
    The panel exists so that both can be dealt with from the same screen as the
    mask: look at what the bar measures, say it is right, or measure it again by
    hand.
    """

    def test_it_says_when_nobody_has_checked(self, opened):
        from sem_review_app import scale

        text = scale.summary()
        assert "2.5 nm/px" in text
        assert "nobody has checked" in text

    def test_confirming_changes_what_gets_saved(self, opened):
        import pandas as pd
        from sem_review_app import scale

        state, loading, root, _frames = opened
        assert state.scale_calibration.provenance == "box_ocr+unconfirmed"

        status, summary, _header = scale.confirm()
        assert status.startswith("✅")
        assert "confirmed" in summary
        assert state.scale_calibration.provenance == "box_ocr"

        loading.save_here()
        saved = pd.read_csv(root / loading.REVIEWED_CSV)
        assert saved["scale_method"].iloc[0] == "box_ocr"

    def test_clicks_go_to_the_bar_while_it_is_asking(self, opened):
        from sem_review_app import loading, scale

        class Event:
            def __init__(self, x, y):
                self.index = (x, y)

        state, _loading, _root, _frames = opened
        assert scale.clicking() is False

        scale.start_points()
        assert scale.clicking() is True
        _viz, status = loading.review_click(Event(10, 20))
        assert "One end" in status
        _viz, status = loading.review_click(Event(110, 20))
        assert "100.0 px" in status
        assert len(state.scale_points) == 2

    def test_a_third_click_starts_again(self, opened):
        from sem_review_app import scale

        state, _loading, _root, _frames = opened
        scale.start_points()
        for point in ((10, 20), (110, 20), (50, 50)):
            scale.add_point(*point)
        assert state.scale_points == [(50.0, 50.0)]

    def test_two_points_replace_the_scale_and_the_sizes(self, opened):
        from sem_review_app import scale

        state, _loading, _root, _frames = opened
        before = state.analyzer.get_measurements(in_nm=True)["diameters"][0]

        scale.start_points()
        scale.add_point(10, 20)
        scale.add_point(110, 20)          # 100 px
        _viz, status, summary, _header, _table = scale.apply_points(500, "nm")

        # 500 nm over 100 px is 5 nm/px, twice the 2.5 the analysis recorded.
        assert state.scale_calibration.nm_per_px == pytest.approx(5.0)
        assert state.analyzer.conversion == pytest.approx(5.0)
        after = state.analyzer.get_measurements(in_nm=True)["diameters"][0]
        assert after == pytest.approx(2 * before)
        assert "2x" in status
        assert "confirmed" in summary

    def test_a_hand_measured_scale_is_saved_as_such(self, opened):
        import pandas as pd
        from sem_review_app import scale

        _state, loading, root, _frames = opened
        scale.start_points()
        scale.add_point(10, 20)
        scale.add_point(110, 20)
        scale.apply_points(500, "nm")
        loading.save_here()

        saved = pd.read_csv(root / loading.REVIEWED_CSV)
        assert saved["scale_method"].iloc[0] == "two_points"
        assert saved["nm_per_px"].iloc[0] == pytest.approx(5.0)

    def test_the_fix_survives_leaving_the_frame(self, opened):
        from sem_review_app import scale

        state, loading, _root, _frames = opened
        scale.start_points()
        scale.add_point(10, 20)
        scale.add_point(110, 20)
        scale.apply_points(500, "nm")

        loading.skip_to_next()
        loading.go_back()
        assert state.scale_calibration.nm_per_px == pytest.approx(5.0)

    def test_applying_without_two_points_says_so(self, opened):
        from sem_review_app import scale

        scale.start_points()
        _viz, status, _summary, _header, _table = scale.apply_points(500, "nm")
        assert "both ends" in status

    def test_applying_without_a_length_says_so(self, opened):
        from sem_review_app import scale

        scale.start_points()
        scale.add_point(10, 20)
        scale.add_point(110, 20)
        _viz, status, _summary, _header, _table = scale.apply_points(None, "nm")
        assert "printed" in status

    def test_two_points_in_the_same_place_are_refused(self, opened):
        from sem_review_app import scale

        scale.start_points()
        scale.add_point(10, 20)
        scale.add_point(10, 21)
        _viz, status, _summary, _header, _table = scale.apply_points(500, "nm")
        assert status.startswith("❌")

    def test_cancelling_gives_the_clicks_back_to_the_mask(self, opened):
        from sem_review_app import scale

        state, _loading, _root, _frames = opened
        scale.start_points()
        scale.add_point(10, 20)
        status, picture = scale.clear_points()
        assert scale.clicking() is False
        assert state.scale_points == []
        assert "editing the mask" in status

    def test_opening_a_frame_stops_it_asking_for_ends(self, opened):
        # Otherwise the next frame's first click silently becomes a bar end.
        from sem_review_app import scale

        state, loading, _root, _frames = opened
        scale.start_points()
        loading.skip_to_next()
        assert scale.clicking() is False
        assert state.scale_points == []


class TestReviewedFramesComeBackTicked:
    """
    The gallery has to show what has already been done.

    Ticks were matched on the full file name. The review app reads PNG copies of
    frames the analysis measured as TIFFs, so the names never agreed and a folder
    reopened after a break looked untouched — with no way to tell which frames
    still needed doing except by opening every one.
    """

    def test_a_saved_frame_is_ticked_when_the_folder_is_reopened(self, opened):
        state, loading, root, _frames = opened
        loading.save_here()

        status, gallery = loading.load_folder(str(root), min_size=30)
        assert "1 already reviewed" in status
        assert gallery[0][1].startswith("✅")
        assert all(item[1].startswith("⚪") for item in gallery[1:])

    def test_it_matches_across_a_change_of_extension(self, opened):
        # What the analysis recorded as .tif is read here as .png.
        import pandas as pd

        state, loading, root, _frames = opened
        loading.save_here()

        path = root / loading.REVIEWED_CSV
        table = pd.read_csv(path)
        table["file_name"] = [n.replace(".png", ".tif") for n in table.file_name]
        table.to_csv(path, index=False)

        status, gallery = loading.load_folder(str(root), min_size=30)
        assert "1 already reviewed" in status
        assert gallery[0][1].startswith("✅")

    def test_the_count_shown_is_the_reviewed_one(self, opened):
        state, loading, _root, _frames = opened
        state.analyzer.delete_particles([state.analyzer.regions[0].label])
        loading.save_here()
        _status, gallery, _folder = loading.restore()
        assert "(2)" in gallery[0][1]
