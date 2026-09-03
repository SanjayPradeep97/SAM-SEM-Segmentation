"""
Click-driven refinement: what a queued click still refers to when it is applied.

Every edit relabels the mask from scratch — ``delete_particles`` and
``merge_particles`` both end in ``measure.label`` — so a label recorded when a
particle was clicked names a *different* particle once anything else has been
applied. Queue a delete and a merge together and the merge joined whichever
particles happened to inherit those numbers.

The other half of this file is the redraw (point-refine) path, where two
separate faults made a click do nothing at all: the region SAM was allowed to
answer in was fixed to the starting particle's own bounding box, and an empty
result was still stored as a refinement.

Needs gradio, which the callbacks import, but never touches SAM.
"""

import numpy as np
import pytest

from sem_particle_analysis import ParticleAnalyzer
from synthetic import make_disk_mask

pytest.importorskip("gradio")


@pytest.fixture
def three_particles():
    """Three well-separated disks, and the app state holding them."""
    from sem_analysis_app.state import state

    mask, _ = make_disk_mask(shape=(300, 300), centers_radii=(
        (60, 60, 20), (150, 150, 20), (240, 240, 20)))
    analyzer = ParticleAnalyzer(conversion_factor=1.0, min_size=30)
    analyzer.analyze_mask(mask, remove_border=False)

    kept = (state.analyzer, state.cropped_image, state.pending_deletes,
            state.pending_merge, state.pending_anchors,
            state.point_refine_particle, state.point_refine_preview_mask,
            state.point_refine_points, state.mask_history, state.undo_history)
    state.analyzer = analyzer
    state.cropped_image = np.zeros((300, 300, 3), dtype=np.uint8)
    state.pending_deletes, state.pending_merge, state.pending_anchors = [], [], {}
    state.point_refine_particle = None
    state.point_refine_preview_mask = None
    state.point_refine_points, state.point_refine_labels = [], []
    state.mask_history, state.undo_history = [], []
    yield state
    (state.analyzer, state.cropped_image, state.pending_deletes,
     state.pending_merge, state.pending_anchors, state.point_refine_particle,
     state.point_refine_preview_mask, state.point_refine_points,
     state.mask_history, state.undo_history) = kept


def centre_of(analyzer, label):
    """A pixel inside the labelled particle, as (x, y)."""
    rows, cols = np.where(analyzer.labeled_mask == label)
    return int(cols[len(cols) // 2]), int(rows[len(rows) // 2])


def label_at(analyzer, point):
    x, y = point
    return int(analyzer.labeled_mask[y, x])


class TestASelectionSurvivesARelabel:
    def test_a_deleted_particle_does_not_shift_a_merge(self, three_particles):
        # Delete the first, merge the other two. Deleting renumbers 2 and 3 to
        # 1 and 2, so the un-anchored merge used to act on the wrong labels.
        from sem_analysis_app.callbacks.refinement import apply_refinement_changes

        state = three_particles
        first, second, third = (r.label for r in state.analyzer.regions)
        doomed = centre_of(state.analyzer, first)
        keep_a = centre_of(state.analyzer, second)
        keep_b = centre_of(state.analyzer, third)

        state.pending_deletes = [first]
        state.anchor_selection(first, doomed)
        state.pending_merge = [second, third]
        state.anchor_selection(second, keep_a)
        state.anchor_selection(third, keep_b)

        apply_refinement_changes()

        # The deleted one is gone and both survivors are still there — a merge
        # of two far-apart particles cannot join them, but it must not have
        # touched anything else either.
        assert label_at(state.analyzer, doomed) == 0
        assert label_at(state.analyzer, keep_a) != 0
        assert label_at(state.analyzer, keep_b) != 0

    def test_the_right_particle_is_redrawn_after_a_delete(self, three_particles):
        from sem_analysis_app.callbacks.refinement import apply_refinement_changes

        state = three_particles
        first, second, _third = (r.label for r in state.analyzer.regions)
        doomed = centre_of(state.analyzer, first)
        target = centre_of(state.analyzer, second)

        state.pending_deletes = [first]
        state.anchor_selection(first, doomed)

        # Redraw the second particle as a square over roughly the same place.
        state.point_refine_particle = second
        state.anchor_selection(second, target)
        preview = np.zeros((300, 300), dtype=bool)
        preview[135:170, 135:170] = True
        state.point_refine_preview_mask = preview
        state.point_refine_points = [target]

        apply_refinement_changes()

        # Two particles left: the redrawn one and the untouched third.
        assert len(state.analyzer.regions) == 2
        assert label_at(state.analyzer, doomed) == 0
        assert state.analyzer.mask[137, 137], "the redrawn square is not there"
        # And the third was not the one deleted in the redrawn one's place.
        assert state.analyzer.mask[240, 240]

    def test_a_selection_whose_particle_is_gone_is_dropped(self, three_particles):
        from sem_analysis_app.callbacks.refinement import apply_refinement_changes

        state = three_particles
        first, second, _ = (r.label for r in state.analyzer.regions)
        point = centre_of(state.analyzer, first)

        # The same particle queued for deletion and for merging.
        state.pending_deletes = [first]
        state.anchor_selection(first, point)
        state.pending_merge = [first, second]
        state.anchor_selection(second, centre_of(state.analyzer, second))

        _viz, _df, status, _r, _s = apply_refinement_changes()

        assert "Deleted 1 particles" in status
        # Only one of the two merge targets still exists, so the merge is
        # reported as impossible rather than applied to something else.
        assert "at least 2" in status

    def test_an_edge_cleanup_repoints_queued_clicks(self, three_particles):
        from sem_analysis_app.callbacks.refinement import clear_edge_particles

        state = three_particles
        first, second, third = (r.label for r in state.analyzer.regions)
        middle = centre_of(state.analyzer, second)
        state.pending_deletes = [second]
        state.anchor_selection(second, middle)

        # Drops the two particles near the frame edge, renumbering the middle.
        clear_edge_particles(90)

        assert len(state.analyzer.regions) == 1
        assert state.pending_deletes == [label_at(state.analyzer, middle)]
        assert state.pending_deletes[0] != 0


class TestTheRedrawRegion:
    """
    SAM cannot return anything outside the box it is given.

    The box used to be the starting particle's own bounding box, so a click
    beyond its edge — the click someone makes precisely when the mask has missed
    part of an object — could not change the answer at all.
    """

    def test_a_click_outside_the_particle_widens_the_box(self):
        from sem_analysis_app.callbacks.refinement import roi_box

        base = np.zeros((300, 300), dtype=bool)
        base[100:140, 100:140] = True

        without = roi_box(base, [], (300, 300))
        with_click = roi_box(base, [(260, 120)], (300, 300))

        assert without[0][2] < 260, "the starting box already reached the click"
        assert with_click[0][2] >= 260, "the click is still outside the box"

    def test_the_box_stays_inside_the_frame(self):
        from sem_analysis_app.callbacks.refinement import roi_box

        base = np.zeros((60, 60), dtype=bool)
        base[0:5, 0:5] = True
        box = roi_box(base, [(59, 59)], (60, 60))[0]
        assert box[0] >= 0 and box[1] >= 0
        assert box[2] <= 59 and box[3] <= 59

    def test_drawing_from_scratch_is_unconstrained(self):
        # No starting particle means no box: SAM sees the whole frame, which is
        # what makes it possible to draw a new particle anywhere.
        from sem_analysis_app.callbacks.refinement import roi_box

        assert roi_box(None, [(10, 10)], (300, 300)) is None
        assert roi_box(np.zeros((300, 300), bool), [(10, 10)], (300, 300)) is None


class TestAnEmptyRedrawIsNotARefinement:
    def test_it_leaves_the_particle_alone(self, three_particles):
        # Applying an empty preview used to delete the particle being refined
        # and add nothing back, so the particle simply vanished.
        from sem_analysis_app.callbacks.refinement import apply_refinement_changes

        state = three_particles
        target = state.analyzer.regions[1].label
        point = centre_of(state.analyzer, target)
        before = len(state.analyzer.regions)

        state.point_refine_particle = target
        state.anchor_selection(target, point)
        state.point_refine_preview_mask = np.zeros((300, 300), dtype=bool)

        _viz, _df, status, _r, _s = apply_refinement_changes()

        assert len(state.analyzer.regions) == before
        assert label_at(state.analyzer, point) != 0
        assert "Nothing to redraw" in status

    def test_it_does_not_count_as_a_change(self, three_particles):
        from sem_analysis_app.callbacks.refinement import apply_refinement_changes

        state = three_particles
        state.point_refine_preview_mask = np.zeros((300, 300), dtype=bool)
        _viz, _df, status, _r, _s = apply_refinement_changes()
        assert "Nothing to redraw" in status
        # Nothing was committed, so nothing was pushed onto the undo stack.
        assert state.mask_history == []
