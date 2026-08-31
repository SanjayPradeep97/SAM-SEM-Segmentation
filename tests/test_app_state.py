"""
Application state: the undo history that makes editing recoverable.

Applying an edit used to be a point of no return — the click-level undo stack
was cleared on Apply, and nothing recorded the mask beforehand. Since the
analyst applies constantly to see the result, one wrong Apply meant
re-segmenting the image from scratch, part way through a folder of hundreds.

Imports the state module directly rather than through the app, so this runs
without Gradio.
"""

import numpy as np
import pytest

from sem_analysis_app.state import AppState
from sem_particle_analysis import ParticleAnalyzer
from synthetic import make_disk_mask


@pytest.fixture
def state_with_particles():
    mask, _ = make_disk_mask(shape=(300, 300), centers_radii=(
        (70, 70, 22), (150, 90, 22), (220, 200, 22)))
    analyzer = ParticleAnalyzer(conversion_factor=2.0, min_size=30)
    analyzer.analyze_mask(mask, remove_border=False)

    state = AppState()
    state.analyzer = analyzer
    return state


class TestMaskSnapshots:
    def test_a_snapshot_restores_a_deleted_particle(self, state_with_particles):
        state = state_with_particles
        before = len(state.analyzer.regions)

        state.snapshot_mask()
        state.analyzer.delete_particles([state.analyzer.regions[0].label])
        assert len(state.analyzer.regions) == before - 1

        assert state.restore_mask() is True
        assert len(state.analyzer.regions) == before

    def test_snapshots_stack(self, state_with_particles):
        state = state_with_particles
        counts = [len(state.analyzer.regions)]
        for _ in range(2):
            state.snapshot_mask()
            state.analyzer.delete_particles([state.analyzer.regions[0].label])
            counts.append(len(state.analyzer.regions))

        assert state.restore_mask() is True
        assert len(state.analyzer.regions) == counts[1]
        assert state.restore_mask() is True
        assert len(state.analyzer.regions) == counts[0]

    def test_restoring_with_no_snapshot_reports_so(self, state_with_particles):
        assert state_with_particles.restore_mask() is False

    def test_restoring_without_an_analyzer_is_harmless(self):
        assert AppState().restore_mask() is False

    def test_snapshotting_without_an_analyzer_is_harmless(self):
        state = AppState()
        state.snapshot_mask()
        assert state.mask_history == []

    def test_the_stack_is_bounded(self, state_with_particles):
        # Each snapshot is a full-frame boolean mask; an unbounded stack would
        # grow without limit across a long editing session.
        state = state_with_particles
        for _ in range(state.MAX_MASK_HISTORY + 8):
            state.snapshot_mask()
        assert len(state.mask_history) == state.MAX_MASK_HISTORY

    def test_the_oldest_snapshot_is_the_one_dropped(self, state_with_particles):
        state = state_with_particles
        state.MAX_MASK_HISTORY  # documented cap
        first = state.analyzer.mask.copy()
        state.snapshot_mask()
        for _ in range(state.MAX_MASK_HISTORY):
            state.analyzer.delete_particles([state.analyzer.regions[0].label]) \
                if state.analyzer.regions else None
            state.snapshot_mask()
        assert not np.array_equal(state.mask_history[0], first)

    def test_a_snapshot_is_a_copy_not_a_reference(self, state_with_particles):
        # Holding a reference would make the snapshot follow later edits and
        # restore nothing.
        state = state_with_particles
        state.snapshot_mask()
        state.analyzer.mask[:] = False
        state.analyzer._relabel_and_filter()
        assert len(state.analyzer.regions) == 0
        assert state.restore_mask() is True
        assert len(state.analyzer.regions) > 0


class TestResetClearsHistory:
    def test_opening_a_new_image_forgets_the_old_one(self, state_with_particles):
        state = state_with_particles
        state.snapshot_mask()
        assert state.mask_history

        state.reset_image_state()
        assert state.mask_history == []
        assert state.undo_history == []

    def test_reset_clears_the_frame_geometry_too(self, state_with_particles):
        state = state_with_particles
        state.modality = object()
        state.analysable_region = np.ones((10, 10), dtype=bool)
        state.reset_image_state()
        assert state.modality is None
        assert state.analysable_region is None
