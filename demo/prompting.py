"""prompting.py -- how a click becomes a mask in the demo.

One small, explainable rule sits between SAM and the classifier, tuned for
TEM micrographs, where the particle is darker than the support film.  It is
evaluated in demo/README.md.  (Moving a click to the darkest nearby pixel was
also tried and measured no benefit, so clicks are used exactly as made.)

choose_mask(masks, scores, gray, mode)
    SAM returns three candidate masks per prompt with a confidence score.
    The score often prefers a single fibre inside a bundle, or a large patch
    of film.  The default rule keeps the LARGEST candidate that covers less
    than 60 % of the frame and is at least 30 % as dark, relative to its
    surroundings, as the darkest candidate.  MODE_SCORE keeps SAM's own
    highest-scoring candidate instead.
"""
from __future__ import annotations
import numpy as np

MODE_RULE = "largest dark candidate (default)"
MODE_SCORE = "SAM's highest score"


def choose_mask(masks, scores, gray, mode=MODE_RULE, area_cap=0.6, rel_contrast=0.3):
    """Pick one of SAM's candidate masks; returns a boolean array."""
    masks = [np.asarray(m).astype(bool) for m in masks]
    if mode == MODE_SCORE or len(masks) == 1:
        return masks[int(np.argmax(scores))]
    stats = []
    for m in masks:
        area = float(m.mean())
        contrast = (float(gray[~m].mean() - gray[m].mean())
                    if m.any() and (~m).any() else -1e9)
        stats.append((m, area, contrast))
    mx = max(s[2] for s in stats)
    valid = [s for s in stats if s[1] < area_cap and (mx <= 0 or s[2] >= rel_contrast * mx)]
    pool = valid or [s for s in stats if s[1] < area_cap] or stats
    return max(pool, key=lambda s: s[1])[0]
