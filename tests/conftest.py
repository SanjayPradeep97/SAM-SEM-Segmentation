import os
import sys
from pathlib import Path

import pytest

# Test helpers live alongside the tests rather than in the installed package.
sys.path.insert(0, str(Path(__file__).parent))

# Candidate locations for SAM weights, in preference order. sam_weights/ is the
# documented location; the others cover a checkout that hasn't run the downloader.
_WEIGHT_DIRS = [
    Path(__file__).parent.parent / "sam_weights",
    Path.home() / "segment-anything" / "models",
]
_WEIGHT_FILES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


def find_sam_checkpoint(model_type="vit_b"):
    """Return a path to local SAM weights, or None if they aren't downloaded."""
    env = os.environ.get("SAM_CHECKPOINT")
    if env and Path(env).exists():
        return Path(env)
    for directory in _WEIGHT_DIRS:
        candidate = directory / _WEIGHT_FILES[model_type]
        if candidate.exists():
            return candidate
    return None


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs SAM weights or OCR models; deselect with -m 'not slow'")


@pytest.fixture(scope="session")
def scale_detector():
    """
    A single ScaleDetector for the whole session.

    EasyOCR takes several seconds to spin up its models, so building one per test
    would dominate the runtime.
    """
    pytest.importorskip("easyocr")
    from sem_particle_analysis import ScaleDetector

    return ScaleDetector(use_gpu=False)


@pytest.fixture(scope="session")
def sam_checkpoint():
    path = find_sam_checkpoint("vit_b")
    if path is None:
        pytest.skip("SAM weights not available; run python download_sam_weights.py")
    return path
