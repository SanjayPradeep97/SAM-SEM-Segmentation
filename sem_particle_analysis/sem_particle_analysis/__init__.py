"""
SEM Particle Analysis Package

A Python package for segmenting and analyzing particles in SEM/TEM images
using Meta's Segment Anything Model (SAM).
"""

from .model import SAMModel, discover_checkpoints, infer_model_type
from .scale_detection import ScaleDetector
from .segmentation import ParticleSegmenter
from .analysis import ParticleAnalyzer
from .data_manager import ResultsManager

# Optional import for Jupyter notebook support
try:
    from .interactive import InteractiveRefiner
    _has_interactive = True
except ImportError:
    InteractiveRefiner = None
    _has_interactive = False

# Single source of truth is the version in pyproject.toml; read it back from the
# installed distribution so the two can't drift apart.
try:
    from importlib.metadata import PackageNotFoundError, version as _version

    __version__ = _version("sem-particle-analysis")
except (ImportError, PackageNotFoundError):  # not installed (e.g. run from source tree)
    __version__ = "unknown"
__all__ = [
    "SAMModel",
    "discover_checkpoints",
    "infer_model_type",
    "ScaleDetector",
    "ParticleSegmenter",
    "ParticleAnalyzer",
    "ResultsManager",
]

# Only add InteractiveRefiner if ipywidgets is available
if _has_interactive:
    __all__.append("InteractiveRefiner")
