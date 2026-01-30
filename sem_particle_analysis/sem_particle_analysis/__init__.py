"""
SEM Particle Analysis Package

A Python package for segmenting and analyzing particles in SEM/TEM images
using Meta's Segment Anything Model (SAM).
"""

from .model import SAMModel
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

__version__ = "0.1.0"
__all__ = [
    "SAMModel",
    "ScaleDetector",
    "ParticleSegmenter",
    "ParticleAnalyzer",
    "ResultsManager",
]

# Only add InteractiveRefiner if ipywidgets is available
if _has_interactive:
    __all__.append("InteractiveRefiner")
