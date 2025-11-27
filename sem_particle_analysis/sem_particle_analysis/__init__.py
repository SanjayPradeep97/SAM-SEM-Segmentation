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
from .interactive import InteractiveRefiner

__version__ = "0.1.0"
__all__ = [
    "SAMModel",
    "ScaleDetector",
    "ParticleSegmenter",
    "ParticleAnalyzer",
    "ResultsManager",
    "InteractiveRefiner",
]
