"""
Apourvoir - Vector Alignment of Knowledge through the Prism of Power.

A framework for modeling, analyzing, and optimizing the alignment between
knowledge structures and power dynamics using vector-based approaches.

Main Modules:
- core: Fundamental mathematical structures and operations
- utils: Utility functions and helper classes
- visualization: Visualization tools for the framework
"""

from . import core
from . import utils

__version__ = "0.1.0"
__author__ = "Albo-co"
__description__ = "Vector Alignment of Knowledge through the Prism of Power"

# Re-export main classes for convenience
from .core.alignment import (
    SavoirVector,
    PouvoirMatrix,
    SavoirSpace,
    AlignmentEngine,
    create_alignment_tensor,
    calculate_alignment_gradient
)

from .utils.quantization import (
    Quantizer,
    UniformQuantizer,
    AdaptiveQuantizer,
    MatrixQuantizer,
    VectorQuantizer
)

__all__ = [
    'core',
    'utils',
    'SavoirVector',
    'PouvoirMatrix',
    'SavoirSpace',
    'AlignmentEngine',
    'create_alignment_tensor',
    'calculate_alignment_gradient',
    'Quantizer',
    'UniformQuantizer',
    'AdaptiveQuantizer',
    'MatrixQuantizer',
    'VectorQuantizer'
]
