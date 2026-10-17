"""
Core module for Apourvoir - Vector Alignment of Knowledge through the Prism of Power.

This module provides the fundamental building blocks for modeling, analyzing,
and optimizing the alignment between knowledge structures and power dynamics.
"""

from .alignment import (
    SavoirVector,
    PouvoirMatrix,
    SavoirSpace,
    AlignmentEngine,
    create_alignment_tensor,
    calculate_alignment_gradient
)

__all__ = [
    'SavoirVector',
    'PouvoirMatrix',
    'SavoirSpace',
    'AlignmentEngine',
    'create_alignment_tensor',
    'calculate_alignment_gradient'
]

__version__ = "0.1.0"
