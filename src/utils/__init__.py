"""
Utilities for Apourvoir - Vector Alignment of Knowledge through the Prism of Power.

This module provides utility functions and classes for working with the
core alignment framework, including quantization, visualization, and data handling.
"""

from .quantization import (
    Quantizer,
    UniformQuantizer,
    AdaptiveQuantizer,
    MatrixQuantizer,
    VectorQuantizer,
    quantization_error,
    relative_quantization_error,
    quantization_snr,
    compression_ratio
)

__all__ = [
    'Quantizer',
    'UniformQuantizer',
    'AdaptiveQuantizer',
    'MatrixQuantizer',
    'VectorQuantizer',
    'quantization_error',
    'relative_quantization_error',
    'quantization_snr',
    'compression_ratio'
]

__version__ = "0.1.0"
