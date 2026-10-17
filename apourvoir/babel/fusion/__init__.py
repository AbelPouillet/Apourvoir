"""
Fusion Module for Babel Space

Combines the P (phonetic), S (syntax), and M (semantic) dimensions
into a unified multivector canonical space.

Components:
- BabelSpace: Main class for managing the multivector space
- BabelConfig: Configuration for the fusion process
- BabelAlgebra: Algebraic operations on the canonical space
"""

from .babel_space import BabelSpace, BabelConfig
from .operations import BabelAlgebra

__all__ = [
    "BabelSpace",
    "BabelConfig",
    "BabelAlgebra",
]
