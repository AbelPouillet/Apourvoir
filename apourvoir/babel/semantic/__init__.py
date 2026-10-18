"""
Semantic Module for Babel Space

Provides semantic projection and encoding for the M (semantic) dimension.

Components:
- CanonicalProjector: Projects embeddings onto the canonical 3D space (pou/voir/z(f))
- SemanticEncoder: Converts semantic information to vector embeddings
"""

from .canonical import CanonicalProjector, CanonicalConfig
from .embedder import SemanticEmbedder, DEFAULT_EMBEDDING_MODEL

__all__ = [
    "CanonicalProjector",
    "CanonicalConfig",
    "SemanticEmbedder",
    "DEFAULT_EMBEDDING_MODEL",
]
