"""
Apourvoir Babel Module

Multivector canonical space combining:
- P: Phonetic dimension (sound, phonemes, pronunciation)
- S: Syntax dimension (string structure, morphemes, grammar)
- M: Semantic dimension (meaning, canonical projection on pou/voir/savoir)

This module enables:
- Compression of embeddings via 3D canonical projection
- Multilingual support through phonetic encoding
- Syntax-aware semantic relationships
- Thales' theorem validation for geometric consistency
"""

from .phonetic import PhoneticEncoder, PHONEME_MAP, PHONEME_WEIGHTS
from .phonetic.phoneme_map import PHONEME_EMBEDDING_DIM
from .syntax import SyntaxTreeBuilder, SyntaxEncoder, SyntaxTree, SyntaxNode
from .semantic import CanonicalProjector, CanonicalConfig
from .fusion import BabelSpace, BabelConfig, BabelAlgebra
from .validation import BabelMetrics

__all__ = [
    # Phonetic
    "PhoneticEncoder",
    "PHONEME_MAP",
    "PHONEME_WEIGHTS",
    "PHONEME_EMBEDDING_DIM",
    
    # Syntax
    "SyntaxTreeBuilder",
    "SyntaxEncoder",
    "SyntaxTree",
    "SyntaxNode",
    
    # Semantic
    "CanonicalProjector",
    "CanonicalConfig",
    
    # Fusion
    "BabelSpace",
    "BabelConfig",
    "BabelAlgebra",
    
    # Validation
    "BabelMetrics",
]
