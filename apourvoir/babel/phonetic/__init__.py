"""
Phonetic Dimension for Babel Space

Handles:
- Phoneme mapping (IPA)
- Phonetic encoding
- Phoneme weights (articulatory complexity)
"""

from .phoneme_map import PHONEME_MAP, PHONEME_WEIGHTS, PHONEME_EMBEDDING_DIM
from .phonetic_encoder import PhoneticEncoder

__all__ = [
    "PHONEME_MAP",
    "PHONEME_WEIGHTS",
    "PHONEME_EMBEDDING_DIM",
    "PhoneticEncoder",
]
