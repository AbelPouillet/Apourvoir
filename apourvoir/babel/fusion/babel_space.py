"""
Babel Space - Multivector Canonical Space

Combines P (phonetic), S (syntax), and M (semantic) dimensions
into a unified canonical space for maximal compression and operational leverage.

Key features:
- Multidimensional encoding: P + S + M
- Canonical 3D projection for semantic meaning
- Phonetic encoding for sound and pronunciation
- Syntax encoding for string structure
- Torus topology for periodic relationships
- Right-angle checks between vectors (exploratory; no semantic property demonstrated)
- Concentric layers for granularity

Example:
    >>> babel = BabelSpace()
    >>> embedding = babel.encode("pouvoir", lang="fr")
    >>> print(embedding.shape)  # Combined dimension
    >>> quantized = babel.quantize(embedding, n_bits=8)
    >>> reconstructed = babel.dequantize(quantized)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math

import torch
import numpy as np

from ..phonetic import PhoneticEncoder
from ..syntax import SyntaxTreeBuilder, SyntaxEncoder, SyntaxTree
from ..semantic import CanonicalProjector, CanonicalConfig, SemanticEmbedder

logger = logging.getLogger(__name__)


@dataclass
class BabelConfig:
    """Configuration for Babel space fusion."""
    # Dimension configuration
    phonetic_dim: int = 64
    syntax_dim: int = 64
    semantic_dim: int = 3  # Canonical 3D space (projection of the real embedding)
    semantic_model: Optional[str] = None  # sentence-transformers model (None = default)
    output_dim: int = 128  # Unused (kept for config compatibility); combined dim = P + S + semantic model dim
    
    # Weighting for dimensions
    p_weight: float = 1.0  # Phonetic weight
    s_weight: float = 1.0  # Syntax weight
    m_weight: float = 1.5  # Semantic weight (higher for meaning)
    
    # Quantization
    n_bits: int = 8  # Bits per angle for semantic quantization
    
    # Torus parameters
    R: float = 2.0
    r: float = 1.0
    
    # Language
    default_lang: str = "fr"
    
    # Granularity
    syntax_granularity: str = "word"  # char, morpheme, word, phrase
    
    # Layer configuration
    num_layers: int = 4
    layer_weights: Dict[str, float] = field(default_factory=lambda: {
        "fundamentals": 1.0,
        "sub_concepts": 0.8,
        "actions": 0.6,
        "techniques": 0.4,
    })


class BabelSpace:
    """
    Multivector canonical space combining P, S, and M dimensions.
    
    The Babel space encodes information across three dimensions:
    - P (Phonetic): Sound, phonemes, pronunciation
    - S (Syntax): String structure, morphemes, grammar
    - M (Semantic): Meaning via canonical projection
    
    This enables:
    - Maximal compression through angular quantization
    - Multilingual support through phonetic encoding
    - Syntax-aware semantic relationships
    - Mathematical operational leverage
    """
    
    def __init__(self, config: Optional[BabelConfig] = None):
        """
        Initialize the Babel space.
        
        Args:
            config: Configuration for the space
        """
        self.config = config or BabelConfig()
        
        # Initialize components
        self.phonetic_encoder = PhoneticEncoder(
            lang=self.config.default_lang,
            embedding_dim=self.config.phonetic_dim
        )
        
        self.syntax_builder = SyntaxTreeBuilder(
            lang=self.config.default_lang,
            granularity=self.config.syntax_granularity
        )
        
        self.syntax_encoder = SyntaxEncoder()
        
        self.canonical_projector = CanonicalProjector(
            CanonicalConfig(
                R=self.config.R,
                r=self.config.r,
                n_bits=self.config.n_bits
            )
        )
        
        self._semantic_embedder: Optional[SemanticEmbedder] = None

        # Reference cache
        self._reference_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        
        logger.info(f"BabelSpace initialized: P={self.config.phonetic_dim}, S={self.config.syntax_dim}, M={self.config.semantic_dim}")
    
    def encode(
        self,
        text: str,
        lang: Optional[str] = None,
        method: str = "combined"
    ) -> torch.Tensor:
        """
        Encode text into Babel space.
        
        Args:
            text: Text to encode
            lang: Language (default from config)
            method: Encoding method ('combined', 'phonetic', 'syntax', 'semantic')
        
        Returns:
            torch.Tensor: Babel space embedding
        """
        lang = lang or self.config.default_lang
        
        if method == "phonetic":
            return self._encode_phonetic(text, lang)
        elif method == "syntax":
            return self._encode_syntax(text, lang)
        elif method == "semantic":
            return self._encode_semantic(text, lang)
        else:
            return self._encode_combined(text, lang)
    
    def _encode_phonetic(self, text: str, lang: str) -> torch.Tensor:
        """Encode using only phonetic dimension."""
        # For now, use the first word
        word = text.split()[0] if text else ""
        return self.phonetic_encoder.get_phonetic_embedding(word)
    
    def _encode_syntax(self, text: str, lang: str) -> torch.Tensor:
        """Encode using only syntax dimension."""
        # Update syntax builder language
        self.syntax_builder.lang = lang
        tree = self.syntax_builder.build_tree(text)
        return self.syntax_encoder.encode_tree(tree)
    
    @property
    def semantic_embedder(self) -> SemanticEmbedder:
        """Real sentence-transformers embedder (loaded lazily, cached locally by HuggingFace)."""
        if self._semantic_embedder is None:
            self._semantic_embedder = SemanticEmbedder(self.config.semantic_model)
        return self._semantic_embedder

    def semantic_embedding(self, text: str) -> torch.Tensor:
        """Full-dimensional semantic embedding of the text (L2-normalised)."""
        return self.semantic_embedder.encode(text)

    def _ensure_canonical_basis(self) -> None:
        if getattr(self.canonical_projector, "basis", None) is None:
            self.canonical_projector.fit_axes_from_embedder(self.semantic_embedder)

    def _encode_semantic(self, text: str, lang: str) -> torch.Tensor:
        """
        Semantic dimension: real embedding projected onto the 3D canonical basis
        spanned by the embeddings of the reference words (pou / voir / savoir).
        The 3D projection keeps only part of the information; similarity
        should be computed with :meth:`semantic_embedding`.
        """
        self._ensure_canonical_basis()
        return self.canonical_projector.project_to_canonical(self.semantic_embedding(text), method="linear")

    def _encode_combined(self, text: str, lang: str) -> torch.Tensor:
        """
        Concatenate the three normalised, weighted views: [P | S | M].
        M is the full semantic embedding (not its 3D projection).
        """
        P = self._encode_phonetic(text, lang)
        S = self._encode_syntax(text, lang)
        M = self.semantic_embedding(text)

        parts = []
        for vec, weight in ((P, self.config.p_weight), (S, self.config.s_weight), (M, self.config.m_weight)):
            norm = torch.norm(vec)
            parts.append(vec / norm * weight if norm > 0 else vec)
        combined = torch.cat(parts)

        if torch.norm(combined) > 0:
            combined = combined / torch.norm(combined)

        return combined
    
    def quantize(
        self,
        embedding: torch.Tensor,
        n_bits: Optional[int] = None
    ) -> Tuple[Tuple[int, int], torch.Tensor]:
        """
        Quantize a Babel space embedding.
        
        Args:
            embedding: Input embedding
            n_bits: Number of bits for quantization
        
        Returns:
            Tuple of (quantized_angles, quantized_vector)
        """
        n_bits = n_bits or self.config.n_bits

        # Accept a 3D vector, a semantic embedding, or a combined [P | S | M] vector
        # (in which case only its M part is projected).
        if embedding.shape[-1] != 3:
            self._ensure_canonical_basis()
            sem_dim = self.canonical_projector.basis.shape[1]
            if embedding.shape[-1] != sem_dim:
                embedding = embedding[..., -sem_dim:]

        # Project to canonical 3D. Two angles keep only the direction of the
        # 3D projection: this is a lossy code, not a reversible compression.
        # The angles are those of the 3D coordinates; the canonical point is the
        # torus point with these (theta, phi), so dequantize() reconstructs it.
        coords = self.canonical_projector.project_to_canonical(embedding, method="linear")
        theta, phi = self.canonical_projector.get_angles(coords)
        canonical = self.canonical_projector._project_to_torus(coords)
        theta_q, phi_q = self.canonical_projector.quantize_angles(theta, phi, n_bits)
        
        return (theta_q, phi_q), canonical
    
    def dequantize(
        self,
        theta_q: int,
        phi_q: int,
        n_bits: Optional[int] = None
    ) -> torch.Tensor:
        """
        Dequantize from canonical angles.
        
        Args:
            theta_q: Quantized theta
            phi_q: Quantized phi
            n_bits: Number of bits used for quantization
        
        Returns:
            torch.Tensor: Reconstructed embedding
        """
        return self.canonical_projector.dequantize_from_canonical(
            theta_q, phi_q, n_bits
        )
    
    def encode_batch(
        self,
        texts: List[str],
        lang: Optional[str] = None,
        method: str = "combined"
    ) -> torch.Tensor:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of texts to encode
            lang: Language
            method: Encoding method
        
        Returns:
            torch.Tensor: Batch of embeddings
        """
        embeddings = [self.encode(text, lang, method) for text in texts]
        return torch.stack(embeddings)
    
    def get_relationship(
        self,
        text1: str,
        text2: str,
        text3: str,
        lang: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get canonical relationship between three texts via a right-angle check.
        
        Args:
            text1: First text
            text2: Second text (vertex)
            text3: Third text
            lang: Language
        
        Returns:
            Dict with relationship information
        """
        # Encode all three
        v1 = self._encode_semantic(text1, lang or self.config.default_lang)
        v2 = self._encode_semantic(text2, lang or self.config.default_lang)
        v3 = self._encode_semantic(text3, lang or self.config.default_lang)
        
        # Validate Thales
        is_thales = self.canonical_projector.validate_thales(v1, v2, v3)
        
        # Get angles
        angle_v1v2 = self._angle_between(v1, v2)
        angle_v2v3 = self._angle_between(v2, v3)
        angle_v1v3 = self._angle_between(v1, v3)
        
        return {
            "text1": text1,
            "text2": text2,
            "text3": text3,
            "thales_valid": is_thales,
            "angle_v1v2_deg": math.degrees(angle_v1v2),
            "angle_v2v3_deg": math.degrees(angle_v2v3),
            "angle_v1v3_deg": math.degrees(angle_v1v3),
            "v1": v1.tolist(),
            "v2": v2.tolist(),
            "v3": v3.tolist(),
        }
    
    def _angle_between(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """Calculate angle between two vectors."""
        dot = torch.dot(v1, v2)
        mag1 = torch.norm(v1)
        mag2 = torch.norm(v2)
        
        if mag1 > 0 and mag2 > 0:
            cos_angle = dot / (mag1 * mag2)
            return math.acos(torch.clamp(cos_angle, -1.0, 1.0).item())
        return 0.0
    
    def get_layer(
        self,
        text: str,
        lang: Optional[str] = None
    ) -> Tuple[int, str]:
        """
        Get the concentric layer for a text.
        
        Args:
            text: Text to classify
            lang: Language
        
        Returns:
            Tuple of (layer_index, layer_name)
        """
        # Encode
        embedding = self._encode_semantic(text, lang or self.config.default_lang)
        
        # Get distance from origin
        distance = torch.norm(embedding).item()
        
        # Get layer
        layer_idx = self.canonical_projector.get_layer(distance)
        layer_name = self.canonical_projector.get_layer_name(layer_idx)
        
        return layer_idx, layer_name
    
    def set_reference(
        self,
        token: str,
        vector: torch.Tensor,
        lang: str = "fr"
    ) -> None:
        """
        Set a reference vector for a token.
        
        Args:
            token: Token name
            vector: Reference vector
            lang: Language
        """
        if lang not in self._reference_cache:
            self._reference_cache[lang] = {}
        
        # Normalize
        vector = vector / torch.norm(vector)
        self._reference_cache[lang][token] = vector
        
        # Also set in canonical projector
        self.canonical_projector.set_reference_vector(token, vector)
    
    def get_reference(
        self,
        token: str,
        lang: str = "fr"
    ) -> Optional[torch.Tensor]:
        """
        Get a reference vector for a token.
        
        Args:
            token: Token name
            lang: Language
        
        Returns:
            Optional torch.Tensor: Reference vector or None
        """
        if lang in self._reference_cache:
            return self._reference_cache[lang].get(token)
        return None
    
    def initialize_references(self, lang: str = "fr") -> None:
        """
        Initialize standard reference vectors.
        
        Args:
            lang: Language
        """
        # Canonical references
        references = {
            "pou": torch.tensor([1.0, 0.0, 0.0]),
            "voir": torch.tensor([0.0, 1.0, 0.0]),
            "savoir": torch.tensor([0.0, 0.0, 1.0]),
            "pouvoir": torch.tensor([1.0, 1.0, 1.0]),
            "lumiere": torch.tensor([1.0, 1.0, 1.0]),
        }
        
        for token, vector in references.items():
            self.set_reference(token, vector, lang)
        
        logger.info(f"Initialized {len(references)} reference vectors for {lang}")
