"""
Semantic Embedder for Babel Space

Embeddings sémantiques réels produits par un modèle sentence-transformers
pré-entraîné (téléchargé depuis HuggingFace puis mis en cache localement).

Modèle par défaut : ``sentence-transformers/paraphrase-multilingual-mpnet-base-v2``
(768 dimensions, multilingue, donc utilisable en français). Il peut être
remplacé via le paramètre ``model_name`` ou la variable d'environnement
``APOURVOIR_EMBEDDING_MODEL``.

Les embeddings sont déterministes : inférence en mode ``eval`` sur CPU, sans
dropout ni échantillonnage, normalisés L2.
"""

import logging
import os
from functools import lru_cache
from typing import List, Optional, Union

import torch

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@lru_cache(maxsize=4)
def _load_model(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers est requis pour les embeddings sémantiques : "
            "pip install sentence-transformers"
        ) from e
    logger.info(f"Chargement du modèle d'embedding {model_name} sur {device}")
    model = SentenceTransformer(model_name, device=device)
    model.eval()
    return model


class SemanticEmbedder:
    """
    Encodeur sémantique basé sur sentence-transformers.

    Example:
        >>> embedder = SemanticEmbedder()
        >>> e = embedder.encode(["roi", "reine"])
        >>> e.shape
        torch.Size([2, 768])
    """

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        self.model_name = model_name or os.environ.get("APOURVOIR_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.device = device
        self._cache: dict = {}

    @property
    def model(self):
        return _load_model(self.model_name, self.device)

    @property
    def dim(self) -> int:
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return getter()

    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode un texte (-> [dim]) ou une liste de textes (-> [n, dim]).
        Les vecteurs sont normalisés L2.
        """
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        missing = [t for t in dict.fromkeys(batch) if t not in self._cache]
        if missing:
            with torch.no_grad():
                vectors = self.model.encode(missing, normalize_embeddings=True, convert_to_tensor=True)
            for text, vec in zip(missing, vectors.cpu().float()):
                self._cache[text] = vec
        out = torch.stack([self._cache[t] for t in batch])
        return out[0] if single else out

    def similarity(self, a: str, b: str) -> float:
        """Similarité cosinus entre deux textes."""
        ea, eb = self.encode([a, b])
        return float(torch.dot(ea, eb))
