"""
Canonical Quantization for Apourvoir

Quantisation "canonique" sphérique par triplets, applicable à un tenseur de
n'importe quelle dimension.

Principe (vérifiable, aucune affirmation non mesurée) :

1. Le tenseur est aplati en lignes ``(N, D)``. Chaque ligne est complétée par
   des zéros jusqu'à un multiple de 3, puis découpée en ``G = ceil(D / 3)``
   triplets ``(x, y, z)``.
2. Chaque triplet est converti en coordonnées sphériques ``(r, theta, phi)`` :
   ``r = ||v||``, ``theta = atan2(y, x) ∈ [0, 2π)``, ``phi = acos(z / r) ∈ [0, π]``.
3. ``theta`` est quantifié uniformément sur ``2^n`` niveaux (circulaire),
   ``phi`` sur ``2^n`` niveaux couvrant ``[0, π]`` bornes incluses,
   et ``r`` sur ``2^n`` niveaux après normalisation par le rayon maximal de
   la ligne (une échelle float32 par ligne). Par défaut ``r`` est encodé en
   ``sqrt(r / r_max)``, ce qui réduit l'erreur sur les distributions à queue
   lourde (voir ``scripts/validate_quantization.py`` pour la mesure).
4. Les codes sont stockés en ``uint8`` (n_bits = 8), ``int16`` (n_bits = 16)
   ou bit-packés dans un buffer ``uint8`` (autres valeurs de n_bits).

Coût de stockage exact : ``3 * G * n_bits`` bits par ligne + 32 bits d'échelle
par ligne. À 8 bits, cela fait ~8 bits par valeur, soit le même budget qu'un
INT8 : le ratio par rapport au FP32 est donc au plus ~4x (et ~2x par rapport
au BF16 natif des modèles Mistral). Le ratio réellement obtenu est calculé à
partir des octets effectivement stockés et renvoyé dans les métadonnées.

Pourquoi pas une projection PCA en 3D ? Conserver 3 composantes d'un vecteur
de dimension 4096 jette presque toute l'information : la reconstruction est
alors inutilisable (mesure dans ``scripts/validate_quantization.py``).

L'ancienne implémentation (projection sur tore puis 2 angles) ne pouvait
reconstruire que des vecteurs 3D de norme fixe ; elle ignorait toutes les
dimensions au-delà de 3 et la norme. Elle a été remplacée.
"""

import logging
import math
from typing import Tuple, Optional, Dict, Any

import torch
import numpy as np

logger = logging.getLogger(__name__)

_TWO_PI = 2 * math.pi

# Nombre d'éléments traités par bloc, pour borner la mémoire sur les gros tenseurs.
_CHUNK_ELEMENTS = 1 << 22


def normaliser_vecteur(v: torch.Tensor) -> torch.Tensor:
    """
    Normalise un vecteur sur sa dernière dimension.

    Args:
        v: Vecteur à normaliser

    Returns:
        Vecteur normalisé
    """
    norm = torch.norm(v, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-10)
    return v / norm


def coordonnées_sphériques(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convertit des vecteurs cartésiens 3D en coordonnées sphériques.

    Retourne (r, theta, phi) où:
    - r: norme du vecteur
    - theta: angle dans le plan x-y [0, 2π)
    - phi: angle par rapport à z [0, π]

    Args:
        v: Vecteurs cartésiens de forme (..., 3)

    Returns:
        Tuple de (r, theta, phi)
    """
    if v.shape[-1] != 3:
        raise ValueError(f"coordonnées_sphériques attend une dernière dimension de 3, reçu {v.shape[-1]}")
    r = torch.norm(v, dim=-1)
    theta = torch.atan2(v[..., 1], v[..., 0]) % _TWO_PI
    phi = torch.acos(torch.clamp(v[..., 2] / torch.clamp(r, min=1e-30), -1.0, 1.0))
    return r, theta, phi


def cartésiennes(r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Inverse de :func:`coordonnées_sphériques`.

    Returns:
        Vecteurs cartésiens de forme (..., 3)
    """
    sin_phi = torch.sin(phi)
    return torch.stack(
        [r * sin_phi * torch.cos(theta), r * sin_phi * torch.sin(theta), r * torch.cos(phi)],
        dim=-1,
    )


def projeter_sur_tore(v: torch.Tensor, R: float = 2.0, r: float = 0.5) -> torch.Tensor:
    """
    Projette des vecteurs 3D sur un tore (visualisation uniquement).

    Équations paramétriques:
    X = (R + r * cos(phi)) * cos(theta)
    Y = (R + r * cos(phi)) * sin(theta)
    Z = r * sin(phi)

    Attention : cette projection perd la norme du vecteur et n'est pas
    utilisée par la quantisation (elle n'est pas inversible).

    Args:
        v: Vecteurs de forme (..., 3)
        R: Rayon majeur du tore
        r: Rayon mineur du tore

    Returns:
        Points sur le tore, forme (..., 3)
    """
    _, theta, phi = coordonnées_sphériques(v)

    X = (R + r * torch.cos(phi)) * torch.cos(theta)
    Y = (R + r * torch.cos(phi)) * torch.sin(theta)
    Z = r * torch.sin(phi)

    return torch.stack([X, Y, Z], dim=-1)


def quantifier_angles(theta: torch.Tensor, phi: torch.Tensor, n_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantifie les angles sur n_bits.

    - theta ∈ [0, 2π) → {0, …, 2^N - 1} (circulaire : 2π ≡ 0)
    - phi ∈ [0, π]    → {0, …, 2^N - 1} (bornes 0 et π représentées exactement)

    Args:
        theta: Angle theta
        phi: Angle phi
        n_bits: Nombre de bits pour la quantification

    Returns:
        Tuple de (theta_quantifié, phi_quantifié) en int64
    """
    levels = 2 ** n_bits
    theta_q = torch.round((theta % _TWO_PI) / _TWO_PI * levels).long() % levels
    phi_q = torch.round(torch.clamp(phi, 0.0, math.pi) / math.pi * (levels - 1)).long()
    return theta_q, phi_q


def déquantifier_angles(theta_q: torch.Tensor, phi_q: torch.Tensor, n_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Déquantifie les angles produits par :func:`quantifier_angles`.

    Args:
        theta_q: Angle theta quantifié
        phi_q: Angle phi quantifié
        n_bits: Nombre de bits utilisé pour la quantification

    Returns:
        Tuple de (theta, phi) déquantifiés
    """
    levels = 2 ** n_bits
    theta = theta_q.float() / levels * _TWO_PI
    phi = phi_q.float() / (levels - 1) * math.pi
    return theta, phi


# ============================================================================
# BIT PACKING
# ============================================================================

def _pack_bits(codes: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Bit-packe des entiers non signés de n_bits dans un buffer uint8 (big-endian)."""
    if 8 % n_bits == 0:
        # Chemin rapide (2 ou 4 bits) : plusieurs codes entiers par octet.
        per_byte = 8 // n_bits
        flat = codes.reshape(-1).to(torch.uint8)
        pad = (-flat.numel()) % per_byte
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8, device=flat.device)])
        flat = flat.view(-1, per_byte)
        packed = torch.zeros(flat.shape[0], dtype=torch.uint8, device=flat.device)
        for j in range(per_byte):
            packed |= flat[:, j] << (8 - n_bits * (j + 1))
        return packed
    flat = codes.reshape(-1).to(torch.int32)
    shifts = torch.arange(n_bits - 1, -1, -1, dtype=torch.int32, device=flat.device)
    bits = ((flat.unsqueeze(1) >> shifts) & 1).to(torch.uint8).reshape(-1)
    pad = (-bits.numel()) % 8
    if pad:
        bits = torch.cat([bits, torch.zeros(pad, dtype=torch.uint8, device=bits.device)])
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=bits.device)
    return (bits.view(-1, 8) * weights).sum(dim=1, dtype=torch.int32).to(torch.uint8)


def _unpack_bits(packed: torch.Tensor, n_bits: int, count: int) -> torch.Tensor:
    """Inverse de :func:`_pack_bits`. Retourne ``count`` entiers en int64."""
    if 8 % n_bits == 0:
        per_byte = 8 // n_bits
        mask = (1 << n_bits) - 1
        parts = [(packed >> (8 - n_bits * (j + 1))) & mask for j in range(per_byte)]
        return torch.stack(parts, dim=1).reshape(-1)[:count].to(torch.int32)
    shifts = torch.arange(7, -1, -1, dtype=torch.int32, device=packed.device)
    bits = ((packed.to(torch.int32).unsqueeze(1) >> shifts) & 1).reshape(-1)[: count * n_bits]
    weights = 2 ** torch.arange(n_bits - 1, -1, -1, dtype=torch.int64, device=packed.device)
    return (bits.view(count, n_bits).long() * weights).sum(dim=1)


def _store_codes(codes: torch.Tensor, n_bits: int) -> torch.Tensor:
    if n_bits == 8:
        return codes.to(torch.uint8)
    if n_bits == 16:
        # int16 signé : décalage de 2^15 pour tenir dans la plage.
        return (codes - 2 ** 15).to(torch.int16)
    return _pack_bits(codes, n_bits)


def _load_codes(stored: torch.Tensor, n_bits: int, count: int) -> torch.Tensor:
    if n_bits == 8:
        return stored.reshape(-1).to(torch.int32)
    if n_bits == 16:
        return stored.reshape(-1).long() + 2 ** 15
    return _unpack_bits(stored, n_bits, count)


def taille_octets(tensor_quant: torch.Tensor, metadata: Dict[str, Any]) -> int:
    """
    Nombre d'octets réellement stockés (codes + échelles par ligne).

    C'est la base de tous les ratios de compression rapportés.
    """
    scales = metadata["scales"]
    return tensor_quant.numel() * tensor_quant.element_size() + scales.numel() * scales.element_size()


# ============================================================================
# QUANTISATION D'UN TENSEUR
# ============================================================================

def quantifier_tensor_canonique(
    tensor: torch.Tensor,
    n_bits: int = 8,
    R: Optional[float] = None,
    r: Optional[float] = None,
    r_mode: str = "sqrt",
) -> Tuple[torch.Tensor, dict]:
    """
    Quantifie un tenseur de forme quelconque avec la méthode sphérique par triplets.

    Args:
        tensor: Tenseur à quantifier (n'importe quelle forme ; la dernière
            dimension est découpée en triplets)
        n_bits: Bits par composante (r, theta, phi), entre 2 et 16
        R, r: Ignorés. Conservés pour compatibilité avec l'ancienne API à tore.
        r_mode: Encodage du rayon : "sqrt" (défaut) ou "lin"

    Returns:
        Tuple de (codes_stockés, métadonnées). Les métadonnées contiennent
        ``scales`` (float32, une par ligne), ``stored_bytes`` et
        ``compression_ratio`` (= octets FP32 / octets stockés, calculé sur
        les octets réels).
    """
    if not 2 <= n_bits <= 16:
        raise ValueError(f"n_bits doit être entre 2 et 16, reçu {n_bits}")
    if r_mode not in ("sqrt", "lin"):
        raise ValueError(f"r_mode inconnu: {r_mode}")
    if R is not None or r is not None:
        logger.debug("Paramètres R/r ignorés : la quantisation n'utilise plus la projection sur tore.")

    shape = tuple(tensor.shape)
    original_dtype = tensor.dtype
    x = tensor.detach()
    if x.dim() == 0:
        x = x.reshape(1, 1)
    elif x.dim() == 1:
        x = x.reshape(1, -1)
    else:
        x = x.reshape(-1, shape[-1])

    n_rows, dim = x.shape
    groups = -(-dim // 3)
    levels = 2 ** n_bits

    codes = torch.empty((n_rows, groups, 3), dtype=torch.int32, device=x.device)
    scales = torch.empty(n_rows, dtype=torch.float32, device=x.device)

    rows_per_chunk = max(1, _CHUNK_ELEMENTS // max(dim, 1))
    for start in range(0, n_rows, rows_per_chunk):
        end = min(start + rows_per_chunk, n_rows)
        chunk = x[start:end].float()
        if groups * 3 != dim:
            chunk = torch.nn.functional.pad(chunk, (0, groups * 3 - dim))
        v = chunk.view(end - start, groups, 3)

        radius, theta, phi = coordonnées_sphériques(v)
        theta_q, phi_q = quantifier_angles(theta, phi, n_bits)

        scale = radius.amax(dim=1)
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        rel = radius / safe_scale.unsqueeze(1)
        if r_mode == "sqrt":
            rel = rel.sqrt()
        r_q = torch.round(rel * (levels - 1)).long()

        codes[start:end, :, 0] = r_q.to(torch.int32)
        codes[start:end, :, 1] = theta_q.to(torch.int32)
        codes[start:end, :, 2] = phi_q.to(torch.int32)
        scales[start:end] = scale

    stored = _store_codes(codes, n_bits)
    if n_bits in (8, 16):
        stored = stored.view(n_rows, groups, 3)

    metadata = {
        "method": "spherical_triplets",
        "shape_original": shape,
        "dtype_original": str(original_dtype).replace("torch.", ""),
        "n_bits": n_bits,
        "r_mode": r_mode,
        "n_rows": n_rows,
        "dim": dim,
        "groups": groups,
        "scales": scales,
        "device": str(tensor.device),
    }
    stored_bytes = taille_octets(stored, metadata)
    metadata["stored_bytes"] = stored_bytes
    metadata["compression_ratio"] = (tensor.numel() * 4) / stored_bytes if stored_bytes else 1.0
    metadata["bits_per_value"] = stored_bytes * 8 / max(tensor.numel(), 1)

    return stored, metadata


def déquantifier_tensor_canonique(
    tensor_quant: torch.Tensor,
    metadata: dict,
) -> torch.Tensor:
    """
    Reconstruit le tenseur (float32) à partir des codes et métadonnées.

    Args:
        tensor_quant: Codes produits par :func:`quantifier_tensor_canonique`
        metadata: Métadonnées de la quantification

    Returns:
        Tenseur float32 de forme ``metadata["shape_original"]``
    """
    if metadata.get("method") != "spherical_triplets":
        raise ValueError(
            "Métadonnées incompatibles : ce tenseur n'a pas été produit par la "
            "quantisation sphérique par triplets (ancien format à tore non supporté)."
        )

    n_bits = metadata["n_bits"]
    n_rows, dim, groups = metadata["n_rows"], metadata["dim"], metadata["groups"]
    levels = 2 ** n_bits
    scales = metadata["scales"].to(tensor_quant.device)

    codes = _load_codes(tensor_quant, n_bits, n_rows * groups * 3).view(n_rows, groups, 3)
    out = torch.empty((n_rows, dim), dtype=torch.float32, device=tensor_quant.device)

    rows_per_chunk = max(1, _CHUNK_ELEMENTS // max(dim, 1))
    for start in range(0, n_rows, rows_per_chunk):
        end = min(start + rows_per_chunk, n_rows)
        c = codes[start:end]
        rel = c[..., 0].float() / (levels - 1)
        if metadata["r_mode"] == "sqrt":
            rel = rel * rel
        radius = rel * scales[start:end].unsqueeze(1)
        theta, phi = déquantifier_angles(c[..., 1], c[..., 2], n_bits)
        v = cartésiennes(radius, theta, phi).view(end - start, groups * 3)
        out[start:end] = v[:, :dim]

    return out.view(metadata["shape_original"])


def erreur_relative(original: torch.Tensor, reconstruit: torch.Tensor) -> float:
    """Erreur relative de Frobenius ``||x - x̂|| / ||x||``."""
    original = original.float()
    norm = torch.linalg.vector_norm(original).item()
    if norm == 0:
        return 0.0
    return torch.linalg.vector_norm(original - reconstruit.float()).item() / norm


class CanonicalQuantizer:
    """
    Quantiseur canonique avec un nombre de bits par "couche" de granularité.

    Les couches 1..4 réduisent le nombre de bits (8, 6, 4, 4 avec n_bits=8).
    C'est une simple politique de bits ; aucun gain de qualité lié à cette
    hiérarchie n'a été mesuré.
    """

    def __init__(
        self,
        n_bits: int = 8,
        R: float = 2.0,
        r: float = 0.5,
        num_layers: int = 4,
        r_mode: str = "sqrt",
    ):
        """
        Initialise le quantiseur canonique.

        Args:
            n_bits: Nombre de bits pour la quantification
            R, r: Conservés pour compatibilité (non utilisés par la quantisation)
            num_layers: Nombre de couches
            r_mode: Encodage du rayon ("sqrt" ou "lin")
        """
        self.n_bits = n_bits
        self.R = R
        self.r = r
        self.num_layers = num_layers
        self.r_mode = r_mode

        self.layer_definitions = {
            1: {"name": "fundamentaux", "weight": 1.0},
            2: {"name": "sous-concepts", "weight": 0.8},
            3: {"name": "actions", "weight": 0.6},
            4: {"name": "techniques", "weight": 0.4},
        }

    def bits_for_layer(self, layer: int = 1) -> int:
        """Nombre de bits effectif pour une couche."""
        layer_weight = self.layer_definitions.get(layer, {}).get("weight", 1.0)
        return max(4, int(self.n_bits * layer_weight))

    def quantize(self, tensor: torch.Tensor, layer: int = 1) -> Tuple[torch.Tensor, dict]:
        """
        Quantifie un tenseur avec la méthode canonique.

        Args:
            tensor: Tenseur à quantifier
            layer: Couche pour la granularisation (1-4)

        Returns:
            Tuple de (tenseur_quantifié, métadonnées)
        """
        return quantifier_tensor_canonique(tensor, n_bits=self.bits_for_layer(layer), r_mode=self.r_mode)

    def dequantize(self, tensor_quant: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Déquantifie un tenseur."""
        return déquantifier_tensor_canonique(tensor_quant, metadata)

    def get_compression_ratio(self, tensor: torch.Tensor, layer: int = 1) -> float:
        """
        Ratio de compression par rapport au FP32, calculé en quantifiant
        réellement le tenseur et en comptant les octets stockés.
        """
        _, metadata = self.quantize(tensor, layer)
        return metadata["compression_ratio"]


class ThalesTriangle:
    """
    Calcul d'angles dans un triangle et test d'angle droit.

    Le théorème de Thalès (un triangle inscrit dans un cercle dont un côté
    est un diamètre est rectangle) est un résultat géométrique classique.
    Cette classe se contente de mesurer des angles ; aucune propriété
    sémantique n'est démontrée ni mesurée par ce test.
    """

    def __init__(self, R: float = 2.0):
        """
        Args:
            R: Rayon du cercle circonscrit (informatif)
        """
        self.R = R

    def calculate_angles(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> float:
        """
        Angle au sommet A du triangle ABC, en degrés.
        """
        AB = B - A
        AC = C - A
        cos_angle = torch.dot(AB, AC) / (torch.norm(AB) * torch.norm(AC) + 1e-10)
        angle_rad = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        return math.degrees(angle_rad.item())

    def is_right_triangle(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, tolerance_deg: float = 1.0) -> bool:
        """
        True si l'un des angles du triangle est à ``tolerance_deg`` près de 90°.
        """
        angles = [
            self.calculate_angles(A, B, C),
            self.calculate_angles(B, A, C),
            self.calculate_angles(C, A, B),
        ]
        return any(abs(angle - 90) < tolerance_deg for angle in angles)


if __name__ == "__main__":
    # Démonstration sur un tenseur synthétique (pas une mesure sur un modèle réel)
    gen = torch.Generator().manual_seed(0)
    tensor = torch.randn(100, 4096, generator=gen)

    quant, meta = quantifier_tensor_canonique(tensor, n_bits=8)
    recon = déquantifier_tensor_canonique(quant, meta)
    print(f"Forme originale : {tuple(tensor.shape)}")
    print(f"Octets stockés  : {meta['stored_bytes']}")
    print(f"Ratio vs FP32   : {meta['compression_ratio']:.3f}x")
    print(f"Erreur relative : {erreur_relative(tensor, recon):.5f}")
