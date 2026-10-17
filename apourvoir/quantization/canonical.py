"""
Canonical Quantization for Apourvoir

Implements the mathematical framework for canonical quantization:
- 3D torus topology
- Angular quantization (theta, phi)
- Thales' theorem for semantic relationships
- Concentric layers for granularization
"""

import logging
from typing import Tuple, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


def normaliser_vecteur(v: torch.Tensor) -> torch.Tensor:
    """
    Normalise un vecteur.
    
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
    Convertit un vecteur cartésien en coordonnées sphériques.
    
    Retourne (r, theta, phi) où:
    - r: norme du vecteur
    - theta: angle dans le plan x-y [0, 2π]
    - phi: angle par rapport à z [0, π]
    
    Args:
        v: Vecteur cartésien de forme (..., 3)
    
    Returns:
        Tuple de (r, theta, phi)
    """
    r = torch.norm(v, dim=-1)
    
    # Calcul de theta (angle dans le plan x-y)
    theta = torch.atan2(v[..., 1], v[..., 0])  # [-π, π]
    theta = (theta + 2 * np.pi) % (2 * np.pi)  # Normaliser à [0, 2π]
    
    # Calcul de phi (angle par rapport à z)
    phi = torch.acos(torch.clamp(v[..., 2] / (r + 1e-10), -1.0, 1.0))  # [0, π]
    
    return r, theta, phi


def projeter_sur_tore(v: torch.Tensor, R: float = 2.0, r: float = 0.5) -> torch.Tensor:
    """
    Projette un vecteur sur un tore 3D.
    
    Équations paramétriques:
    X = (R + r * cos(phi)) * cos(theta)
    Y = (R + r * cos(phi)) * sin(theta)
    Z = r * sin(phi)
    
    Args:
        v: Vecteur à projeter
        R: Rayon majeur du tore
        r: Rayon mineur du tore
    
    Returns:
        Vecteur projeté sur le tore
    """
    _, theta, phi = coordonnées_sphériques(v)
    
    X = (R + r * torch.cos(phi)) * torch.cos(theta)
    Y = (R + r * torch.cos(phi)) * torch.sin(theta)
    Z = r * torch.sin(phi)
    
    return torch.stack([X, Y, Z], dim=-1)


def quantifier_angles(theta: torch.Tensor, phi: torch.Tensor, n_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantifie les angles sur n_bits.
    
    - theta ∈ [0, 2π] → [0, 2^N-1]
    - phi ∈ [0, π] → [0, 2^N-1]
    
    Args:
        theta: Angle theta
        phi: Angle phi
        n_bits: Nombre de bits pour la quantification
    
    Returns:
        Tuple de (theta_quantifié, phi_quantifié)
    """
    # Normaliser theta à [0, 2π]
    theta_norm = theta % (2 * np.pi)
    
    # Quantifier theta
    theta_q = torch.round(theta_norm / (2 * np.pi) * (2**n_bits)).long() % (2**n_bits)
    
    # Quantifier phi
    phi_q = torch.round(phi / np.pi * (2**n_bits)).long() % (2**n_bits)
    
    return theta_q, phi_q


def déquantifier_angles(theta_q: torch.Tensor, phi_q: torch.Tensor, n_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Déquantifie les angles.
    
    Args:
        theta_q: Angle theta quantifié
        phi_q: Angle phi quantifié
        n_bits: Nombre de bits utilisé pour la quantification
    
    Returns:
        Tuple de (theta, phi) déquantifiés
    """
    theta = (theta_q.float() / (2**n_bits)) * 2 * np.pi
    phi = (phi_q.float() / (2**n_bits)) * np.pi
    return theta, phi


def quantifier_tensor_canonique(
    tensor: torch.Tensor,
    n_bits: int = 8,
    R: float = 2.0,
    r: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """
    Quantifie un tenseur complet avec la méthode canonique.
    
    Args:
        tensor: Tenseur à quantifier
        n_bits: Nombre de bits pour la quantification
        R: Rayon majeur du tore
        r: Rayon mineur du tore
    
    Returns:
        Tuple de (tenseur_quantifié, métadonnées)
    """
    device = tensor.device
    shape = tensor.shape
    
    # Reshape pour traiter chaque élément comme un vecteur 3D
    # Si le tenseur est 2D, on ajoute une dimension
    if len(shape) == 2:
        tensor_2d = tensor.view(-1, 1, shape[-1])
    else:
        tensor_2d = tensor.view(-1, shape[-1])
    
    # Normaliser
    tensor_norm = normaliser_vecteur(tensor_2d)
    
    # Projeter sur tore
    tensor_projeté = projeter_sur_tore(tensor_norm, R, r)
    
    # Convertir en coordonnées sphériques
    _, theta, phi = coordonnées_sphériques(tensor_projeté)
    
    # Quantifier les angles
    theta_q, phi_q = quantifier_angles(theta, phi, n_bits)
    
    # Stocker les angles quantifiés
    if n_bits <= 8:
        tensor_quant = torch.stack([theta_q, phi_q], dim=-1).to(torch.int8)
    elif n_bits <= 16:
        tensor_quant = torch.stack([theta_q, phi_q], dim=-1).to(torch.int16)
    else:
        tensor_quant = torch.stack([theta_q, phi_q], dim=-1).to(torch.int32)
    
    # Métadonnées
    metadata = {
        "shape_original": shape,
        "n_bits": n_bits,
        "R": R,
        "r": r,
        "compression_ratio": (tensor.numel() * 4) / (tensor_quant.numel() * (n_bits // 8)),
        "device": str(device),
    }
    
    return tensor_quant, metadata


def déquantifier_tensor_canonique(
    tensor_quant: torch.Tensor,
    metadata: dict,
) -> torch.Tensor:
    """
    Déquantifie un tenseur.
    
    Args:
        tensor_quant: Tenseur quantifié
        metadata: Métadonnées de la quantification
    
    Returns:
        Tenseur déquantifié
    """
    n_bits = metadata["n_bits"]
    R = metadata["R"]
    r = metadata["r"]
    
    # Extraire theta_q et phi_q
    theta_q = tensor_quant[..., 0].long()
    phi_q = tensor_quant[..., 1].long()
    
    # Déquantifier
    theta, phi = déquantifier_angles(theta_q, phi_q, n_bits)
    
    # Reconstruire les vecteurs
    X = (R + r * torch.cos(phi)) * torch.cos(theta)
    Y = (R + r * torch.cos(phi)) * torch.sin(theta)
    Z = r * torch.sin(phi)
    
    tensor_reconstruit = torch.stack([X, Y, Z], dim=-1)
    
    # Reshape pour correspondre à la forme originale
    original_shape = metadata["shape_original"]
    if len(original_shape) == 2:
        tensor_reconstruit = tensor_reconstruit.view(original_shape)
    else:
        tensor_reconstruit = tensor_reconstruit.view(*original_shape)
    
    return tensor_reconstruit


class CanonicalQuantizer:
    """
    Quantiseur canonique pour les modèles MoE.
    
    Implémente la quantification canonique avec:
    - Topologie de tore concentrique
    - Quantification angulaire
    - Granularisation en couches
    """
    
    def __init__(
        self,
        n_bits: int = 8,
        R: float = 2.0,
        r: float = 0.5,
        num_layers: int = 4,
    ):
        """
        Initialise le quantiseur canonique.
        
        Args:
            n_bits: Nombre de bits pour la quantification
            R: Rayon majeur du tore
            r: Rayon mineur du tore
            num_layers: Nombre de couches concentriques
        """
        self.n_bits = n_bits
        self.R = R
        self.r = r
        self.num_layers = num_layers
        
        # Définition des couches (1=fundamentaux, 2=sub-concepts, 3=actions, 4=techniques)
        self.layer_definitions = {
            1: {"name": "fundamentaux", "weight": 1.0},
            2: {"name": "sous-concepts", "weight": 0.8},
            3: {"name": "actions", "weight": 0.6},
            4: {"name": "techniques", "weight": 0.4},
        }
    
    def quantize(self, tensor: torch.Tensor, layer: int = 1) -> Tuple[torch.Tensor, dict]:
        """
        Quantifie un tenseur avec la méthode canonique.
        
        Args:
            tensor: Tenseur à quantifier
            layer: Couche pour la granularisation (1-4)
        
        Returns:
            Tuple de (tenseur_quantifié, métadonnées)
        """
        # Adapter le nombre de bits selon la couche
        layer_weight = self.layer_definitions.get(layer, {}).get("weight", 1.0)
        effective_bits = max(4, int(self.n_bits * layer_weight))
        
        # Adapter le rayon mineur selon la couche
        effective_r = self.r * (layer / self.num_layers)
        
        return quantifier_tensor_canonique(
            tensor,
            n_bits=effective_bits,
            R=self.R,
            r=effective_r,
        )
    
    def dequantize(self, tensor_quant: torch.Tensor, metadata: dict) -> torch.Tensor:
        """
        Déquantifie un tenseur.
        
        Args:
            tensor_quant: Tenseur quantifié
            metadata: Métadonnées de la quantification
        
        Returns:
            Tenseur déquantifié
        """
        return déquantifier_tensor_canonique(tensor_quant, metadata)
    
    def get_compression_ratio(self, tensor: torch.Tensor) -> float:
        """
        Calcule le ratio de compression.
        
        Args:
            tensor: Tenseur à compresser
        
        Returns:
            Ratio de compression
        """
        # Taille originale (FP32: 4 bytes par élément)
        original_size = tensor.numel() * 4
        
        # Taille quantifiée (2 angles * n_bits/8 bytes)
        quant_size = tensor.numel() * 2 * (self.n_bits // 8)
        
        return original_size / quant_size if quant_size > 0 else 1.0


class ThalesTriangle:
    """
    Implémente le théorème de Thales pour les relations sémantiques.
    
    Utilisé pour valider les relations canoniques entre les concepts.
    """
    
    def __init__(self, R: float = 2.0):
        """
        Initialise le triangle de Thales.
        
        Args:
            R: Rayon du cercle circonscrit
        """
        self.R = R
    
    def calculate_angles(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> float:
        """
        Calcule les angles d'un triangle en utilisant le théorème de Thales.
        
        Args:
            A: Point A du triangle
            B: Point B du triangle
            C: Point C du triangle
        
        Returns:
            Angle au point A en degrés
        """
        # Vecteurs
        AB = B - A
        AC = C - A
        
        # Produit scalaire
        dot = torch.dot(AB, AC)
        
        # Normes
        norm_AB = torch.norm(AB)
        norm_AC = torch.norm(AC)
        
        # Cosinus de l'angle
        cos_angle = dot / (norm_AB * norm_AC + 1e-10)
        
        # Angle en radians puis en degrés
        angle_rad = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        angle_deg = torch.degrees(angle_rad)
        
        return angle_deg.item()
    
    def is_right_triangle(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> bool:
        """
        Vérifie si le triangle est rectangle (théorème de Thales).
        
        Args:
            A: Point A
            B: Point B
            C: Point C
        
        Returns:
            True si le triangle est rectangle
        """
        # Calculer les angles
        angle_A = self.calculate_angles(A, B, C)
        angle_B = self.calculate_angles(B, A, C)
        angle_C = self.calculate_angles(C, A, B)
        
        # Vérifier si un angle est proche de 90 degrés
        return any(abs(angle - 90) < 1 for angle in [angle_A, angle_B, angle_C])


if __name__ == "__main__":
    # Test des fonctions de quantification canonique
    print("Canonical Quantization Test")
    print("=" * 80)
    
    # Créer un tenseur de test
    tensor = torch.randn(10, 10, 3)  # 10x10 tenseur 3D
    
    # Quantifier
    quant_tensor, metadata = quantifier_tensor_canonique(tensor, n_bits=8)
    print(f"Original shape: {tensor.shape}")
    print(f"Quantized shape: {quant_tensor.shape}")
    print(f"Compression ratio: {metadata['compression_ratio']:.1f}x")
    
    # Déquantifier
    dequant_tensor = déquantifier_tensor_canonique(quant_tensor, metadata)
    print(f"Dequantized shape: {dequant_tensor.shape}")
    
    # Calculer la différence
    diff = torch.abs(tensor - dequant_tensor).max().item()
    print(f"Max difference: {diff:.6f}")
    
    # Test du triangle de Thales
    print("\nThales Triangle Test")
    A = torch.tensor([1.0, 0.0, 0.0])
    B = torch.tensor([0.0, 1.0, 0.0])
    C = torch.tensor([0.0, 0.0, 1.0])
    
    thales = ThalesTriangle()
    angle = thales.calculate_angles(A, B, C)
    print(f"Angle at A: {angle:.2f} degrees")
    print(f"Is right triangle: {thales.is_right_triangle(A, B, C)}")
