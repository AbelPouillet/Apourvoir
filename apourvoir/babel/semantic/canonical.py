"""
Canonical Projector for Babel Space

Projects embeddings onto the canonical 3D space (pou/voir/z(f)) for semantic encoding.

Key concepts:
- x-axis (pou): Reference of the possible and the goal
- y-axis (voir): Reference of perception
- z-axis (z(f)): Dynamic realization as a function of x and y

The canonical space uses:
- Torus topology (R=2.0, r=1.0) for periodic semantic relationships
- Right-angle checks between reference vectors (exploratory, not validated)
- Concentric layers for granular knowledge organization

Example:
    >>> projector = CanonicalProjector()
    >>> embedding_3d = projector.project_to_canonical(token_embedding)
    >>> theta, phi = projector.get_angles(embedding_3d)
    >>> quantized = projector.quantize_angles(theta, phi, n_bits=8)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CanonicalConfig:
    """Configuration for canonical projection."""
    # Torus parameters
    R: float = 2.0  # Major radius
    r: float = 1.0  # Minor radius
    
    # Quantization parameters
    n_bits: int = 8  # Number of bits for angle quantization
    max_layers: int = 4  # Number of concentric layers
    
    # Reference tokens (French)
    ref_pou: str = "pou"  # Reference for x-axis (possible/goal)
    ref_voir: str = "voir"  # Reference for y-axis (perception)
    ref_savoir: str = "savoir"  # Reference for semantic validation
    ref_lumiere: str = "lumiere"  # Reference for z=1 (perfect power)
    
    # Right-angle check
    thales_tolerance: float = 0.01  # Tolerance for right angle validation
    
    # Layer definitions (concentric)
    layer_names: List[str] = field(default_factory=lambda: [
        "fundamentals",
        "sub_concepts", 
        "actions",
        "techniques"
    ])


class CanonicalProjector:
    """
    Projects embeddings onto the canonical 3D space.
    
    The canonical space is defined by:
    - x-axis: pou (possible/but) - reference of the possible and the goal
    - y-axis: voir (to see) - reference of perception
    - z-axis: z(f) - dynamic realization as a function of x and y
    
    The space uses a torus topology to encapsulate semantic angular relationships.
    """
    
    def __init__(self, config: Optional[CanonicalConfig] = None):
        """
        Initialize the canonical projector.
        
        Args:
            config: Configuration for projection
        """
        self.config = config or CanonicalConfig()
        
        # Reference vectors (will be initialized)
        self.ref_vectors: Dict[str, torch.Tensor] = {}
        
        # Initialize reference vectors
        self._init_reference_vectors()
        
        logger.info(f"CanonicalProjector initialized: R={self.config.R}, r={self.config.r}")
    
    def _init_reference_vectors(self) -> None:
        """Initialize reference vectors for the canonical space."""
        # x-axis reference: pou (possible/goal)
        # This defines the reference for the x-axis
        self.ref_vectors[self.config.ref_pou] = torch.tensor([1.0, 0.0, 0.0])
        
        # y-axis reference: voir (perception)
        # This defines the reference for the y-axis
        self.ref_vectors[self.config.ref_voir] = torch.tensor([0.0, 1.0, 0.0])
        
        # z-axis reference: savoir (knowledge)
        # This is used for semantic validation
        self.ref_vectors[self.config.ref_savoir] = torch.tensor([0.0, 0.0, 1.0])
        
        # Perfect power reference: lumiere at z=1
        # This represents the perfect realization
        self.ref_vectors[self.config.ref_lumiere] = torch.tensor([1.0, 1.0, 1.0])
        self.ref_vectors[self.config.ref_lumiere] = self.ref_vectors[self.config.ref_lumiere] / torch.norm(self.ref_vectors[self.config.ref_lumiere])
    
    def set_reference_vector(self, token: str, vector: torch.Tensor) -> None:
        """
        Set a reference vector for a token.
        
        Args:
            token: Token name
            vector: Reference vector (will be normalized)
        """
        vector = vector / torch.norm(vector)
        self.ref_vectors[token] = vector
        logger.info(f"Set reference vector for '{token}': {vector.tolist()}")
    
    def set_axes(self, x_axis: torch.Tensor, y_axis: torch.Tensor, z_axis: torch.Tensor) -> None:
        """
        Define the 3D canonical subspace inside a high-dimensional embedding space.

        The three vectors (e.g. the real embeddings of "pou", "voir", "savoir")
        are orthonormalised with Gram-Schmidt. Projecting an embedding then
        returns its 3 coordinates in that basis. This keeps only the component
        of the embedding lying in that 3D subspace: it is a lossy projection.
        """
        basis = []
        for axis in (x_axis, y_axis, z_axis):
            v = axis.float().clone()
            for b in basis:
                v = v - torch.dot(v, b) * b
            norm = torch.norm(v)
            if norm < 1e-8:
                raise ValueError("Axes are linearly dependent; cannot build a 3D basis")
            basis.append(v / norm)
        self.basis = torch.stack(basis)  # [3, D]

    def fit_axes_from_embedder(self, embedder) -> None:
        """Build the canonical basis from the real embeddings of the reference words."""
        refs = [self.config.ref_pou, self.config.ref_voir, self.config.ref_savoir]
        x, y, z = embedder.encode(refs)
        self.set_axes(x, y, z)

    def to_canonical_coordinates(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Coordinates (..., 3) of an embedding in the canonical basis.

        3D inputs are returned unchanged. Higher-dimensional inputs require a
        basis set with :meth:`set_axes` / :meth:`fit_axes_from_embedder`;
        no silent truncation to the first 3 components is performed.
        """
        embedding = embedding.float()
        if embedding.shape[-1] == 3:
            return embedding
        basis = getattr(self, "basis", None)
        if basis is None:
            raise ValueError(
                f"Embedding of dimension {embedding.shape[-1]} needs a canonical basis: "
                "call set_axes() or fit_axes_from_embedder() first"
            )
        if basis.shape[1] != embedding.shape[-1]:
            raise ValueError(f"Basis dimension {basis.shape[1]} != embedding dimension {embedding.shape[-1]}")
        return embedding @ basis.T

    def project_to_canonical(
        self,
        embedding: torch.Tensor,
        method: str = "spherical"
    ) -> torch.Tensor:
        """
        Project an embedding onto the canonical 3D space.

        Args:
            embedding: Input embedding (..., D). D == 3 or D == basis dimension.
            method: 'linear' returns the 3D coordinates in the canonical basis;
                'spherical' / 'torus' additionally map the direction onto the
                torus surface (the norm is discarded).

        Returns:
            torch.Tensor: 3D canonical embedding (..., 3)
        """
        if method not in ("spherical", "linear", "torus"):
            raise ValueError(f"Unknown projection method: {method}")
        coords = self.to_canonical_coordinates(embedding)
        if method == "linear":
            return coords
        return self._project_to_torus(coords)

    def _project_to_torus(self, v: torch.Tensor) -> torch.Tensor:
        """
        Project 3D vectors (..., 3) onto the torus.

        Parametric equations:
        X = (R + r * cos(phi)) * cos(theta)
        Y = (R + r * cos(phi)) * sin(theta)
        Z = r * sin(phi)
        """
        _, theta, phi = self._cartesian_to_spherical(v)

        X = (self.config.R + self.config.r * torch.cos(phi)) * torch.cos(theta)
        Y = (self.config.R + self.config.r * torch.cos(phi)) * torch.sin(theta)
        Z = self.config.r * torch.sin(phi)

        return torch.stack([X, Y, Z], dim=-1)

    def _cartesian_to_spherical(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert cartesian (..., 3) to spherical coordinates."""
        r = torch.norm(v, dim=-1)
        theta = torch.atan2(v[..., 1], v[..., 0])  # [-\u03c0, \u03c0]
        theta = (theta + 2 * math.pi) % (2 * math.pi)  # Normalize to [0, 2\u03c0]
        phi = torch.acos(torch.clamp(v[..., 2] / (r + 1e-10), -1.0, 1.0))  # [0, \u03c0]
        return r, theta, phi
    
    def get_angles(self, v: torch.Tensor) -> Tuple[float, float]:
        """
        Get theta and phi angles from a 3D vector.
        
        Args:
            v: 3D vector
        
        Returns:
            Tuple of (theta, phi) in radians
        """
        r, theta, phi = self._cartesian_to_spherical(v)
        return theta.item(), phi.item()
    
    def quantize_angles(
        self,
        theta: float,
        phi: float,
        n_bits: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Quantize angles to n bits.
        
        Args:
            theta: Theta angle in radians [0, 2\u03c0]
            phi: Phi angle in radians [0, \u03c0]
            n_bits: Number of bits (default from config)
        
        Returns:
            Tuple of (theta_quantized, phi_quantized) as integers
        """
        n_bits = n_bits or self.config.n_bits
        
        # Normalize theta to [0, 2\u03c0]
        theta_norm = theta % (2 * math.pi)
        
        # Quantize theta
        theta_q = round(theta_norm / (2 * math.pi) * (2**n_bits)) % (2**n_bits)
        
        # Quantize phi
        phi_q = round(phi / math.pi * (2**n_bits)) % (2**n_bits)
        
        return int(theta_q), int(phi_q)
    
    def dequantize_angles(
        self,
        theta_q: int,
        phi_q: int,
        n_bits: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Dequantize angles from integer values.
        
        Args:
            theta_q: Quantized theta
            phi_q: Quantized phi
            n_bits: Number of bits used for quantization
        
        Returns:
            Tuple of (theta, phi) in radians
        """
        n_bits = n_bits or self.config.n_bits
        
        theta = (theta_q / (2**n_bits)) * 2 * math.pi
        phi = (phi_q / (2**n_bits)) * math.pi
        
        return theta, phi
    
    def quantize_to_canonical(
        self,
        embedding: torch.Tensor,
        n_bits: Optional[int] = None
    ) -> Tuple[Tuple[int, int], torch.Tensor]:
        """
        Full quantization pipeline: project + quantize.
        
        Args:
            embedding: Input embedding
            n_bits: Number of bits for quantization
        
        Returns:
            Tuple of (quantized_angles, canonical_vector)
        """
        # Project to canonical space
        canonical = self.project_to_canonical(embedding)
        
        # Get angles
        theta, phi = self.get_angles(canonical)
        
        # Quantize
        theta_q, phi_q = self.quantize_angles(theta, phi, n_bits)
        
        return (theta_q, phi_q), canonical
    
    def dequantize_from_canonical(
        self,
        theta_q: int,
        phi_q: int,
        n_bits: Optional[int] = None
    ) -> torch.Tensor:
        """
        Full dequantization pipeline: dequantize + reconstruct.
        
        Args:
            theta_q: Quantized theta
            phi_q: Quantized phi
            n_bits: Number of bits used for quantization
        
        Returns:
            Reconstructed 3D vector on the torus
        """
        # Dequantize angles
        theta, phi = self.dequantize_angles(theta_q, phi_q, n_bits)
        
        # Reconstruct vector on torus
        X = (self.config.R + self.config.r * math.cos(phi)) * math.cos(theta)
        Y = (self.config.R + self.config.r * math.cos(phi)) * math.sin(theta)
        Z = self.config.r * math.sin(phi)
        
        return torch.tensor([X, Y, Z])
    
    def validate_thales(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        v3: torch.Tensor
    ) -> bool:
        """
        Check whether the angle at the middle vectors is a right angle (Thales configuration).
        
        Checks if the angle at v2 is a right angle (90 degrees).
        
        Args:
            v1: First vector
            v2: Vertex vector
            v3: Third vector
        
        Returns:
            bool: True if angle at v2 is approximately 90 degrees
        """
        # Vectors from v2
        vec1 = v1 - v2
        vec3 = v3 - v2
        
        # Dot product
        dot = torch.dot(vec1, vec3)
        
        # Magnitudes
        mag1 = torch.norm(vec1)
        mag3 = torch.norm(vec3)
        
        # Cosine of angle
        if mag1 > 0 and mag3 > 0:
            cos_angle = dot / (mag1 * mag3)
            angle = math.acos(torch.clamp(cos_angle, -1.0, 1.0).item())
            
            # Check if angle is approximately 90 degrees
            return abs(angle - math.pi/2) < self.config.thales_tolerance
        
        return False
    
    def get_canonical_relationship(
        self,
        token1: str,
        token2: str,
        token3: str
    ) -> Dict[str, Any]:
        """
        Get the canonical relationship between three tokens via a right-angle check.
        
        Args:
            token1: First token
            token2: Second token (vertex)
            token3: Third token
        
        Returns:
            Dict with relationship information
        """
        # Get reference vectors
        v1 = self.ref_vectors.get(token1)
        v2 = self.ref_vectors.get(token2)
        v3 = self.ref_vectors.get(token3)
        
        if v1 is None or v2 is None or v3 is None:
            return {"valid": False, "error": "Missing reference vectors"}
        
        # Validate Thales
        is_thales = self.validate_thales(v1, v2, v3)
        
        # Calculate angles
        angle_v1v2 = self._angle_between(v1, v2)
        angle_v2v3 = self._angle_between(v2, v3)
        angle_v1v3 = self._angle_between(v1, v3)
        
        return {
            "valid": is_thales,
            "token1": token1,
            "token2": token2,
            "token3": token3,
            "angle_v1v2": math.degrees(angle_v1v2),
            "angle_v2v3": math.degrees(angle_v2v3),
            "angle_v1v3": math.degrees(angle_v1v3),
            "thales_valid": is_thales,
        }
    
    def _angle_between(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """Calculate angle between two vectors in radians."""
        dot = torch.dot(v1, v2)
        mag1 = torch.norm(v1)
        mag2 = torch.norm(v2)
        
        if mag1 > 0 and mag2 > 0:
            cos_angle = dot / (mag1 * mag2)
            return math.acos(torch.clamp(cos_angle, -1.0, 1.0).item())
        return 0.0
    
    def get_layer(self, distance: float) -> int:
        """
        Get the concentric layer based on distance from origin.
        
        Args:
            distance: Distance from origin
        
        Returns:
            int: Layer index (0 to max_layers-1)
        """
        # Normalize distance to [0, 1]
        max_distance = self.config.R + self.config.r
        normalized = distance / max_distance
        
        # Map to layer
        layer = int(normalized * self.config.max_layers)
        return min(layer, self.config.max_layers - 1)
    
    def get_layer_name(self, layer: int) -> str:
        """Get the name of a layer."""
        if layer < len(self.config.layer_names):
            return self.config.layer_names[layer]
        return f"layer_{layer}"
    
    def project_batch(
        self,
        embeddings: List[torch.Tensor],
        method: str = "spherical"
    ) -> torch.Tensor:
        """
        Project a batch of embeddings to canonical space.
        
        Args:
            embeddings: List of input embeddings
            method: Projection method
        
        Returns:
            torch.Tensor: Batch of 3D canonical embeddings
        """
        projected = [self.project_to_canonical(emb, method) for emb in embeddings]
        return torch.stack(projected)
