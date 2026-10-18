"""
Babel Algebra - Canonical Algebraic Operations

Provides algebraic operations for the Babel space:
- Composition: Combine embeddings
- Modification: Transform embeddings
- Inversion: Find opposites/negations
- Search: Find similar embeddings
- Fusion: Multi-dimensional fusion

All operations respect the canonical geometry and torus topology.

Example:
    >>> algebra = BabelAlgebra()
    >>> composed = algebra.compose(embedding1, embedding2)
    >>> modified = algebra.modify(embedding, transformation)
    >>> opposite = algebra.invert(embedding)
    >>> similar = algebra.search(query, candidates, k=5)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
import math
import heapq

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Transformation:
    """A transformation to apply to an embedding."""
    # Rotation parameters
    rotation_axis: Optional[torch.Tensor] = None
    rotation_angle: float = 0.0
    
    # Scaling parameters
    scale_factor: float = 1.0
    scale_axis: Optional[torch.Tensor] = None
    
    # Translation parameters
    translation: Optional[torch.Tensor] = None
    
    # Reflection parameters
    reflect_axis: Optional[torch.Tensor] = None


class BabelAlgebra:
    """
    Algebraic operations for Babel space.
    
    Provides operations that respect the canonical geometry:
    - Composition: Geometric composition of embeddings
    - Modification: Apply transformations
    - Inversion: Find opposites
    - Search: Similarity search
    - Fusion: Multi-vector fusion
    """
    
    def __init__(self):
        """Initialize the algebra."""
        logger.info("BabelAlgebra initialized")
    
    def compose(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        method: str = "geometric_mean"
    ) -> torch.Tensor:
        """
        Compose two embeddings in the canonical space.
        
        Args:
            a: First embedding
            b: Second embedding
            method: Composition method
                - 'geometric_mean': Normalized sum (default)
                - 'slerp': Spherical linear interpolation (midpoint)
                - 'product': Element-wise product
                - 'torus_sum': Sum on torus
        
        Returns:
            torch.Tensor: Composed embedding
        """
        if method == "geometric_mean":
            return self._geometric_mean(a, b)
        elif method == "slerp":
            return self._slerp(a, b, t=0.5)
        elif method == "product":
            return self._product(a, b)
        elif method == "torus_sum":
            return self._torus_sum(a, b)
        else:
            raise ValueError(f"Unknown composition method: {method}")
    
    def _geometric_mean(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Normalized sum (geometric mean on sphere)."""
        result = a + b
        if torch.norm(result) > 0:
            result = result / torch.norm(result)
        return result
    
    def _slerp(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        t: float = 0.5
    ) -> torch.Tensor:
        """
        Spherical linear interpolation.
        
        Args:
            a: First vector
            b: Second vector
            t: Interpolation parameter [0, 1]
        
        Returns:
            torch.Tensor: Interpolated vector
        """
        # Normalize inputs
        a_norm = a / torch.norm(a)
        b_norm = b / torch.norm(b)
        
        # Dot product
        dot = torch.dot(a_norm, b_norm)
        
        # Clamp for numerical stability
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # Angle between vectors
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        
        # Interpolation
        if sin_theta > 0:
            coeff_a = torch.sin((1 - t) * theta) / sin_theta
            coeff_b = torch.sin(t * theta) / sin_theta
            result = coeff_a * a_norm + coeff_b * b_norm
        else:
            # Vectors are colinear
            result = (1 - t) * a_norm + t * b_norm
        
        return result
    
    def _product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise product."""
        result = a * b
        if torch.norm(result) > 0:
            result = result / torch.norm(result)
        return result
    
    def _torus_sum(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Sum on the torus (respects periodic boundary conditions).
        
        This converts to angles, adds, and converts back.
        """
        # Get angles for both
        a_theta, a_phi = self._to_angles(a)
        b_theta, b_phi = self._to_angles(b)
        
        # Add angles
        sum_theta = (a_theta + b_theta) % (2 * math.pi)
        sum_phi = (a_phi + b_phi) % math.pi
        
        # Convert back to cartesian
        return self._from_angles(sum_theta, sum_phi)
    
    def _to_angles(self, v: torch.Tensor) -> Tuple[float, float]:
        """Convert vector to spherical angles."""
        r = torch.norm(v)
        theta = torch.atan2(v[1], v[0]).item()
        theta = (theta + 2 * math.pi) % (2 * math.pi)
        phi = torch.acos(torch.clamp(v[2] / (r.item() + 1e-10), -1.0, 1.0)).item()
        return theta, phi
    
    def _from_angles(self, theta: float, phi: float, R: float = 2.0, r: float = 1.0) -> torch.Tensor:
        """Convert angles to torus cartesian coordinates."""
        X = (R + r * math.cos(phi)) * math.cos(theta)
        Y = (R + r * math.cos(phi)) * math.sin(theta)
        Z = r * math.sin(phi)
        return torch.tensor([X, Y, Z])
    
    def modify(
        self,
        embedding: torch.Tensor,
        transformation: Transformation
    ) -> torch.Tensor:
        """
        Apply a transformation to an embedding.
        
        Args:
            embedding: Input embedding
            transformation: Transformation to apply
        
        Returns:
            torch.Tensor: Transformed embedding
        """
        result = embedding.clone()
        
        # Apply rotation
        if transformation.rotation_axis is not None:
            result = self._rotate(
                result,
                transformation.rotation_axis,
                transformation.rotation_angle
            )
        
        # Apply scaling
        if transformation.scale_factor != 1.0:
            result = self._scale(result, transformation.scale_factor, transformation.scale_axis)
        
        # Apply translation
        if transformation.translation is not None:
            result = self._translate(result, transformation.translation)
        
        # Apply reflection
        if transformation.reflect_axis is not None:
            result = self._reflect(result, transformation.reflect_axis)
        
        # Normalize
        if torch.norm(result) > 0:
            result = result / torch.norm(result)
        
        return result
    
    def _rotate(
        self,
        v: torch.Tensor,
        axis: torch.Tensor,
        angle: float
    ) -> torch.Tensor:
        """Rotate a vector around an axis."""
        # Normalize axis
        axis = axis / torch.norm(axis)
        
        # Rotation matrix (Rodrigues' formula)
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        
        # Cross product matrix
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Rotation matrix
        R = torch.eye(3) + sin_theta * K + (1 - cos_theta) * torch.mm(K, K)
        
        return torch.mm(R, v.unsqueeze(1)).squeeze(1)
    
    def _scale(
        self,
        v: torch.Tensor,
        factor: float,
        axis: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Scale a vector."""
        if axis is None:
            # Uniform scaling
            return v * factor
        else:
            # Non-uniform scaling along axis
            axis = axis / torch.norm(axis)
            projection = torch.dot(v, axis) * axis
            perpendicular = v - projection
            return projection * factor + perpendicular
    
    def _translate(
        self,
        v: torch.Tensor,
        translation: torch.Tensor
    ) -> torch.Tensor:
        """Translate a vector (on the sphere, this is a rotation)."""
        # On a sphere, translation is equivalent to rotation
        # For simplicity, we'll just add and renormalize
        return v + translation
    
    def _reflect(
        self,
        v: torch.Tensor,
        axis: torch.Tensor
    ) -> torch.Tensor:
        """Reflect a vector across a plane defined by the normal axis."""
        axis = axis / torch.norm(axis)
        projection = torch.dot(v, axis) * axis
        return 2 * projection - v
    
    def invert(self, embedding: torch.Tensor, method: str = "antipodal") -> torch.Tensor:
        """
        Find the inverse/opposite of an embedding.
        
        Args:
            embedding: Input embedding
            method: Inversion method
                - 'antipodal': Antipodal point on sphere
                - 'negate': Simple negation
                - 'torus_antipodal': Antipodal on torus
        
        Returns:
            torch.Tensor: Inverted embedding
        """
        if method == "antipodal":
            return -embedding
        elif method == "negate":
            return -embedding
        elif method == "torus_antipodal":
            theta, phi = self._to_angles(embedding)
            # Add pi to both angles
            inv_theta = (theta + math.pi) % (2 * math.pi)
            inv_phi = (phi + math.pi) % math.pi
            return self._from_angles(inv_theta, inv_phi)
        else:
            raise ValueError(f"Unknown inversion method: {method}")
    
    def search(
        self,
        query: torch.Tensor,
        candidates: List[torch.Tensor],
        k: int = 5,
        metric: str = "cosine"
    ) -> List[Tuple[int, float]]:
        """
        Search for similar embeddings.
        
        Args:
            query: Query embedding
            candidates: List of candidate embeddings
            k: Number of results to return
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
        
        Returns:
            List of (index, similarity_score) tuples
        """
        scores = []
        
        for i, candidate in enumerate(candidates):
            if metric == "cosine":
                score = self._cosine_similarity(query, candidate)
            elif metric == "euclidean":
                score = -self._euclidean_distance(query, candidate)  # Negative for min-heap
            elif metric == "dot":
                score = self._dot_product(query, candidate)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append((i, score))
        
        # Get top k
        if metric == "euclidean":
            # For euclidean, we want smallest distances (most negative scores)
            top_k = heapq.nsmallest(k, scores, key=lambda x: x[1])
        else:
            # For cosine and dot, we want highest scores
            top_k = heapq.nlargest(k, scores, key=lambda x: x[1])
        
        return top_k
    
    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity."""
        dot = torch.dot(a, b)
        mag_a = torch.norm(a)
        mag_b = torch.norm(b)
        
        if mag_a > 0 and mag_b > 0:
            return dot / (mag_a * mag_b)
        return 0.0
    
    def _euclidean_distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Euclidean distance."""
        return torch.norm(a - b).item()
    
    def _dot_product(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Dot product."""
        return torch.dot(a, b).item()
    
    def fuse(
        self,
        embeddings: List[torch.Tensor],
        weights: Optional[List[float]] = None,
        method: str = "weighted_sum"
    ) -> torch.Tensor:
        """
        Fuse multiple embeddings into one.
        
        Args:
            embeddings: List of embeddings to fuse
            weights: Optional weights for each embedding
            method: Fusion method
                - 'weighted_sum': Weighted sum
                - 'average': Simple average
                - 'geometric_mean': Normalized sum
        
        Returns:
            torch.Tensor: Fused embedding
        """
        if not embeddings:
            return torch.zeros(3)
        
        if weights is None:
            weights = [1.0] * len(embeddings)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        if method == "weighted_sum":
            result = sum(w * e for w, e in zip(weights, embeddings))
        elif method == "average":
            result = sum(embeddings) / len(embeddings)
        elif method == "geometric_mean":
            result = sum(e for e in embeddings)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        # Normalize
        if torch.norm(result) > 0:
            result = result / torch.norm(result)
        
        return result
    
    def validate_thales(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor
    ) -> bool:
        """
        Check whether the angle at the middle points is a right angle (Thales configuration).
        
        Checks if the angle at b is 90 degrees.
        
        Args:
            a: First point
            b: Vertex point
            c: Third point
        
        Returns:
            bool: True if angle at b is 90 degrees
        """
        # Vectors from b
        ba = a - b
        bc = c - b
        
        # Dot product
        dot = torch.dot(ba, bc)
        
        # Magnitudes
        mag_ba = torch.norm(ba)
        mag_bc = torch.norm(bc)
        
        # Check if perpendicular
        if mag_ba > 0 and mag_bc > 0:
            cos_angle = dot / (mag_ba * mag_bc)
            # Check if angle is approximately 90 degrees
            return abs(cos_angle.item()) < 0.01
        
        return False
    
    def create_transformation(
        self,
        rotation_axis: Optional[torch.Tensor] = None,
        rotation_angle: float = 0.0,
        scale_factor: float = 1.0,
        scale_axis: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None,
        reflect_axis: Optional[torch.Tensor] = None
    ) -> Transformation:
        """
        Create a transformation object.
        
        Args:
            rotation_axis: Axis for rotation
            rotation_angle: Angle for rotation in radians
            scale_factor: Scaling factor
            scale_axis: Axis for non-uniform scaling
            translation: Translation vector
            reflect_axis: Axis for reflection
        
        Returns:
            Transformation: The transformation object
        """
        return Transformation(
            rotation_axis=rotation_axis,
            rotation_angle=rotation_angle,
            scale_factor=scale_factor,
            scale_axis=scale_axis,
            translation=translation,
            reflect_axis=reflect_axis
        )
    
    def compose_transformations(
        self,
        t1: Transformation,
        t2: Transformation
    ) -> Transformation:
        """
        Compose two transformations.
        
        Note: This is a simplified composition that combines transformations.
        For full accuracy, transformations should be represented as matrices.
        
        Args:
            t1: First transformation
            t2: Second transformation
        
        Returns:
            Transformation: Composed transformation
        """
        # For now, just combine the parameters
        # In a full implementation, this would use matrix multiplication
        return Transformation(
            rotation_axis=t2.rotation_axis or t1.rotation_axis,
            rotation_angle=t1.rotation_angle + t2.rotation_angle,
            scale_factor=t1.scale_factor * t2.scale_factor,
            scale_axis=t2.scale_axis or t1.scale_axis,
            translation=(t1.translation + t2.translation) if t1.translation is not None and t2.translation is not None else t2.translation or t1.translation,
            reflect_axis=t2.reflect_axis or t1.reflect_axis
        )
