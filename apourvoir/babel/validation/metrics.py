"""
Babel Metrics - Quality Metrics for Babel Space

Provides comprehensive metrics for evaluating Babel space encodings:
- Reconstruction error
- Semantic similarity preservation
- Compression ratio
- Right-angle checks between vectors (exploratory; no semantic property demonstrated)
- Layer consistency
- Multilingual alignment

Example:
    >>> metrics = BabelMetrics()
    >>> error = metrics.reconstruction_error(original, reconstructed)
    >>> similarity = metrics.semantic_similarity(embedding1, embedding2)
    >>> ratio = metrics.compression_ratio(original_size, compressed_size)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import math

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """Configuration for metrics."""
    # Tolerance for the right-angle check
    thales_tolerance: float = 0.01
    
    # Tolerance for semantic similarity
    similarity_tolerance: float = 0.05
    
    # Compression calculation
    original_bits: int = 32  # Bits per float in original
    
    # Layer boundaries
    layer_boundaries: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0])


class BabelMetrics:
    """
    Quality metrics for Babel space encodings.
    
    Provides comprehensive evaluation of:
    - Reconstruction accuracy
    - Semantic preservation
    - Compression effectiveness
    - Geometric consistency
    - Layer organization
    """
    
    def __init__(self, config: Optional[MetricConfig] = None):
        """
        Initialize the metrics calculator.
        
        Args:
            config: Configuration for metrics
        """
        self.config = config or MetricConfig()
        logger.info("BabelMetrics initialized")
    
    def reconstruction_error(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        metric: str = "cosine"
    ) -> float:
        """
        Calculate reconstruction error.
        
        Args:
            original: Original embedding
            reconstructed: Reconstructed embedding
            metric: Error metric ('cosine', 'euclidean', 'mse')
        
        Returns:
            float: Reconstruction error
        """
        if metric == "cosine":
            # Cosine distance (1 - cosine similarity)
            similarity = self._cosine_similarity(original, reconstructed)
            return 1.0 - similarity
        elif metric == "euclidean":
            return self._euclidean_distance(original, reconstructed)
        elif metric == "mse":
            return self._mse(original, reconstructed)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity."""
        dot = torch.dot(a, b)
        mag_a = torch.norm(a)
        mag_b = torch.norm(b)
        
        if mag_a > 0 and mag_b > 0:
            return torch.clamp(dot / (mag_a * mag_b), -1.0, 1.0).item()
        return 0.0
    
    def _euclidean_distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Euclidean distance."""
        return torch.norm(a - b).item()
    
    def _mse(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Mean squared error."""
        return torch.mean((a - b) ** 2).item()
    
    def semantic_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        method: str = "cosine"
    ) -> float:
        """
        Calculate semantic similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            method: Similarity method ('cosine', 'dot', 'euclidean')
        
        Returns:
            float: Similarity score
        """
        if method == "cosine":
            return self._cosine_similarity(embedding1, embedding2)
        elif method == "dot":
            return torch.dot(embedding1, embedding2).item()
        elif method == "euclidean":
            # Convert distance to similarity
            distance = self._euclidean_distance(embedding1, embedding2)
            return 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compression_ratio(
        self,
        original_size: int,
        compressed_size: int,
        original_bits: Optional[int] = None
    ) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original_size: Size of original data
            compressed_size: Size of compressed data
            original_bits: Bits per element in original (default from config)
        
        Returns:
            float: Compression ratio
        """
        original_bits = original_bits or self.config.original_bits
        
        if compressed_size == 0:
            return float('inf')
        
        return float(original_size) / float(compressed_size)
    
    def bits_per_element(
        self,
        original_size: int,
        compressed_size: int,
        num_elements: int
    ) -> float:
        """
        Calculate bits per element.
        
        Args:
            original_size: Size of original data in bytes
            compressed_size: Size of compressed data in bytes
            num_elements: Number of elements
        
        Returns:
            float: Bits per element
        """
        if num_elements == 0:
            return 0.0
        
        return (compressed_size * 8) / num_elements
    
    def validate_thales(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        v3: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Check whether the angle at the middle vectors is a right angle (Thales configuration).
        
        Args:
            v1: First vector
            v2: Vertex vector
            v3: Third vector
        
        Returns:
            Dict with validation results
        """
        # Vectors from v2
        vec1 = v1 - v2
        vec3 = v3 - v2
        
        # Dot product
        dot = torch.dot(vec1, vec3)
        
        # Magnitudes
        mag1 = torch.norm(vec1)
        mag3 = torch.norm(vec3)
        
        # Calculate angle
        if mag1 > 0 and mag3 > 0:
            cos_angle = dot / (mag1 * mag3)
            angle = math.acos(torch.clamp(cos_angle, -1.0, 1.0).item())
            angle_deg = math.degrees(angle)
            
            # Check if right angle
            is_right = abs(angle - math.pi/2) < self.config.thales_tolerance
            
            return {
                "valid": is_right,
                "angle_rad": angle,
                "angle_deg": angle_deg,
                "dot_product": dot.item(),
                "expected_angle": 90.0,
                "tolerance": math.degrees(self.config.thales_tolerance)
            }
        
        return {
            "valid": False,
            "error": "Zero magnitude vector"
        }
    
    def layer_consistency(
        self,
        embeddings: List[torch.Tensor],
        layer_boundaries: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Check layer consistency for a set of embeddings.
        
        Args:
            embeddings: List of embeddings
            layer_boundaries: Boundaries for layers
        
        Returns:
            Dict with layer distribution and consistency metrics
        """
        layer_boundaries = layer_boundaries or self.config.layer_boundaries
        
        # Calculate distances
        distances = [torch.norm(e).item() for e in embeddings]
        
        # Assign to layers
        layer_counts = {}
        for dist in distances:
            layer = self._get_layer(dist, layer_boundaries)
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        # Calculate metrics
        total = len(embeddings)
        entropy = self._calculate_entropy(layer_counts, total)
        
        return {
            "layer_counts": layer_counts,
            "total": total,
            "entropy": entropy,
            "num_layers_used": len(layer_counts),
            "max_distance": max(distances) if distances else 0,
            "min_distance": min(distances) if distances else 0,
            "avg_distance": sum(distances) / len(distances) if distances else 0
        }
    
    def _get_layer(self, distance: float, boundaries: List[float]) -> int:
        """Get layer index for a distance."""
        for i, boundary in enumerate(boundaries):
            if distance < boundary:
                return i
        return len(boundaries)
    
    def _calculate_entropy(
        self,
        counts: Dict[int, int],
        total: int
    ) -> float:
        """Calculate entropy of layer distribution."""
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def multilingual_alignment(
        self,
        translations: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Evaluate multilingual alignment.
        
        Checks if translations of the same concept have similar embeddings.
        
        Args:
            translations: Dict mapping language to embedding
        
        Returns:
            Dict with alignment metrics
        """
        languages = list(translations.keys())
        num_langs = len(languages)
        
        if num_langs < 2:
            return {"error": "Need at least 2 languages"}
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(num_langs):
            for j in range(i + 1, num_langs):
                lang1, lang2 = languages[i], languages[j]
                emb1, emb2 = translations[lang1], translations[lang2]
                sim = self.semantic_similarity(emb1, emb2, "cosine")
                similarities.append({
                    "lang1": lang1,
                    "lang2": lang2,
                    "similarity": sim
                })
        
        # Calculate statistics
        avg_sim = sum(s["similarity"] for s in similarities) / len(similarities)
        min_sim = min(s["similarity"] for s in similarities)
        max_sim = max(s["similarity"] for s in similarities)
        
        return {
            "pairwise_similarities": similarities,
            "average_similarity": avg_sim,
            "min_similarity": min_sim,
            "max_similarity": max_sim,
            "num_pairs": len(similarities)
        }
    
    def canonical_relationship_score(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        v3: torch.Tensor
    ) -> float:
        """
        Calculate a score for canonical relationship quality.
        
        Higher scores indicate better canonical relationships.
        
        Args:
            v1: First vector
            v2: Vertex vector
            v3: Third vector
        
        Returns:
            float: Relationship quality score [0, 1]
        """
        # Validate Thales
        thales = self.validate_thales(v1, v2, v3)
        
        if not thales["valid"]:
            # Not a right angle, but give partial credit based on how close
            angle_diff = abs(thales["angle_deg"] - 90.0)
            max_diff = 45.0  # Maximum acceptable difference
            thales_score = max(0, 1 - angle_diff / max_diff)
        else:
            thales_score = 1.0
        
        # Calculate distances
        d12 = self._euclidean_distance(v1, v2)
        d23 = self._euclidean_distance(v2, v3)
        d13 = self._euclidean_distance(v1, v3)
        
        # Normalize distances (assuming torus with R=2, r=1)
        max_dist = 3.0  # Maximum distance on torus
        d12_norm = d12 / max_dist
        d23_norm = d23 / max_dist
        d13_norm = d13 / max_dist
        
        # Distance score (lower is better)
        distance_score = 1.0 - (d12_norm + d23_norm + d13_norm) / 3.0
        
        # Combined score
        score = 0.7 * thales_score + 0.3 * distance_score
        
        return score
    
    def batch_metrics(
        self,
        originals: List[torch.Tensor],
        reconstructions: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Calculate metrics for a batch of embeddings.
        
        Args:
            originals: List of original embeddings
            reconstructions: List of reconstructed embeddings
        
        Returns:
            Dict with batch metrics
        """
        if len(originals) != len(reconstructions):
            return {"error": "Mismatched lengths"}
        
        # Calculate individual metrics
        cosine_errors = []
        euclidean_errors = []
        mse_errors = []
        
        for orig, recon in zip(originals, reconstructions):
            cosine_errors.append(self.reconstruction_error(orig, recon, "cosine"))
            euclidean_errors.append(self.reconstruction_error(orig, recon, "euclidean"))
            mse_errors.append(self.reconstruction_error(orig, recon, "mse"))
        
        return {
            "num_samples": len(originals),
            "cosine_error": {
                "mean": np.mean(cosine_errors),
                "std": np.std(cosine_errors),
                "min": np.min(cosine_errors),
                "max": np.max(cosine_errors)
            },
            "euclidean_error": {
                "mean": np.mean(euclidean_errors),
                "std": np.std(euclidean_errors),
                "min": np.min(euclidean_errors),
                "max": np.max(euclidean_errors)
            },
            "mse_error": {
                "mean": np.mean(mse_errors),
                "std": np.std(mse_errors),
                "min": np.min(mse_errors),
                "max": np.max(mse_errors)
            },
            "overall_quality": 1.0 - np.mean(cosine_errors)  # Higher is better
        }
    
    def compression_analysis(
        self,
        original_size: int,
        compressed_size: int,
        num_elements: int,
        original_bits: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive compression analysis.
        
        Args:
            original_size: Size of original data in bytes
            compressed_size: Size of compressed data in bytes
            num_elements: Number of elements
            original_bits: Bits per element in original
        
        Returns:
            Dict with compression analysis
        """
        original_bits = original_bits or self.config.original_bits
        
        ratio = self.compression_ratio(original_size, compressed_size, original_bits)
        bits_per_element = self.bits_per_element(original_size, compressed_size, num_elements)
        
        # Theoretical limits
        original_bits_per_element = original_bits
        
        return {
            "compression_ratio": ratio,
            "bits_per_element": bits_per_element,
            "original_bits_per_element": original_bits_per_element,
            "space_saved": original_size - compressed_size,
            "space_saved_percent": (1 - compressed_size / original_size) * 100 if original_size > 0 else 0,
            "efficiency": bits_per_element / original_bits_per_element if original_bits_per_element > 0 else 0
        }
