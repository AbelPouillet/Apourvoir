"""
Quantization utilities for the Apourvoir framework.

This module provides various quantization methods for knowledge vectors
and power matrices to enable efficient storage, transmission, and computation.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from ..core.alignment import SavoirVector, PouvoirMatrix, SavoirSpace


class Quantizer:
    """Base class for quantization operations."""
    
    def __init__(self, n_bits: int = 8):
        """
        Initialize the quantizer.
        
        Args:
            n_bits: Number of bits for quantization
        """
        self.n_bits = n_bits
    
    def quantize(self, data: Union[np.ndarray, SavoirVector, PouvoirMatrix]) -> Tuple:
        """
        Quantize the input data.
        
        Args:
            data: Data to quantize (vector, matrix, or SavoirVector/PouvoirMatrix)
            
        Returns:
            Tuple containing (quantized_data, scale, min_value) or similar
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def dequantize(self, quantized_data: np.ndarray, scale: float, min_value: float = 0.0) -> np.ndarray:
        """
        Dequantize the data.
        
        Args:
            quantized_data: Quantized data
            scale: Scale factor used during quantization
            min_value: Minimum value used during quantization
            
        Returns:
            Dequantized data
        """
        raise NotImplementedError("Subclasses must implement this method")


class UniformQuantizer(Quantizer):
    """Uniform quantization with fixed bit depth."""
    
    def __init__(self, n_bits: int = 8):
        super().__init__(n_bits)
    
    def quantize(
        self, 
        data: Union[np.ndarray, SavoirVector, PouvoirMatrix],
        range_min: Optional[float] = None,
        range_max: Optional[float] = None
    ) -> Tuple[np.ndarray, float, float]:
        """
        Uniform quantization of data.
        
        Args:
            data: Data to quantize
            range_min: Minimum value for quantization range (optional)
            range_max: Maximum value for quantization range (optional)
            
        Returns:
            Tuple of (quantized_data, scale, min_value)
        """
        # Extract numpy array from input
        if isinstance(data, SavoirVector):
            vec = data.vector
        elif isinstance(data, PouvoirMatrix):
            vec = data.matrix
        else:
            vec = np.array(data)
        
        # Determine range
        if range_min is None:
            range_min = np.min(vec)
        if range_max is None:
            range_max = np.max(vec)
        
        # Handle case where range is zero
        if range_max - range_min < 1e-10:
            return np.zeros_like(vec, dtype=np.int16), 1.0, range_min
        
        # Calculate scale
        scale = (2 ** self.n_bits - 1) / (range_max - range_min)
        
        # Quantize
        quantized = np.round((vec - range_min) * scale).astype(np.int16)
        
        return quantized, scale, range_min
    
    def dequantize(
        self,
        quantized_data: np.ndarray,
        scale: float,
        min_value: float = 0.0
    ) -> np.ndarray:
        """Dequantize uniformly quantized data."""
        return quantized_data / scale + min_value
    
    def quantize_savoir(
        self,
        savoir: SavoirVector,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None
    ) -> Tuple[SavoirVector, float, float]:
        """
        Quantize a SavoirVector.
        
        Args:
            savoir: Knowledge vector to quantize
            range_min: Minimum value for quantization range
            range_max: Maximum value for quantization range
            
        Returns:
            Tuple of (quantized_savoir, scale, min_value)
        """
        quantized, scale, min_value = self.quantize(savoir, range_min, range_max)
        return SavoirVector(quantized, savoir.label, savoir.metadata), scale, min_value
    
    def quantize_pouvoir(
        self,
        pouvoir: PouvoirMatrix,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None
    ) -> Tuple[PouvoirMatrix, float, float]:
        """
        Quantize a PouvoirMatrix.
        
        Args:
            pouvoir: Power matrix to quantize
            range_min: Minimum value for quantization range
            range_max: Maximum value for quantization range
            
        Returns:
            Tuple of (quantized_pouvoir, scale, min_value)
        """
        quantized, scale, min_value = self.quantize(pouvoir, range_min, range_max)
        return PouvoirMatrix(quantized, pouvoir.label, pouvoir.metadata), scale, min_value


class AdaptiveQuantizer(Quantizer):
    """Adaptive quantizer that learns optimal ranges from data."""
    
    def __init__(self, n_bits: int = 8):
        super().__init__(n_bits)
        self.ranges = {}
    
    def fit(self, data: Union[np.ndarray, List[SavoirVector]]) -> None:
        """
        Learn the optimal quantization ranges from data.
        
        Args:
            data: Training data to learn ranges from
        """
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], SavoirVector):
            # List of SavoirVectors
            all_vectors = np.array([s.vector for s in data])
        else:
            all_vectors = np.array(data)
        
        # Learn range for each dimension
        for i in range(all_vectors.shape[1]):
            self.ranges[i] = {
                'min': np.min(all_vectors[:, i]),
                'max': np.max(all_vectors[:, i])
            }
    
    def quantize(
        self,
        data: Union[np.ndarray, SavoirVector, PouvoirMatrix],
        range_min: Optional[float] = None,
        range_max: Optional[float] = None
    ) -> Tuple[np.ndarray, dict, float]:
        """
        Adaptive quantization using learned ranges.
        
        Args:
            data: Data to quantize
            range_min: Override minimum value (optional)
            range_max: Override maximum value (optional)
            
        Returns:
            Tuple of (quantized_data, ranges, global_min)
        """
        # Extract numpy array
        if isinstance(data, SavoirVector):
            vec = data.vector
        elif isinstance(data, PouvoirMatrix):
            vec = data.matrix
        else:
            vec = np.array(data)
        
        quantized = np.zeros_like(vec)
        ranges_used = {}
        
        for i in range(vec.shape[0] if len(vec.shape) == 1 else vec.shape[0]):
            for j in range(vec.shape[1] if len(vec.shape) > 1 else 1):
                if len(vec.shape) > 1:
                    value = vec[i, j]
                    idx = (i, j)
                else:
                    value = vec[i]
                    idx = i
                
                # Get range for this dimension/position
                if len(vec.shape) > 1:
                    # For matrices, use column-based ranges
                    range_info = self.ranges.get(j, {'min': np.min(vec[:, j]), 'max': np.max(vec[:, j])})
                else:
                    # For vectors, use dimension-based ranges
                    range_info = self.ranges.get(i, {'min': np.min(vec), 'max': np.max(vec)})
                
                # Override with provided range if specified
                if range_min is not None:
                    range_info['min'] = range_min
                if range_max is not None:
                    range_info['max'] = range_max
                
                ranges_used[idx] = range_info
                
                # Quantize this value
                if range_info['max'] - range_info['min'] < 1e-10:
                    quantized_value = 0
                else:
                    scale = (2 ** self.n_bits - 1) / (range_info['max'] - range_info['min'])
                    quantized_value = np.round((value - range_info['min']) * scale)
                
                if len(vec.shape) > 1:
                    quantized[i, j] = quantized_value
                else:
                    quantized[i] = quantized_value
        
        return quantized.astype(np.int16), ranges_used, range_min or 0.0
    
    def dequantize(
        self,
        quantized_data: np.ndarray,
        ranges: dict,
        default_min: float = 0.0
    ) -> np.ndarray:
        """Dequantize adaptively quantized data."""
        dequantized = np.zeros_like(quantized_data, dtype=float)
        
        for i in range(quantized_data.shape[0]):
            for j in range(quantized_data.shape[1] if len(quantized_data.shape) > 1 else 1):
                if len(quantized_data.shape) > 1:
                    q_val = quantized_data[i, j]
                    idx = (i, j)
                else:
                    q_val = quantized_data[i]
                    idx = i
                
                range_info = ranges.get(idx, {'min': default_min, 'max': default_min + 1})
                
                if range_info['max'] - range_info['min'] < 1e-10:
                    dequantized_value = range_info['min']
                else:
                    scale = (2 ** self.n_bits - 1) / (range_info['max'] - range_info['min'])
                    dequantized_value = q_val / scale + range_info['min']
                
                if len(quantized_data.shape) > 1:
                    dequantized[i, j] = dequantized_value
                else:
                    dequantized[i] = dequantized_value
        
        return dequantized


class MatrixQuantizer:
    """Specialized quantizer for power matrices."""
    
    def __init__(self, n_bits: int = 8):
        self.n_bits = n_bits
        self.quantizer = UniformQuantizer(n_bits)
    
    def quantize_by_svd(
        self,
        pouvoir: PouvoirMatrix,
        k: Optional[int] = None
    ) -> Tuple[PouvoirMatrix, dict]:
        """
        Quantize a power matrix using SVD decomposition.
        
        Args:
            pouvoir: Power matrix to quantize
            k: Number of singular values to keep (optional)
            
        Returns:
            Tuple of (quantized_pouvoir, decomposition_info)
        """
        matrix = pouvoir.matrix
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        
        # Determine k
        if k is None:
            k = min(len(S), min(matrix.shape))
        else:
            k = min(k, len(S), min(matrix.shape))
        
        # Truncate
        U = U[:, :k]
        S = S[:k]
        Vt = Vt[:k, :]
        
        # Quantize each component
        U_quant, U_scale, U_min = self.quantizer.quantize(U)
        S_quant, S_scale, S_min = self.quantizer.quantize(S)
        Vt_quant, Vt_scale, Vt_min = self.quantizer.quantize(Vt)
        
        # Dequantize for reconstruction
        U_dequant = self.quantizer.dequantize(U_quant, U_scale, U_min)
        S_dequant = self.quantizer.dequantize(S_quant, S_scale, S_min)
        Vt_dequant = self.quantizer.dequantize(Vt_quant, Vt_scale, Vt_min)
        
        # Reconstruct matrix
        quantized_matrix = U_dequant @ np.diag(S_dequant) @ Vt_dequant
        
        # Store decomposition info
        decomposition_info = {
            'U_quant': U_quant,
            'U_scale': U_scale,
            'U_min': U_min,
            'S_quant': S_quant,
            'S_scale': S_scale,
            'S_min': S_min,
            'Vt_quant': Vt_quant,
            'Vt_scale': Vt_scale,
            'Vt_min': Vt_min,
            'k': k,
            'original_rank': min(matrix.shape),
            'compressed_rank': k
        }
        
        return PouvoirMatrix(quantized_matrix, pouvoir.label, pouvoir.metadata), decomposition_info
    
    def quantize_by_blocks(
        self,
        pouvoir: PouvoirMatrix,
        block_size: int = 4
    ) -> PouvoirMatrix:
        """
        Quantize a power matrix by blocks.
        
        Args:
            pouvoir: Power matrix to quantize
            block_size: Size of each block
            
        Returns:
            Quantized power matrix
        """
        matrix = pouvoir.matrix
        m, n = matrix.shape
        quantized_matrix = np.zeros_like(matrix)
        
        for i in range(0, m, block_size):
            for j in range(0, n, block_size):
                block = matrix[i:i+block_size, j:j+block_size]
                
                # Quantize the block
                quant_block, scale, min_val = self.quantizer.quantize(block)
                dequant_block = self.quantizer.dequantize(quant_block, scale, min_val)
                
                quantized_matrix[i:i+block_size, j:j+block_size] = dequant_block
        
        return PouvoirMatrix(quantized_matrix, pouvoir.label, pouvoir.metadata)
    
    def quantize_by_rank(
        self,
        pouvoir: PouvoirMatrix,
        target_rank: int
    ) -> Tuple[PouvoirMatrix, dict]:
        """
        Quantize a power matrix by reducing its rank.
        
        Args:
            pouvoir: Power matrix to quantize
            target_rank: Target rank for the matrix
            
        Returns:
            Tuple of (quantized_pouvoir, rank_info)
        """
        return self.quantize_by_svd(pouvoir, k=target_rank)


class VectorQuantizer:
    """Vector quantization using clustering."""
    
    def __init__(self, n_clusters: int = 256):
        """
        Initialize the vector quantizer.
        
        Args:
            n_clusters: Number of clusters for quantization
        """
        self.n_clusters = n_clusters
        self.codebook = None
        self.kmeans = None
    
    def fit(self, vectors: List[SavoirVector]) -> None:
        """
        Learn the codebook from a set of vectors.
        
        Args:
            vectors: List of SavoirVectors to learn from
        """
        from sklearn.cluster import KMeans
        
        # Extract numpy arrays
        X = np.array([v.vector for v in vectors])
        
        # Fit K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        self.kmeans.fit(X)
        self.codebook = self.kmeans.cluster_centers_
    
    def quantize(self, vectors: List[SavoirVector]) -> Tuple[np.ndarray, List[SavoirVector]]:
        """
        Quantize a list of vectors using the learned codebook.
        
        Args:
            vectors: List of SavoirVectors to quantize
            
        Returns:
            Tuple of (indices, quantized_vectors)
        """
        if self.codebook is None:
            raise ValueError("Codebook not learned. Call fit() first.")
        
        # Extract numpy arrays
        X = np.array([v.vector for v in vectors])
        
        # Predict cluster indices
        indices = self.kmeans.predict(X)
        
        # Create quantized vectors
        quantized_vectors = []
        for i, idx in enumerate(indices):
            quantized_vec = self.codebook[idx]
            quantized_vectors.append(SavoirVector(
                quantized_vec,
                vectors[i].label,
                vectors[i].metadata
            ))
        
        return indices, quantized_vectors
    
    def get_codebook(self) -> List[SavoirVector]:
        """Get the codebook as a list of SavoirVectors."""
        if self.codebook is None:
            return []
        
        return [SavoirVector(vec, f"codebook_{i}") for i, vec in enumerate(self.codebook)]


# Utility functions

def quantization_error(original: np.ndarray, quantized: np.ndarray) -> float:
    """Calculate the quantization error."""
    return np.linalg.norm(original - quantized)


def relative_quantization_error(original: np.ndarray, quantized: np.ndarray) -> float:
    """Calculate the relative quantization error."""
    error = quantization_error(original, quantized)
    norm = np.linalg.norm(original)
    return error / norm if norm > 0 else 0.0


def quantization_snr(original: np.ndarray, quantized: np.ndarray) -> float:
    """Calculate the signal-to-noise ratio of quantization."""
    error = quantization_error(original, quantized)
    signal_power = np.linalg.norm(original) ** 2
    noise_power = error ** 2
    
    if noise_power < 1e-10:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def compression_ratio(
    original: np.ndarray,
    quantized: np.ndarray,
    original_bits: int = 64,
    quantized_bits: int = 8
) -> float:
    """Calculate the compression ratio."""
    original_size = original.size * original_bits
    quantized_size = quantized.size * quantized_bits
    return original_size / quantized_size


__all__ = [
    'Quantizer',
    'UniformQuantizer',
    'AdaptiveQuantizer',
    'MatrixQuantizer',
    'VectorQuantizer',
    'quantization_error',
    'relative_quantization_error',
    'quantization_snr',
    'compression_ratio'
]
