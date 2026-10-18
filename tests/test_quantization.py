"""
Tests for the quantization module.

This module tests the quantization functionality including:
- UniformQuantizer
- AdaptiveQuantizer
- MatrixQuantizer
- VectorQuantizer
- Utility functions
"""

import numpy as np
import pytest
from src.core import SavoirVector, PouvoirMatrix, SavoirSpace, AlignmentEngine
from src.utils import (
    UniformQuantizer,
    AdaptiveQuantizer,
    MatrixQuantizer,
    VectorQuantizer,
    quantization_error,
    relative_quantization_error,
    quantization_snr,
    compression_ratio
)


class TestUniformQuantizer:
    """Tests for UniformQuantizer class."""
    
    def test_quantize_vector(self):
        """Test quantization of a simple vector."""
        quantizer = UniformQuantizer(n_bits=8)
        
        # Create a simple vector
        vec = np.array([0.0, 0.5, 1.0])
        
        # Quantize
        quantized, scale, min_val = quantizer.quantize(vec)
        
        # Check that quantized values are integers
        assert np.all(quantized == quantized.astype(int))
        
        # Check that values are in the correct range
        assert np.min(quantized) >= 0
        assert np.max(quantized) <= 255  # 2^8 - 1
    
    def test_quantize_dequantize(self):
        """Test quantization and dequantization round trip."""
        quantizer = UniformQuantizer(n_bits=8)
        
        # Create a test vector
        original = np.array([0.1, 0.5, 0.9])
        
        # Quantize and dequantize
        quantized, scale, min_val = quantizer.quantize(original)
        dequantized = quantizer.dequantize(quantized, scale, min_val)
        
        # Check that dequantized is close to original
        assert np.allclose(dequantized, original, atol=1e-2)
    
    def test_quantize_savoir(self):
        """Test quantization of SavoirVector."""
        space = SavoirSpace(dimension=3)
        quantizer = UniformQuantizer(n_bits=8)
        
        # Create a SavoirVector
        savoir = space.create_vector([1.0, 2.0, 3.0], label="test")
        
        # Quantize
        quantized_savoir, scale, min_val = quantizer.quantize_savoir(savoir)
        
        # Check that result is a SavoirVector
        assert isinstance(quantized_savoir, SavoirVector)
        assert quantized_savoir.label == "test"
    
    def test_quantize_pouvoir(self):
        """Test quantization of PouvoirMatrix."""
        quantizer = UniformQuantizer(n_bits=8)
        
        # Create a PouvoirMatrix
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        pouvoir = PouvoirMatrix(matrix, label="test")
        
        # Quantize
        quantized_pouvoir, scale, min_val = quantizer.quantize_pouvoir(pouvoir)
        
        # Check that result is a PouvoirMatrix
        assert isinstance(quantized_pouvoir, PouvoirMatrix)
        assert quantized_pouvoir.label == "test"
    
    def test_different_bit_depths(self):
        """Test quantization with different bit depths."""
        original = np.array([0.1, 0.5, 0.9])
        
        for n_bits in [4, 8, 12, 16]:
            quantizer = UniformQuantizer(n_bits=n_bits)
            quantized, scale, min_val = quantizer.quantize(original)
            dequantized = quantizer.dequantize(quantized, scale, min_val)
            
            # Check range of quantized values
            max_val = 2 ** n_bits - 1
            assert np.min(quantized) >= 0
            assert np.max(quantized) <= max_val
            
            # Higher bit depth should give better accuracy
            error = np.linalg.norm(original - dequantized)
            assert error < 0.5  # Reasonable error bound


class TestAdaptiveQuantizer:
    """Tests for AdaptiveQuantizer class."""
    
    def test_fit_and_quantize(self):
        """Test fitting and quantization with adaptive quantizer."""
        quantizer = AdaptiveQuantizer(n_bits=8)
        
        # Create training data
        train_data = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0]
        ])
        
        # Fit the quantizer
        quantizer.fit(train_data)
        
        # Quantize some data
        test_vec = np.array([1.5, 1.5, 1.5])
        quantized, ranges, _ = quantizer.quantize(test_vec)
        
        # Check that quantization worked
        assert quantized.shape == test_vec.shape
    
    def test_learned_ranges(self):
        """Test that ranges are learned correctly."""
        quantizer = AdaptiveQuantizer(n_bits=8)
        
        # Create training data with known ranges
        train_data = np.array([
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
            [2.0, 12.0, 22.0]
        ])
        
        quantizer.fit(train_data)
        
        # Check learned ranges
        assert quantizer.ranges[0]['min'] == pytest.approx(0.0)
        assert quantizer.ranges[0]['max'] == pytest.approx(2.0)
        assert quantizer.ranges[1]['min'] == pytest.approx(10.0)
        assert quantizer.ranges[1]['max'] == pytest.approx(12.0)
        assert quantizer.ranges[2]['min'] == pytest.approx(20.0)
        assert quantizer.ranges[2]['max'] == pytest.approx(22.0)
    
    def test_dequantize(self):
        """Test dequantization with adaptive quantizer."""
        quantizer = AdaptiveQuantizer(n_bits=8)
        
        # Create training data
        train_data = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ])
        
        quantizer.fit(train_data)
        
        # Quantize and dequantize
        test_vec = np.array([0.5, 0.5])
        quantized, ranges, _ = quantizer.quantize(test_vec)
        dequantized = quantizer.dequantize(quantized, ranges)
        
        # Check that dequantized is close to original
        assert np.allclose(dequantized, test_vec, atol=0.1)


class TestMatrixQuantizer:
    """Tests for MatrixQuantizer class."""
    
    def test_quantize_by_svd(self):
        """Test SVD-based matrix quantization."""
        quantizer = MatrixQuantizer(n_bits=8)
        
        # Create a test matrix
        matrix = np.random.randn(3, 3)
        pouvoir = PouvoirMatrix(matrix)
        
        # Quantize with SVD
        quantized_pouvoir, info = quantizer.quantize_by_svd(pouvoir, k=2)
        
        # Check that result is a PouvoirMatrix
        assert isinstance(quantized_pouvoir, PouvoirMatrix)
        
        # Check that info contains expected keys
        assert 'U_quant' in info
        assert 'S_quant' in info
        assert 'Vt_quant' in info
        assert 'k' in info
    
    def test_quantize_by_blocks(self):
        """Test block-based matrix quantization."""
        quantizer = MatrixQuantizer(n_bits=8)
        
        # Create a test matrix
        matrix = np.random.randn(4, 4)
        pouvoir = PouvoirMatrix(matrix)
        
        # Quantize by blocks
        quantized_pouvoir = quantizer.quantize_by_blocks(pouvoir, block_size=2)
        
        # Check that result is a PouvoirMatrix
        assert isinstance(quantized_pouvoir, PouvoirMatrix)
        
        # Check that shape is preserved
        assert quantized_pouvoir.shape == pouvoir.shape
    
    def test_quantize_by_rank(self):
        """Test rank-based matrix quantization."""
        quantizer = MatrixQuantizer(n_bits=8)
        
        # Create a test matrix
        matrix = np.random.randn(3, 3)
        pouvoir = PouvoirMatrix(matrix)
        
        # Quantize with rank reduction
        quantized_pouvoir, info = quantizer.quantize_by_rank(pouvoir, target_rank=2)
        
        # Check that result is a PouvoirMatrix
        assert isinstance(quantized_pouvoir, PouvoirMatrix)
        
        # Check that rank is reduced (approximately)
        original_rank = np.linalg.matrix_rank(matrix)
        quantized_rank = np.linalg.matrix_rank(quantized_pouvoir.matrix)
        assert quantized_rank <= 2


class TestVectorQuantizer:
    """Tests for VectorQuantizer class."""
    
    def test_fit(self):
        """Test fitting the vector quantizer."""
        space = SavoirSpace(dimension=2)
        quantizer = VectorQuantizer(n_clusters=4)
        
        # Create training vectors
        train_vectors = []
        for i in range(20):
            vec = space.create_vector(np.random.randn(2))
            train_vectors.append(vec)
        
        # Fit the quantizer
        quantizer.fit(train_vectors)
        
        # Check that codebook was created
        assert quantizer.codebook is not None
        assert quantizer.codebook.shape[0] == 4  # n_clusters
        assert quantizer.codebook.shape[1] == 2  # dimension
    
    def test_quantize(self):
        """Test vector quantization."""
        space = SavoirSpace(dimension=2)
        quantizer = VectorQuantizer(n_clusters=4)
        
        # Create training vectors
        train_vectors = []
        for i in range(20):
            vec = space.create_vector(np.random.randn(2))
            train_vectors.append(vec)
        
        quantizer.fit(train_vectors)
        
        # Create test vectors
        test_vectors = []
        for i in range(5):
            vec = space.create_vector(np.random.randn(2))
            test_vectors.append(vec)
        
        # Quantize
        indices, quantized_vectors = quantizer.quantize(test_vectors)
        
        # Check results
        assert len(indices) == 5
        assert len(quantized_vectors) == 5
        assert all(isinstance(v, SavoirVector) for v in quantized_vectors)
        
        # Check that indices are valid
        assert all(0 <= idx < 4 for idx in indices)
    
    def test_get_codebook(self):
        """Test getting the codebook."""
        space = SavoirSpace(dimension=2)
        quantizer = VectorQuantizer(n_clusters=4)
        
        # Create training vectors
        train_vectors = []
        for i in range(20):
            vec = space.create_vector(np.random.randn(2))
            train_vectors.append(vec)
        
        quantizer.fit(train_vectors)
        
        # Get codebook
        codebook = quantizer.get_codebook()
        
        # Check that codebook is a list of SavoirVectors
        assert len(codebook) == 4
        assert all(isinstance(v, SavoirVector) for v in codebook)
    
    def test_quantize_without_fit(self):
        """Test that quantization fails without fitting first."""
        space = SavoirSpace(dimension=2)
        quantizer = VectorQuantizer(n_clusters=4)
        
        # Create test vectors
        test_vectors = []
        for i in range(5):
            vec = space.create_vector(np.random.randn(2))
            test_vectors.append(vec)
        
        # Try to quantize without fitting
        with pytest.raises(ValueError):
            quantizer.quantize(test_vectors)


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_quantization_error(self):
        """Test quantization error calculation."""
        original = np.array([1.0, 2.0, 3.0])
        quantized = np.array([1.1, 2.1, 2.9])
        
        error = quantization_error(original, quantized)
        expected_error = np.linalg.norm(original - quantized)
        
        assert np.isclose(error, expected_error)
    
    def test_relative_quantization_error(self):
        """Test relative quantization error calculation."""
        original = np.array([1.0, 2.0, 3.0])
        quantized = np.array([1.1, 2.1, 2.9])
        
        rel_error = relative_quantization_error(original, quantized)
        
        # Should be error / norm(original)
        error = quantization_error(original, quantized)
        norm = np.linalg.norm(original)
        expected_rel_error = error / norm
        
        assert np.isclose(rel_error, expected_rel_error)
    
    def test_quantization_snr(self):
        """Test quantization SNR calculation."""
        original = np.array([1.0, 2.0, 3.0])
        quantized = np.array([1.0, 2.0, 3.0])  # Perfect quantization
        
        snr = quantization_snr(original, quantized)
        
        # SNR should be infinite for perfect quantization
        assert snr == float('inf')
        
        # Test with some error
        quantized = np.array([1.1, 2.0, 2.9])
        snr = quantization_snr(original, quantized)
        
        # SNR should be positive
        assert snr > 0
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        original = np.zeros((10, 10))
        quantized = np.zeros((10, 10), dtype=np.int8)
        
        ratio = compression_ratio(original, quantized, 64, 8)
        
        # Should be 64/8 = 8
        assert ratio == 8.0
        
        # Test with different sizes
        original = np.zeros((100, 100))
        quantized = np.zeros((100, 100), dtype=np.int16)
        
        ratio = compression_ratio(original, quantized, 32, 16)
        
        # Should be 32/16 = 2
        assert ratio == 2.0


class TestIntegration:
    """Integration tests for quantization with alignment."""
    
    def test_quantization_preserves_alignment(self):
        """Test that quantization preserves alignment properties."""
        space = SavoirSpace(dimension=2)
        engine = AlignmentEngine(space)
        
        # Create test vectors
        s1 = space.create_vector([1.0, 0.0])
        s2 = space.create_vector([0.0, 1.0])
        
        # Create target vectors
        t1 = space.create_vector([0.0, 1.0])
        t2 = space.create_vector([-1.0, 0.0])
        
        # Optimize alignment
        initial_pouvoir = PouvoirMatrix(np.eye(2))
        optimized_pouvoir = engine.optimize_alignment(
            [s1, s2], [t1, t2], initial_pouvoir=initial_pouvoir, max_iter=100
        )
        
        # Quantize the optimized power matrix
        quantizer = MatrixQuantizer(n_bits=8)
        quantized_pouvoir, _ = quantizer.quantize_by_svd(optimized_pouvoir, k=2)
        
        # Check that quantized matrix still provides good alignment
        original_score = engine.alignment_score(
            [s1, s2], optimized_pouvoir, [t1, t2]
        )
        quantized_score = engine.alignment_score(
            [s1, s2], quantized_pouvoir, [t1, t2]
        )
        
        # Quantized score should be close to original
        # Allow some degradation; absolute tolerance because the optimum can be exact (score ~ 0)
        assert quantized_score < original_score * 2 + 1e-3
    
    def test_quantization_dequantization_roundtrip(self):
        """Test that quantization and dequantization preserve alignment."""
        space = SavoirSpace(dimension=2)
        
        # Create test vectors
        s1 = space.create_vector([1.0, 2.0])
        s2 = space.create_vector([3.0, 4.0])
        
        # Create a power matrix
        matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
        pouvoir = PouvoirMatrix(matrix)
        
        # Quantize the vectors
        quantizer = UniformQuantizer(n_bits=8)
        q_s1, scale1, min1 = quantizer.quantize_savoir(s1)
        q_s2, scale2, min2 = quantizer.quantize_savoir(s2)
        
        # Dequantize
        dq_s1 = quantizer.dequantize(q_s1.vector, scale1, min1)
        dq_s2 = quantizer.dequantize(q_s2.vector, scale2, min2)
        
        # Apply power matrix to original and dequantized
        orig_transformed1 = pouvoir.apply(s1.vector)
        orig_transformed2 = pouvoir.apply(s2.vector)
        
        dq_transformed1 = pouvoir.apply(dq_s1)
        dq_transformed2 = pouvoir.apply(dq_s2)
        
        # Check that results are close
        assert np.allclose(orig_transformed1, dq_transformed1, atol=0.1)
        assert np.allclose(orig_transformed2, dq_transformed2, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
