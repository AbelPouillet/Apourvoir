"""
Tests for the alignment module.

This module tests the core alignment functionality including:
- SavoirVector operations
- PouvoirMatrix operations
- SavoirSpace functionality
- AlignmentEngine operations
"""

import numpy as np
import pytest
from apourvoir.core import SavoirVector, PouvoirMatrix, SavoirSpace, AlignmentEngine
from apourvoir.core import create_alignment_tensor, calculate_alignment_gradient


class TestSavoirVector:
    """Tests for SavoirVector class."""
    
    def test_creation(self):
        """Test vector creation."""
        vec = SavoirVector(np.array([1.0, 2.0, 3.0]), label="test")
        assert vec.dimension == 3
        assert vec.label == "test"
        assert np.array_equal(vec.vector, np.array([1.0, 2.0, 3.0]))
    
    def test_norm(self):
        """Test vector norm calculation."""
        vec = SavoirVector(np.array([3.0, 4.0]))
        assert np.isclose(vec.norm(), 5.0)
    
    def test_normalize(self):
        """Test vector normalization."""
        vec = SavoirVector(np.array([3.0, 4.0]))
        normalized = vec.normalize()
        assert np.isclose(normalized.norm(), 1.0)
        assert np.allclose(normalized.vector, np.array([0.6, 0.8]))
    
    def test_inner_product(self):
        """Test inner product calculation."""
        vec1 = SavoirVector(np.array([1.0, 0.0]))
        vec2 = SavoirVector(np.array([0.0, 1.0]))
        assert vec1.inner_product(vec2) == 0.0
        
        vec3 = SavoirVector(np.array([2.0, 3.0]))
        vec4 = SavoirVector(np.array([4.0, 5.0]))
        assert vec3.inner_product(vec4) == 23.0
    
    def test_angle(self):
        """Test angle calculation between vectors."""
        vec1 = SavoirVector(np.array([1.0, 0.0]))
        vec2 = SavoirVector(np.array([0.0, 1.0]))
        assert np.isclose(vec1.angle(vec2), np.pi / 2)
        
        vec3 = SavoirVector(np.array([1.0, 0.0]))
        vec4 = SavoirVector(np.array([1.0, 0.0]))
        assert np.isclose(vec3.angle(vec4), 0.0)
    
    def test_operations(self):
        """Test vector arithmetic operations."""
        vec1 = SavoirVector(np.array([1.0, 2.0]))
        vec2 = SavoirVector(np.array([3.0, 4.0]))
        
        # Addition
        result = vec1 + vec2
        assert np.array_equal(result.vector, np.array([4.0, 6.0]))
        
        # Subtraction
        result = vec2 - vec1
        assert np.array_equal(result.vector, np.array([2.0, 2.0]))
        
        # Scalar multiplication
        result = vec1 * 2.0
        assert np.array_equal(result.vector, np.array([2.0, 4.0]))
        
        result = 2.0 * vec1
        assert np.array_equal(result.vector, np.array([2.0, 4.0]))
    
    def test_with_metric(self):
        """Test vector operations with custom metric."""
        metric = np.array([[2.0, 0.0], [0.0, 2.0]])
        vec1 = SavoirVector(np.array([1.0, 1.0]))
        vec2 = SavoirVector(np.array([1.0, 1.0]))
        
        # Inner product with metric
        ip = vec1.inner_product(vec2, metric)
        assert ip == 4.0  # 1*2*1 + 1*2*1 = 4
        
        # Norm with metric
        norm = vec1.norm(metric)
        assert np.isclose(norm, np.sqrt(4.0))


class TestPouvoirMatrix:
    """Tests for PouvoirMatrix class."""
    
    def test_creation(self):
        """Test matrix creation."""
        matrix = np.eye(3)
        p = PouvoirMatrix(matrix, label="identity")
        assert p.shape == (3, 3)
        assert p.is_square
        assert p.label == "identity"
    
    def test_apply(self):
        """Test matrix application to vector."""
        matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
        p = PouvoirMatrix(matrix)
        
        vec = np.array([1.0, 1.0])
        result = p.apply(vec)
        assert np.array_equal(result, np.array([2.0, 3.0]))
    
    def test_apply_to_savoir(self):
        """Test matrix application to SavoirVector."""
        matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
        p = PouvoirMatrix(matrix)
        
        s = SavoirVector(np.array([1.0, 1.0]), label="test")
        result = p.apply_to_savoir(s)
        
        assert isinstance(result, SavoirVector)
        assert np.array_equal(result.vector, np.array([2.0, 3.0]))
    
    def test_determinant(self):
        """Test determinant calculation."""
        matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
        p = PouvoirMatrix(matrix)
        assert p.determinant() == 6.0
    
    def test_eigenvalues(self):
        """Test eigenvalue calculation."""
        matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
        p = PouvoirMatrix(matrix)
        eigvals = p.eigenvalues()
        assert np.allclose(np.sort(eigvals), np.array([2.0, 3.0]))
    
    def test_norms(self):
        """Test matrix norm calculations."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = PouvoirMatrix(matrix)
        
        # Spectral norm (largest singular value)
        spectral_norm = p.spectral_norm()
        assert np.isclose(spectral_norm, np.linalg.norm(matrix, ord=2))
        
        # Frobenius norm
        frobenius_norm = p.frobenius_norm()
        assert np.isclose(frobenius_norm, np.linalg.norm(matrix, ord='fro'))
    
    def test_invertible(self):
        """Test invertibility check."""
        # Invertible matrix
        matrix1 = np.eye(2)
        p1 = PouvoirMatrix(matrix1)
        assert p1.is_invertible()
        
        # Non-invertible matrix
        matrix2 = np.array([[1.0, 1.0], [1.0, 1.0]])
        p2 = PouvoirMatrix(matrix2)
        assert not p2.is_invertible()
    
    def test_inverse(self):
        """Test matrix inversion."""
        matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
        p = PouvoirMatrix(matrix)
        inv_p = p.inverse()
        
        # Check that inverse is correct
        identity = p.matrix @ inv_p.matrix
        assert np.allclose(identity, np.eye(2))
    
    def test_operations(self):
        """Test matrix arithmetic operations."""
        matrix1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        matrix2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        
        p1 = PouvoirMatrix(matrix1)
        p2 = PouvoirMatrix(matrix2)
        
        # Addition
        result = p1 + p2
        expected = np.array([[6.0, 8.0], [10.0, 12.0]])
        assert np.array_equal(result.matrix, expected)
        
        # Subtraction
        result = p2 - p1
        expected = np.array([[4.0, 4.0], [4.0, 4.0]])
        assert np.array_equal(result.matrix, expected)
        
        # Scalar multiplication
        result = p1 * 2.0
        expected = np.array([[2.0, 4.0], [6.0, 8.0]])
        assert np.array_equal(result.matrix, expected)
        
        result = 2.0 * p1
        assert np.array_equal(result.matrix, expected)
        
        # Matrix multiplication
        p3 = PouvoirMatrix(np.eye(2))
        result = p1 * p3
        assert np.array_equal(result.matrix, matrix1)


class TestSavoirSpace:
    """Tests for SavoirSpace class."""
    
    def test_creation(self):
        """Test space creation."""
        space = SavoirSpace(dimension=3)
        assert space.dimension == 3
        assert np.array_equal(space.metric, np.eye(3))
    
    def test_create_vector(self):
        """Test vector creation in space."""
        space = SavoirSpace(dimension=2)
        vec = space.create_vector([1.0, 2.0], label="test")
        
        assert isinstance(vec, SavoirVector)
        assert vec.dimension == 2
        assert vec.label == "test"
    
    def test_create_vector_wrong_dimension(self):
        """Test vector creation with wrong dimension."""
        space = SavoirSpace(dimension=2)
        
        with pytest.raises(ValueError):
            space.create_vector([1.0, 2.0, 3.0])
    
    def test_inner_product(self):
        """Test inner product in space."""
        space = SavoirSpace(dimension=2)
        vec1 = space.create_vector([1.0, 0.0])
        vec2 = space.create_vector([0.0, 1.0])
        
        assert space.inner_product(vec1, vec2) == 0.0
    
    def test_norm(self):
        """Test norm calculation in space."""
        space = SavoirSpace(dimension=2)
        vec = space.create_vector([3.0, 4.0])
        
        assert np.isclose(space.norm(vec), 5.0)
    
    def test_angle(self):
        """Test angle calculation in space."""
        space = SavoirSpace(dimension=2)
        vec1 = space.create_vector([1.0, 0.0])
        vec2 = space.create_vector([0.0, 1.0])
        
        assert np.isclose(space.angle(vec1, vec2), np.pi / 2)
    
    def test_distance(self):
        """Test distance calculation in space."""
        space = SavoirSpace(dimension=2)
        vec1 = space.create_vector([0.0, 0.0])
        vec2 = space.create_vector([3.0, 4.0])
        
        assert np.isclose(space.distance(vec1, vec2), 5.0)
    
    def test_set_metric(self):
        """Test setting custom metric."""
        space = SavoirSpace(dimension=2)
        custom_metric = np.array([[2.0, 0.0], [0.0, 2.0]])
        space.set_metric(custom_metric)
        
        assert np.array_equal(space.metric, custom_metric)
    
    def test_set_metric_wrong_shape(self):
        """Test setting metric with wrong shape."""
        space = SavoirSpace(dimension=2)
        wrong_metric = np.eye(3)
        
        with pytest.raises(ValueError):
            space.set_metric(wrong_metric)
    
    def test_random_vector(self):
        """Test random vector generation."""
        space = SavoirSpace(dimension=3)
        vec = space.random_vector()
        
        assert isinstance(vec, SavoirVector)
        assert vec.dimension == 3
    
    def test_basis(self):
        """Test basis vector generation."""
        space = SavoirSpace(dimension=3)
        basis = space.basis()
        
        assert len(basis) == 3
        for i, vec in enumerate(basis):
            assert isinstance(vec, SavoirVector)
            assert vec.dimension == 3
            # Check that only the i-th component is 1
            expected = np.zeros(3)
            expected[i] = 1.0
            assert np.array_equal(vec.vector, expected)


class TestAlignmentEngine:
    """Tests for AlignmentEngine class."""
    
    def test_alignment_score(self):
        """Test alignment score calculation."""
        space = SavoirSpace(dimension=2)
        engine = AlignmentEngine(space)
        
        # Create test vectors
        s1 = space.create_vector([1.0, 0.0])
        s2 = space.create_vector([0.0, 1.0])
        
        # Create target vectors (same as source)
        t1 = space.create_vector([1.0, 0.0])
        t2 = space.create_vector([0.0, 1.0])
        
        # Identity matrix should give perfect alignment
        identity = PouvoirMatrix(np.eye(2))
        score = engine.alignment_score([s1, s2], identity, [t1, t2])
        
        assert np.isclose(score, 0.0)
    
    def test_alignment_score_with_mismatch(self):
        """Test alignment score with mismatched vectors."""
        space = SavoirSpace(dimension=2)
        engine = AlignmentEngine(space)
        
        # Create test vectors
        s1 = space.create_vector([1.0, 0.0])
        s2 = space.create_vector([0.0, 1.0])
        
        # Create different target vectors
        t1 = space.create_vector([0.0, 1.0])
        t2 = space.create_vector([1.0, 0.0])
        
        # Identity matrix should give non-zero score
        identity = PouvoirMatrix(np.eye(2))
        score = engine.alignment_score([s1, s2], identity, [t1, t2])
        
        assert score > 0
    
    def test_coherence_score(self):
        """Test coherence score calculation."""
        space = SavoirSpace(dimension=2)
        engine = AlignmentEngine(space)
        
        # Create orthogonal vectors
        s1 = space.create_vector([1.0, 0.0])
        s2 = space.create_vector([0.0, 1.0])
        
        # Identity matrix preserves angles
        identity = PouvoirMatrix(np.eye(2))
        coherence = engine.coherence_score([s1, s2], identity)
        
        # Should be perfect coherence (1.0) for identity with orthogonal vectors
        assert coherence == 1.0
    
    def test_optimize_alignment(self):
        """Test alignment optimization."""
        space = SavoirSpace(dimension=2)
        engine = AlignmentEngine(space)
        
        # Create test vectors
        s1 = space.create_vector([1.0, 0.0])
        s2 = space.create_vector([0.0, 1.0])
        
        # Create target vectors (rotated by 90 degrees)
        t1 = space.create_vector([0.0, 1.0])
        t2 = space.create_vector([-1.0, 0.0])
        
        # Initial power matrix (identity)
        initial_pouvoir = PouvoirMatrix(np.eye(2))
        
        # Optimize
        optimized_pouvoir = engine.optimize_alignment(
            [s1, s2],
            [t1, t2],
            initial_pouvoir=initial_pouvoir,
            max_iter=100
        )
        
        # Check that optimization improved the score
        initial_score = engine.alignment_score(
            [s1, s2], initial_pouvoir, [t1, t2]
        )
        optimized_score = engine.alignment_score(
            [s1, s2], optimized_pouvoir, [t1, t2]
        )
        
        assert optimized_score < initial_score
    
    def test_optimize_alignment_perfect_case(self):
        """Test optimization with perfect alignment possible."""
        space = SavoirSpace(dimension=2)
        engine = AlignmentEngine(space)
        
        # Create test vectors
        s1 = space.create_vector([1.0, 0.0])
        s2 = space.create_vector([0.0, 1.0])
        
        # Create target vectors (same as source)
        t1 = space.create_vector([1.0, 0.0])
        t2 = space.create_vector([0.0, 1.0])
        
        # Initial power matrix (identity)
        initial_pouvoir = PouvoirMatrix(np.eye(2))
        
        # Optimize
        optimized_pouvoir = engine.optimize_alignment(
            [s1, s2],
            [t1, t2],
            initial_pouvoir=initial_pouvoir,
            max_iter=100
        )
        
        # Should achieve perfect alignment
        optimized_score = engine.alignment_score(
            [s1, s2], optimized_pouvoir, [t1, t2]
        )
        
        assert np.isclose(optimized_score, 0.0, atol=1e-6)
    
    def test_justice_score(self):
        """Test justice score calculation."""
        space = SavoirSpace(dimension=2)
        engine = AlignmentEngine(space)
        
        # Create identical power matrices (perfect equality)
        matrix = np.eye(2)
        pouvoir_matrices = [PouvoirMatrix(matrix) for _ in range(10)]
        
        justice = engine.justice_score(pouvoir_matrices)
        
        # Perfect equality should give justice score of 1.0
        assert justice == 1.0
    
    def test_justice_score_inequality(self):
        """Test justice score with unequal power matrices."""
        space = SavoirSpace(dimension=2)
        engine = AlignmentEngine(space)
        
        # Create power matrices with different scales
        matrices = []
        for i in range(1, 11):
            matrix = np.array([[i, 0.0], [0.0, i]])
            matrices.append(PouvoirMatrix(matrix))
        
        justice = engine.justice_score(matrices)
        
        # Unequal matrices should give justice score < 1.0
        assert justice < 1.0


class TestAlignmentFunctions:
    """Tests for standalone alignment functions."""
    
    def test_create_alignment_tensor(self):
        """Test alignment tensor creation."""
        space = SavoirSpace(dimension=2)
        
        # Create test data
        s1 = space.create_vector([1.0, 0.0])
        s2 = space.create_vector([0.0, 1.0])
        savoir_vectors = [s1, s2]
        
        p1 = PouvoirMatrix(np.eye(2))
        p2 = PouvoirMatrix(np.array([[0.0, -1.0], [1.0, 0.0]]))
        pouvoir_matrices = [p1, p2]
        
        t1 = space.create_vector([1.0, 0.0])
        t2 = space.create_vector([0.0, 1.0])
        target_vectors = [t1, t2]
        
        # Create tensor
        tensor = create_alignment_tensor(
            space, savoir_vectors, pouvoir_matrices, target_vectors
        )
        
        # Check tensor shape
        assert tensor.shape == (2, 2, 2)
        
        # Check some values
        # For identity matrix and matching target, distance should be 0
        assert np.isclose(tensor[0, 0, 0], 0.0)  # s1, p1, t1
        assert np.isclose(tensor[1, 0, 1], 0.0)  # s2, p1, t2
    
    def test_calculate_alignment_gradient(self):
        """Test alignment gradient calculation."""
        space = SavoirSpace(dimension=2)
        
        # Create test data
        s = space.create_vector([1.0, 0.0])
        p = PouvoirMatrix(np.eye(2))
        t = space.create_vector([0.0, 1.0])
        
        # Calculate gradient
        gradient = calculate_alignment_gradient(space, s, p, t)
        
        # Check gradient shape
        assert gradient.shape == (2, 2)
        
        # For this simple case, we can verify some properties
        # The gradient should be related to the error vector [0-1, 1-0] = [-1, 1]
        # and the input vector [1, 0]
        # So gradient[0,0] = 2 * (-1) * 1 = -2
        # And gradient[1,0] = 2 * (1) * 1 = 2
        assert gradient[0, 0] == -2.0
        assert gradient[1, 0] == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
