"""
Core module for vector alignment of knowledge through the prism of power.

This module implements the fundamental mathematical operations for the
alignment framework, including space definitions, power transformations,
and alignment optimization.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class SavoirVector:
    """Represents a knowledge vector in the savoir space."""
    vector: np.ndarray
    label: str = ""
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the vector."""
        return len(self.vector)
    
    def norm(self, metric: Optional[np.ndarray] = None) -> float:
        """Calculate the norm of the vector with respect to a metric."""
        if metric is None:
            return np.linalg.norm(self.vector)
        else:
            return np.sqrt(self.vector @ metric @ self.vector)
    
    def normalize(self, metric: Optional[np.ndarray] = None) -> 'SavoirVector':
        """Normalize the vector with respect to a metric."""
        n = self.norm(metric)
        if n > 0:
            return SavoirVector(self.vector / n, self.label, self.metadata)
        return self
    
    def inner_product(self, other: 'SavoirVector', metric: Optional[np.ndarray] = None) -> float:
        """Calculate the inner product with another vector."""
        if self.dimension != other.dimension:
            raise ValueError(f"Vector dimensions don't match: {self.dimension} vs {other.dimension}")
        
        if metric is None:
            return np.dot(self.vector, other.vector)
        else:
            return self.vector @ metric @ other.vector
    
    def angle(self, other: 'SavoirVector', metric: Optional[np.ndarray] = None) -> float:
        """Calculate the angle between this vector and another."""
        ip = self.inner_product(other, metric)
        n1 = self.norm(metric)
        n2 = other.norm(metric)
        
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # Clamp to avoid numerical errors
        cos_theta = np.clip(ip / (n1 * n2), -1.0, 1.0)
        return np.arccos(cos_theta)
    
    def __add__(self, other: 'SavoirVector') -> 'SavoirVector':
        """Vector addition."""
        if self.dimension != other.dimension:
            raise ValueError(f"Vector dimensions don't match for addition")
        return SavoirVector(self.vector + other.vector)
    
    def __sub__(self, other: 'SavoirVector') -> 'SavoirVector':
        """Vector subtraction."""
        if self.dimension != other.dimension:
            raise ValueError(f"Vector dimensions don't match for subtraction")
        return SavoirVector(self.vector - other.vector)
    
    def __mul__(self, scalar: float) -> 'SavoirVector':
        """Scalar multiplication."""
        return SavoirVector(self.vector * scalar, self.label, self.metadata)
    
    def __rmul__(self, scalar: float) -> 'SavoirVector':
        """Reverse scalar multiplication."""
        return self.__mul__(scalar)
    
    def __repr__(self) -> str:
        return f"SavoirVector({self.vector}, label='{self.label}')"


@dataclass
class PouvoirMatrix:
    """Represents a power transformation matrix."""
    matrix: np.ndarray
    label: str = ""
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the matrix."""
        return self.matrix.shape
    
    @property
    def is_square(self) -> bool:
        """Check if the matrix is square."""
        return self.matrix.shape[0] == self.matrix.shape[1]
    
    def apply(self, vector: Union[np.ndarray, SavoirVector]) -> np.ndarray:
        """Apply the power transformation to a vector."""
        if isinstance(vector, SavoirVector):
            vec = vector.vector
        else:
            vec = vector
        
        if vec.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"Vector dimension {vec.shape[0]} doesn't match matrix columns {self.matrix.shape[1]}")
        
        return self.matrix @ vec
    
    def apply_to_savoir(self, savoir: SavoirVector) -> SavoirVector:
        """Apply the transformation to a SavoirVector."""
        transformed = self.apply(savoir)
        return SavoirVector(transformed, f"{savoir.label}_x_{self.label}", savoir.metadata)
    
    def determinant(self) -> float:
        """Calculate the determinant of the matrix."""
        if not self.is_square:
            raise ValueError("Determinant is only defined for square matrices")
        return np.linalg.det(self.matrix)
    
    def eigenvalues(self) -> np.ndarray:
        """Calculate the eigenvalues of the matrix."""
        if not self.is_square:
            raise ValueError("Eigenvalues are only defined for square matrices")
        return np.linalg.eigvals(self.matrix)
    
    def spectral_norm(self) -> float:
        """Calculate the spectral norm (largest singular value)."""
        return np.linalg.norm(self.matrix, ord=2)
    
    def frobenius_norm(self) -> float:
        """Calculate the Frobenius norm."""
        return np.linalg.norm(self.matrix, ord='fro')
    
    def is_invertible(self, tol: float = 1e-10) -> bool:
        """Check if the matrix is invertible."""
        if not self.is_square:
            return False
        return np.abs(self.determinant()) > tol
    
    def inverse(self) -> 'PouvoirMatrix':
        """Calculate the inverse of the matrix."""
        if not self.is_invertible():
            raise ValueError("Matrix is not invertible")
        return PouvoirMatrix(np.linalg.inv(self.matrix), f"inv_{self.label}", self.metadata)
    
    def __add__(self, other: 'PouvoirMatrix') -> 'PouvoirMatrix':
        """Matrix addition."""
        if self.shape != other.shape:
            raise ValueError(f"Matrix shapes don't match for addition")
        return PouvoirMatrix(self.matrix + other.matrix)
    
    def __sub__(self, other: 'PouvoirMatrix') -> 'PouvoirMatrix':
        """Matrix subtraction."""
        if self.shape != other.shape:
            raise ValueError(f"Matrix shapes don't match for subtraction")
        return PouvoirMatrix(self.matrix - other.matrix)
    
    def __mul__(self, other: Union['PouvoirMatrix', float, SavoirVector]) -> Union['PouvoirMatrix', SavoirVector]:
        """Matrix multiplication or scalar multiplication."""
        if isinstance(other, (float, int)):
            return PouvoirMatrix(self.matrix * other, self.label, self.metadata)
        elif isinstance(other, PouvoirMatrix):
            if self.matrix.shape[1] != other.matrix.shape[0]:
                raise ValueError(f"Matrix dimensions incompatible for multiplication: {self.shape} vs {other.shape}")
            return PouvoirMatrix(self.matrix @ other.matrix, f"{self.label}_x_{other.label}")
        elif isinstance(other, SavoirVector):
            return self.apply_to_savoir(other)
        else:
            raise TypeError(f"Unsupported type for multiplication: {type(other)}")
    
    def __rmul__(self, scalar: float) -> 'PouvoirMatrix':
        """Reverse scalar multiplication."""
        return self.__mul__(scalar)
    
    def __repr__(self) -> str:
        return f"PouvoirMatrix(shape={self.shape}, label='{self.label}')"


class SavoirSpace:
    """Represents the vector space of knowledge."""
    
    def __init__(self, dimension: int, metric: Optional[np.ndarray] = None):
        """
        Initialize a savoir space.
        
        Args:
            dimension: The dimension of the space
            metric: The metric tensor (Gram matrix) for the space
        """
        self.dimension = dimension
        self.metric = metric if metric is not None else np.eye(dimension)
        self._validate_metric()
    
    def _validate_metric(self):
        """Validate that the metric is positive definite."""
        try:
            np.linalg.cholesky(self.metric)
        except np.linalg.LinAlgError:
            warnings.warn("Metric matrix is not positive definite. Some operations may not work correctly.")
    
    def create_vector(self, vector: Union[np.ndarray, List[float]], label: str = "", metadata: dict = None) -> SavoirVector:
        """Create a SavoirVector in this space."""
        vec = np.array(vector)
        if len(vec) != self.dimension:
            raise ValueError(f"Vector dimension {len(vec)} doesn't match space dimension {self.dimension}")
        return SavoirVector(vec, label, metadata)
    
    def inner_product(self, v1: SavoirVector, v2: SavoirVector) -> float:
        """Calculate the inner product of two vectors in this space."""
        return v1.inner_product(v2, self.metric)
    
    def norm(self, vector: SavoirVector) -> float:
        """Calculate the norm of a vector in this space."""
        return vector.norm(self.metric)
    
    def angle(self, v1: SavoirVector, v2: SavoirVector) -> float:
        """Calculate the angle between two vectors in this space."""
        return v1.angle(v2, self.metric)
    
    def distance(self, v1: SavoirVector, v2: SavoirVector) -> float:
        """Calculate the distance between two vectors in this space."""
        diff = v1 - v2
        return self.norm(diff)
    
    def set_metric(self, metric: np.ndarray):
        """Set the metric tensor for the space."""
        if metric.shape != (self.dimension, self.dimension):
            raise ValueError(f"Metric shape {metric.shape} doesn't match space dimension {self.dimension}")
        self.metric = metric
        self._validate_metric()
    
    def random_vector(self, label: str = "") -> SavoirVector:
        """Generate a random vector in this space."""
        vec = np.random.randn(self.dimension)
        return self.create_vector(vec, label)
    
    def basis(self) -> List[SavoirVector]:
        """Get the standard basis vectors for this space."""
        basis_vectors = []
        for i in range(self.dimension):
            vec = np.zeros(self.dimension)
            vec[i] = 1.0
            basis_vectors.append(self.create_vector(vec, f"e_{i}"))
        return basis_vectors


class AlignmentEngine:
    """Engine for optimizing alignment between knowledge and power."""
    
    def __init__(self, space: SavoirSpace):
        """
        Initialize the alignment engine.
        
        Args:
            space: The savoir space in which to operate
        """
        self.space = space
    
    def alignment_score(
        self, 
        savoir_vectors: List[SavoirVector], 
        pouvoir: PouvoirMatrix,
        target_vectors: List[SavoirVector]
    ) -> float:
        """
        Calculate the alignment score between savoir vectors and power transformation.
        
        Args:
            savoir_vectors: List of knowledge vectors
            pouvoir: Power transformation matrix
            target_vectors: List of target vectors for alignment
            
        Returns:
            Alignment score (lower is better)
        """
        if len(savoir_vectors) != len(target_vectors):
            raise ValueError("Number of savoir vectors and target vectors must match")
        
        total_error = 0.0
        for s, t in zip(savoir_vectors, target_vectors):
            transformed = pouvoir.apply_to_savoir(s)
            error = self.space.distance(transformed, t)
            total_error += error ** 2
        
        return total_error / len(savoir_vectors)
    
    def coherence_score(
        self,
        savoir_vectors: List[SavoirVector],
        pouvoir: PouvoirMatrix
    ) -> float:
        """
        Calculate the coherence score for a set of vectors under a power transformation.
        
        Args:
            savoir_vectors: List of knowledge vectors
            pouvoir: Power transformation matrix
            
        Returns:
            Coherence score between 0 and 1 (higher is better)
        """
        if len(savoir_vectors) < 2:
            return 1.0
        
        # Calculate pairwise angles before transformation
        original_angles = []
        for i in range(len(savoir_vectors)):
            for j in range(i + 1, len(savoir_vectors)):
                angle = self.space.angle(savoir_vectors[i], savoir_vectors[j])
                original_angles.append(angle)
        
        # Calculate pairwise angles after transformation
        transformed_vectors = [pouvoir.apply_to_savoir(s) for s in savoir_vectors]
        transformed_angles = []
        for i in range(len(transformed_vectors)):
            for j in range(i + 1, len(transformed_vectors)):
                angle = self.space.angle(transformed_vectors[i], transformed_vectors[j])
                transformed_angles.append(angle)
        
        # Calculate similarity between angle distributions
        if len(original_angles) == 0:
            return 1.0
        
        # Simple correlation-based similarity
        if np.std(original_angles) > 0 and np.std(transformed_angles) > 0:
            correlation = np.corrcoef(original_angles, transformed_angles)[0, 1]
            return (correlation + 1) / 2  # Map from [-1,1] to [0,1]
        return 1.0
    
    def optimize_alignment(
        self,
        savoir_vectors: List[SavoirVector],
        target_vectors: List[SavoirVector],
        initial_pouvoir: Optional[PouvoirMatrix] = None,
        method: str = 'L-BFGS-B',
        max_iter: int = 100,
        constraints: Optional[List[dict]] = None
    ) -> PouvoirMatrix:
        """
        Optimize the power transformation to align savoir vectors with target vectors.
        
        Args:
            savoir_vectors: List of knowledge vectors
            target_vectors: List of target vectors
            initial_pouvoir: Initial power transformation (optional)
            method: Optimization method
            max_iter: Maximum number of iterations
            constraints: List of constraints for the optimization
            
        Returns:
            Optimized power transformation matrix
        """
        from scipy.optimize import minimize
        
        dim = self.space.dimension
        
        # Initialize power matrix
        if initial_pouvoir is None:
            # Start with identity matrix
            p_matrix = np.eye(dim)
        else:
            p_matrix = initial_pouvoir.matrix
        
        # Flatten the matrix for optimization
        x0 = p_matrix.flatten()
        
        def objective(x):
            """Objective function for alignment optimization."""
            # Reshape to matrix
            P = x.reshape((dim, dim))
            pouvoir = PouvoirMatrix(P)
            
            # Calculate alignment score
            score = self.alignment_score(savoir_vectors, pouvoir, target_vectors)
            return score
        
        # Define constraints if provided
        opt_constraints = []
        if constraints:
            for constraint in constraints:
                if constraint['type'] == 'determinant':
                    # Constraint: det(P) > 0 (for invertibility)
                    def det_constraint(x):
                        P = x.reshape((dim, dim))
                        return np.linalg.det(P) - 1e-6
                    opt_constraints.append({'type': 'ineq', 'fun': det_constraint})
                elif constraint['type'] == 'norm':
                    # Constraint: ||P|| <= max_norm
                    max_norm = constraint.get('max_norm', 10.0)
                    def norm_constraint(x):
                        P = x.reshape((dim, dim))
                        return max_norm - np.linalg.norm(P)
                    opt_constraints.append({'type': 'ineq', 'fun': norm_constraint})
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method=method,
            constraints=opt_constraints,
            options={'maxiter': max_iter, 'disp': False}
        )
        
        # Create and return the optimized power matrix
        optimized_matrix = result.x.reshape((dim, dim))
        return PouvoirMatrix(optimized_matrix, "optimized")
    
    def justice_score(self, pouvoir_matrices: List[PouvoirMatrix]) -> float:
        """
        Calculate the justice score for a set of power matrices.
        
        Args:
            pouvoir_matrices: List of power transformation matrices
            
        Returns:
            Justice score between 0 and 1 (higher is better)
        """
        # Collect all eigenvalues
        all_eigenvalues = []
        for p in pouvoir_matrices:
            if p.is_square:
                eigvals = p.eigenvalues()
                # Take absolute values of real parts
                all_eigenvalues.extend(np.abs(eigvals.real))
        
        if len(all_eigenvalues) == 0:
            return 1.0
        
        # Calculate Gini coefficient for inequality
        def gini_coefficient(values):
            """Calculate the Gini coefficient for a list of values."""
            values = np.array(values)
            values = values[values > 0]  # Remove zeros
            if len(values) == 0:
                return 0.0
            
            sorted_values = np.sort(values)
            n = len(sorted_values)
            cumsum = np.cumsum(sorted_values) / sorted_values.sum()
            cumsum = np.concatenate([[0], cumsum])  # Add 0 at the beginning
            
            # Gini coefficient formula
            gini = (n + 1 - 2 * np.sum(cumsum)) / n
            return gini
        
        gini = gini_coefficient(all_eigenvalues)
        # Justice score is 1 - Gini (higher is more equal)
        return 1.0 - gini


def create_alignment_tensor(
    space: SavoirSpace,
    savoir_vectors: List[SavoirVector],
    pouvoir_matrices: List[PouvoirMatrix],
    target_vectors: List[SavoirVector]
) -> np.ndarray:
    """
    Create the alignment tensor that captures relationships between
    savoir, pouvoir, and objectives.
    
    Args:
        space: The savoir space
        savoir_vectors: List of knowledge vectors
        pouvoir_matrices: List of power transformation matrices
        target_vectors: List of target vectors
        
    Returns:
        Alignment tensor of shape (n_savoir, n_pouvoir, n_target)
    """
    n_savoir = len(savoir_vectors)
    n_pouvoir = len(pouvoir_matrices)
    n_target = len(target_vectors)
    
    tensor = np.zeros((n_savoir, n_pouvoir, n_target))
    
    for i, s in enumerate(savoir_vectors):
        for j, p in enumerate(pouvoir_matrices):
            for k, t in enumerate(target_vectors):
                # Apply power transformation
                transformed = p.apply_to_savoir(s)
                # Calculate distance in the space
                distance = space.distance(transformed, t)
                tensor[i, j, k] = distance
    
    return tensor


def calculate_alignment_gradient(
    space: SavoirSpace,
    savoir_vector: SavoirVector,
    pouvoir_matrix: PouvoirMatrix,
    target_vector: SavoirVector
) -> np.ndarray:
    """
    Calculate the gradient of the alignment error with respect to the power matrix.
    
    Args:
        space: The savoir space
        savoir_vector: Knowledge vector
        pouvoir_matrix: Power transformation matrix
        target_vector: Target vector
        
    Returns:
        Gradient matrix of the same shape as pouvoir_matrix
    """
    # Current transformation
    transformed = pouvoir_matrix.apply_to_savoir(savoir_vector)
    
    # Error vector in the space
    error_vector = transformed - target_vector
    
    # Gradient calculation
    # For each element P_ij of the power matrix:
    # dE/dP_ij = sum_k (dE/dy_k) * (dy_k/dP_ij)
    # where y = P @ s, so dy_k/dP_ij = s_j
    
    gradient = np.zeros_like(pouvoir_matrix.matrix)
    
    for i in range(pouvoir_matrix.shape[0]):
        for j in range(pouvoir_matrix.shape[1]):
            # dy_i/dP_ij = s_j
            ds_dp = savoir_vector.vector[j]
            
            # dE/dy_i = 2 * (y_i - t_i) * g_ii (considering metric)
            # For simplicity, we use the standard metric here
            de_dy = 2 * error_vector.vector[i]
            
            gradient[i, j] = de_dy * ds_dp
    
    return gradient


__all__ = [
    'SavoirVector',
    'PouvoirMatrix',
    'SavoirSpace',
    'AlignmentEngine',
    'create_alignment_tensor',
    'calculate_alignment_gradient'
]
