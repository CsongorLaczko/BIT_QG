"""
Diagonal preconditioner for quantum graph problems.

This module implements a diagonal preconditioner that computes the diagonal
entries of the Schur complement system through matrix-vector products,
corresponding to the DiagonalPreconditioner in the C++ implementation.
"""

from typing import TYPE_CHECKING

import numpy as np

from .base import PreconditionerBase

if TYPE_CHECKING:
    from ..core.mf_quantum_graph import MFQuantumGraph


class DiagonalPreconditioner(PreconditionerBase):
    """
    Diagonal preconditioner for quantum graph finite element systems.

    This preconditioner computes the diagonal entries of the Schur complement
    system matrix by applying the matrix to unit vectors. It then uses the
    inverse of these diagonal entries as a diagonal preconditioner.

    The preconditioner matrix M is diagonal with:
    M[i,i] = 1 / A[i,i] if A[i,i] != 0, else 1
    where A[i,i] is the i-th diagonal entry of the Schur complement system.
    """

    def __init__(self):
        """Initialize the diagonal preconditioner."""
        super().__init__()
        self.inverse_diagonal = None
        self._size = 0

    def compute(self, mfqg: "MFQuantumGraph") -> "DiagonalPreconditioner":
        """
        Compute the diagonal preconditioner from Schur complement diagonal entries.

        This method computes the diagonal entries of the Schur complement system
        by applying the system matrix to unit vectors: A * e_i where e_i is the
        i-th unit vector. The diagonal entry A[i,i] is then extracted from the
        result.

        Args:
            mfqg: The MFQuantumGraph instance

        Returns:
            Self for method chaining
        """
        n = mfqg.vertices
        self._size = n
        self.inverse_diagonal = np.zeros(n)

        # Compute diagonal entries by applying the system matrix to unit vectors
        for i in range(n):
            # Create unit vector e_i
            unit_vector = np.zeros(n)
            unit_vector[i] = 1.0

            # Apply the Schur complement system: A * e_i
            # This gives us the i-th column of the system matrix
            result = mfqg.solve_direct(unit_vector)

            # Extract the diagonal entry A[i,i]
            diagonal_entry = result[i]

            # Store the inverse diagonal entry for preconditioning
            if abs(diagonal_entry) > 1e-14:  # Avoid division by zero
                self.inverse_diagonal[i] = 1.0 / diagonal_entry
            else:
                self.inverse_diagonal[i] = 1.0  # Identity for zero diagonal entries

        self.is_initialized = True
        return self

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        Apply the diagonal preconditioner to vector b.

        Args:
            b: Input vector of length vertices

        Returns:
            Preconditioned vector M^(-1) * b where M is the diagonal matrix

        Raises:
            RuntimeError: If compute() has not been called
            ValueError: If RHS size doesn't match system size
        """
        if not self.is_initialized:
            raise RuntimeError("DiagonalPreconditioner must be computed before use")

        if len(b) != self._size:
            raise ValueError(
                f"RHS size {len(b)} does not match system size {self._size}"
            )

        return self.inverse_diagonal * b

    def _validate_computed(self) -> None:
        """Validate that the preconditioner has been computed."""
        if not self.is_initialized:
            raise RuntimeError("DiagonalPreconditioner must be computed before use")
        if self.inverse_diagonal is None:
            raise RuntimeError("Preconditioner computation failed")

    def __repr__(self) -> str:
        """String representation."""
        if self.is_initialized:
            n_vertices = len(self.inverse_diagonal)
            min_diag = np.min(self.inverse_diagonal)
            max_diag = np.max(self.inverse_diagonal)
            return (
                f"DiagonalPreconditioner(vertices={n_vertices}, "
                f"diag_range=[{min_diag:.2e}, {max_diag:.2e}])"
            )
        else:
            return "DiagonalPreconditioner(not computed)"
