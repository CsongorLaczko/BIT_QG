"""
Degree-based preconditioner for quantum graph problems.

This module implements a simple preconditioner based on vertex degrees,
corresponding to the DegreePreconditioner in the C++ implementation.
"""

from typing import TYPE_CHECKING

import numpy as np

from .base import PreconditionerBase

if TYPE_CHECKING:
    from ..core.mf_quantum_graph import MFQuantumGraph


class DegreePreconditioner(PreconditionerBase):
    """
    Degree-based preconditioner for quantum graph finite element systems.

    This preconditioner uses the inverse of vertex weights (degrees) as a
    diagonal preconditioner. It's the simplest and fastest preconditioner,
    equivalent to Jacobi preconditioning with vertex degrees.

    The preconditioner matrix M is diagonal with:
    M[i,i] = 1 / vertex_weight[i] if vertex_weight[i] != 0, else 1
    """

    def __init__(self):
        """Initialize the degree preconditioner."""
        super().__init__()
        self.vertex_weights = None

    def compute(self, mfqg: "MFQuantumGraph") -> "DegreePreconditioner":
        """
        Compute the degree preconditioner from vertex weights.

        Args:
            mfqg: The MFQuantumGraph instance containing vertex weights

        Returns:
            Self for method chaining

        Raises:
            ValueError: If the quantum graph has no vertices
        """
        if not mfqg.vertex_weights:
            raise ValueError("Quantum graph must have vertex weights")

        self._size = len(mfqg.vertex_weights)
        self.vertex_weights = np.zeros(self._size)

        # Compute inverse weights, handling zero weights
        for i, weight in enumerate(mfqg.vertex_weights):
            if weight != 0:
                self.vertex_weights[i] = 1.0 / weight
            else:
                self.vertex_weights[i] = 1.0

        self.is_initialized = True
        return self

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Apply the degree preconditioner: x = M^(-1) * rhs.

        This computes the element-wise product of the inverse vertex weights
        with the right-hand side vector.

        Args:
            rhs: Right-hand side vector

        Returns:
            Solution vector x = M^(-1) * rhs

        Raises:
            RuntimeError: If preconditioner has not been computed
            ValueError: If rhs has wrong size
        """
        if not self.is_initialized:
            raise RuntimeError("Preconditioner must be computed before solving")

        if len(rhs) != self._size:
            raise ValueError(
                f"Right-hand side size {len(rhs)} does not match "
                f"preconditioner size {self._size}"
            )

        # Element-wise multiplication (equivalent to diagonal matrix multiply)
        return self.vertex_weights * rhs

    def __repr__(self) -> str:
        """String representation of the degree preconditioner."""
        if self.is_initialized:
            return f"DegreePreconditioner(size={self._size}, initialized=True)"
        else:
            return "DegreePreconditioner(initialized=False)"
