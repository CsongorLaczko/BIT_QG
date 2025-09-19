"""
Base classes and interfaces for quantum graph preconditioners.

This module provides the abstract base class for all preconditioners used with
iterative solvers in the quantum graph finite element methods.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from ..core.mf_quantum_graph import MFQuantumGraph


class PreconditionerBase(ABC):
    """
    Abstract base class for quantum graph preconditioners.

    All preconditioners should inherit from this class and implement the
    required methods. The preconditioner should be compatible with SciPy's
    iterative solvers by providing a LinearOperator interface.
    """

    def __init__(self):
        """Initialize the preconditioner."""
        self.is_initialized = False
        self._size = 0

    @abstractmethod
    def compute(self, mfqg: "MFQuantumGraph") -> "PreconditionerBase":
        """
        Compute/factorize the preconditioner from the quantum graph.

        Args:
            mfqg: The MFQuantumGraph instance to compute preconditioner for

        Returns:
            Self for method chaining
        """

    @abstractmethod
    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Apply the preconditioner to solve M * x = rhs for x.

        Args:
            rhs: Right-hand side vector

        Returns:
            Solution vector x such that M * x = rhs
        """

    @property
    def size(self) -> int:
        """Get the size of the preconditioner matrix."""
        return self._size

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the preconditioner matrix."""
        return (self._size, self._size)

    def as_linear_operator(self) -> LinearOperator:
        """
        Create a SciPy LinearOperator wrapper for this preconditioner.

        This allows the preconditioner to be used with SciPy's iterative solvers
        like CG, BiCGSTAB, etc.

        Returns:
            LinearOperator that applies the preconditioner
        """
        if not self.is_initialized:
            raise RuntimeError("Preconditioner must be computed before use")

        def matvec(x):
            return self.solve(x)

        return LinearOperator(shape=self.shape, matvec=matvec, dtype=np.float64)

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        """
        Apply the preconditioner using the @ operator.

        Args:
            other: Vector to apply preconditioner to

        Returns:
            Result of applying preconditioner
        """
        return self.solve(other)
