"""
Polynomial preconditioner for quantum graph problems.

This module implements a polynomial preconditioner that uses the Schur complement
system to compute diagonal entries, corresponding to the PolynomialPreconditioner
in the C++ implementation.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import splu

from .base import PreconditionerBase

if TYPE_CHECKING:
    from ..core.mf_quantum_graph import MFQuantumGraph


class PolynomialPreconditioner(PreconditionerBase):
    """
    Polynomial preconditioner for quantum graph finite element systems.

    This preconditioner computes diagonal entries of the Schur complement system
    by solving unit vector problems. It provides better conditioning than the
    degree preconditioner at higher computational cost.

    The preconditioner applies the formula:
    x = vertex_weights * (2*b - (AGG*vertex_weights*b - AIG.T*solver.solve(AIG*vertex_weights*b)))
    """

    def __init__(self):
        """Initialize the polynomial preconditioner."""
        super().__init__()
        self.vertex_weights = None
        self.AII_solver = None
        self.AIG = None
        self.AGG = None

    def compute(self, mfqg: "MFQuantumGraph") -> "PolynomialPreconditioner":
        """
        Compute the polynomial preconditioner from the quantum graph matrices.

        This method extracts the matrices from the quantum graph, factors AII,
        and computes the diagonal entries of the Schur complement.

        Args:
            mfqg: The MFQuantumGraph instance containing the system matrices

        Returns:
            Self for method chaining

        Raises:
            ValueError: If the quantum graph is not properly initialized
        """
        if (
            not hasattr(mfqg, "AII")
            or not hasattr(mfqg, "AIG")
            or not hasattr(mfqg, "AGG")
        ):
            raise ValueError("Quantum graph must have AII, AIG, and AGG matrices")

        # Store matrix references and create solver for AII
        self.AIG = mfqg.AIG
        self.AGG = mfqg.AGG
        self.AII_solver = splu(mfqg.AII.tocsc())

        # Compute size from AGG matrix (graph vertices)
        self._size = self.AGG.shape[0]
        self.vertex_weights = np.zeros(self._size)

        # Compute diagonal entries of Schur complement
        # For each graph vertex i, solve: dii = (AGG*e_i - AIG.T*AII^(-1)*AIG*e_i)[i]
        # where e_i is the i-th unit vector
        for i in range(self._size):
            # Create unit vector
            unit_rhs = np.zeros(self._size)
            unit_rhs[i] = 1.0

            # Compute Schur complement operation: AGG*e_i - AIG.T*AII^(-1)*AIG*e_i
            aig_unit = self.AIG @ unit_rhs
            aii_solve = self.AII_solver.solve(aig_unit)
            schur_result = self.AGG @ unit_rhs - self.AIG.T @ aii_solve

            # Extract diagonal entry
            dii = schur_result[i]

            # Compute inverse, handling zeros
            if abs(dii) > 1e-12:  # Numerical tolerance
                self.vertex_weights[i] = 1.0 / dii
            else:
                self.vertex_weights[i] = 1.0

        self.is_initialized = True
        return self

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Apply the polynomial preconditioner.

        This implements the polynomial preconditioner formula:
        x = vertex_weights * (2*b - (AGG*vertex_weights*b - AIG.T*AII^(-1)*AIG*vertex_weights*b))

        Args:
            rhs: Right-hand side vector

        Returns:
            Solution vector after applying the polynomial preconditioner

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

        # Apply polynomial preconditioner formula
        # Step 1: vertex_weights * rhs
        weighted_rhs = self.vertex_weights * rhs

        # Step 2: AGG * weighted_rhs
        agg_weighted = self.AGG @ weighted_rhs

        # Step 3: AIG * weighted_rhs
        aig_weighted = self.AIG @ weighted_rhs

        # Step 4: AII^(-1) * (AIG * weighted_rhs)
        aii_solve = self.AII_solver.solve(aig_weighted)

        # Step 5: AIG.T * (AII^(-1) * (AIG * weighted_rhs))
        aig_t_solve = self.AIG.T @ aii_solve

        # Step 6: Complete polynomial formula
        # vertex_weights * (2*rhs - (AGG*vertex_weights*rhs - AIG.T*AII^(-1)*AIG*vertex_weights*rhs))
        polynomial_term = 2 * rhs - (agg_weighted - aig_t_solve)
        result = self.vertex_weights * polynomial_term

        return result

    def __repr__(self) -> str:
        """String representation of the polynomial preconditioner."""
        if self.is_initialized:
            return f"PolynomialPreconditioner(size={self._size}, initialized=True)"
        else:
            return "PolynomialPreconditioner(initialized=False)"
