"""
Neumann-Neumann domain decomposition preconditioner for quantum graph problems.

This module implements the Neumann-Neumann preconditioner using domain decomposition
methods, corresponding to the NeumannNeumannPreconditioner in the C++ implementation.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from .base import PreconditionerBase

if TYPE_CHECKING:
    from ..core.mf_quantum_graph import MFQuantumGraph


class NeumannNeumannPreconditioner(PreconditionerBase):
    """
    Neumann-Neumann domain decomposition preconditioner.

    This preconditioner uses domain decomposition methods where each edge is treated
    as a separate subdomain. Local Neumann problems are solved on each edge and
    results are combined using vertex weight-based averaging.

    The preconditioner works by:
    1. Assembling local finite element matrices for each edge
    2. Solving local Neumann problems with Dirichlet boundary conditions
    3. Combining solutions using vertex weights as scaling factors
    """

    def __init__(self):
        """Initialize the Neumann-Neumann preconditioner."""
        super().__init__()
        self.neumann_solvers: list = []
        self.vertex_weights: np.ndarray = None
        self.edges = []
        self.vertices: int = 0
        self.N: int = 0  # Number of discretization points per edge

    def compute(self, mfqg: "MFQuantumGraph") -> "NeumannNeumannPreconditioner":
        """
        Compute the Neumann-Neumann preconditioner from the quantum graph.

        This method assembles local finite element matrices for each edge and
        prepares the local solvers.

        Args:
            mfqg: The MFQuantumGraph instance containing the quantum graph

        Returns:
            Self for method chaining

        Raises:
            ValueError: If the quantum graph is incomplete
        """
        # Validate input
        if not hasattr(mfqg, "edges") or not hasattr(mfqg, "vertex_weights"):
            raise ValueError("Quantum graph must have edges and vertex_weights")

        if not mfqg.edges or not mfqg.vertex_weights:
            raise ValueError("Quantum graph must have edges and vertex_weights")

        # Store quantum graph parameters
        self.vertices = mfqg.vertices
        self.N = mfqg.N
        self.edges = mfqg.edges
        self._size = self.vertices

        # Clear any existing solvers
        self.neumann_solvers = []

        # Assemble local matrices for each edge
        for edge in self.edges:
            local_matrix = self._assemble_edge_matrix(edge)
            self.neumann_solvers.append(local_matrix)

        # Compute vertex weights (inverse of weights for preconditioning)
        self.vertex_weights = np.zeros(len(mfqg.vertex_weights))
        for i, weight in enumerate(mfqg.vertex_weights):
            if weight != 0:
                self.vertex_weights[i] = 1.0 / weight
            else:
                self.vertex_weights[i] = 1.0

        self.is_initialized = True
        return self

    def _assemble_edge_matrix(self, edge) -> csc_matrix:
        """
        Assemble the local finite element matrix for a single edge.

        This method creates the local stiffness matrix following the C++ implementation
        exactly. The C++ code structure is:
        - sizeI = N-2 interior points (indices 0 to sizeI-1)
        - 2 boundary points (indices sizeI and sizeI+1)
        - Total matrix size: sizeI+2 = N

        Args:
            edge: QGEdge instance containing edge data

        Returns:
            Local sparse matrix for the edge
        """
        # Create discretization points
        x = np.linspace(0, 1, self.N)
        h = x[1] - x[0]

        # Evaluate coefficient functions
        c_vals = np.array([edge.c(xi) for xi in x])
        v_vals = np.array([edge.v(xi) for xi in x])

        # Size calculations (following C++ exactly)
        size_interior = self.N - 2  # Interior points
        size_total = size_interior + 2  # Interior + 2 boundary points

        # Calculate coupling coefficients (used later)
        lcoeff = -(c_vals[0] + c_vals[1]) / (2 * h)
        rcoeff = -(c_vals[self.N - 2] + c_vals[self.N - 1]) / (2 * h)

        # Initialize sparse matrix storage
        row_indices = []
        col_indices = []
        data = []

        # Assemble interior-interior block exactly following C++ code

        # First interior point (C++ loop: emplace_back(0, 0, ...) and emplace_back(0, 1, ...))
        if size_interior > 0:
            row_indices.append(0)
            col_indices.append(0)
            data.append(
                (c_vals[0] + 2 * c_vals[1] + c_vals[2]) / (2 * h) + h * v_vals[1]
            )

            if size_interior > 1:
                row_indices.append(0)
                col_indices.append(1)
                data.append(-(c_vals[1] + c_vals[2]) / (2 * h))

        # Middle interior points (C++ loop: for i=1; i<N-3; ++i)
        # Note: N-3 = (sizeI+2)-3 = sizeI-1, so this is for i=1 to sizeI-2
        for i in range(1, size_interior - 1):
            # Sub-diagonal
            row_indices.append(i)
            col_indices.append(i - 1)
            data.append(-(c_vals[i] + c_vals[i + 1]) / (2 * h))

            # Diagonal
            row_indices.append(i)
            col_indices.append(i)
            data.append(
                (c_vals[i] + 2 * c_vals[i + 1] + c_vals[i + 2]) / (2 * h)
                + h * v_vals[i + 1]
            )

            # Super-diagonal
            row_indices.append(i)
            col_indices.append(i + 1)
            data.append(-(c_vals[i + 1] + c_vals[i + 2]) / (2 * h))

        # Last interior point (C++: emplace_back(N-3, N-4, ...) and emplace_back(N-3, N-3, ...))
        # N-3 = sizeI-1, N-4 = sizeI-2
        if size_interior > 1:
            last_idx = size_interior - 1
            row_indices.append(last_idx)
            col_indices.append(last_idx - 1)
            data.append(-(c_vals[self.N - 3] + c_vals[self.N - 2]) / (2 * h))

            row_indices.append(last_idx)
            col_indices.append(last_idx)
            data.append(
                (c_vals[self.N - 3] + 2 * c_vals[self.N - 2] + c_vals[self.N - 1])
                / (2 * h)
                + h * v_vals[self.N - 2]
            )

        # Create the interior block first
        interior_matrix = csc_matrix(
            (data, (row_indices, col_indices)), shape=(size_total, size_total)
        )

        # Convert to lil_matrix for efficient insertion of coupling terms
        local_matrix = interior_matrix.tolil()

        # Add coupling terms (C++: local_A.insert(...))
        # Interior-boundary coupling
        if size_interior > 0:
            # Left boundary coupling (out vertex)
            local_matrix[0, size_interior] = lcoeff
            local_matrix[size_interior, 0] = lcoeff

            # Right boundary coupling (in vertex)
            if size_interior > 1:
                local_matrix[size_interior - 1, size_interior + 1] = rcoeff
                local_matrix[size_interior + 1, size_interior - 1] = rcoeff

        # Boundary-boundary block
        local_matrix[size_interior, size_interior] = -lcoeff + v_vals[0] * h / 2
        local_matrix[size_interior + 1, size_interior + 1] = (
            -rcoeff + v_vals[self.N - 1] * h / 2
        )

        return local_matrix.tocsc()

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Apply the Neumann-Neumann preconditioner.

        This method solves local Neumann problems on each edge and combines
        the results using vertex weight-based averaging.

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

        # Initialize solution vector
        solution = np.zeros(self.vertices)

        # Solve local problems for each edge
        for edge, local_matrix in zip(self.edges, self.neumann_solvers, strict=True):
            # Create local right-hand side
            local_rhs = np.zeros(self.N)

            # Set boundary conditions from global rhs, scaled by vertex weights
            local_rhs[self.N - 2] = (
                rhs[edge.out] * self.vertex_weights[edge.out]
            )  # Left boundary
            local_rhs[self.N - 1] = (
                rhs[edge.in_] * self.vertex_weights[edge.in_]
            )  # Right boundary

            # Solve local Neumann problem
            try:
                local_solution = spsolve(local_matrix, local_rhs)
                # Check for NaN or infinite results
                if not np.all(np.isfinite(local_solution)):
                    # Fallback: use least squares solution for singular systems
                    from scipy.sparse.linalg import lsqr
                    local_solution = lsqr(local_matrix, local_rhs)[0]
            except Exception:
                # Final fallback for singular/ill-conditioned matrices
                # Use a simple approximation based on boundary values
                local_solution = np.zeros(self.N)
                local_solution[self.N - 2] = local_rhs[self.N - 2]
                local_solution[self.N - 1] = local_rhs[self.N - 1]

            # Extract boundary values and accumulate to global solution
            solution[edge.out] += (
                local_solution[self.N - 2] * self.vertex_weights[edge.out]
            )
            solution[edge.in_] += (
                local_solution[self.N - 1] * self.vertex_weights[edge.in_]
            )

        return solution

    def __repr__(self) -> str:
        """String representation of the Neumann-Neumann preconditioner."""
        if self.is_initialized:
            return (
                f"NeumannNeumannPreconditioner(size={self._size}, "
                f"edges={len(self.edges)}, initialized=True)"
            )
        else:
            return "NeumannNeumannPreconditioner(uninitialized)"
