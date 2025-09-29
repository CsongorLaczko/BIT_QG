"""
Neumann-Neumann domain decomposition preconditioner for quantum graph problems.

This module implements the Neumann-Neumann preconditioner using domain decomposition
methods, corresponding to the NeumannNeumannPreconditioner in the C++ implementation.
"""

from enum import Enum
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, spsolve

from .base import PreconditionerBase

if TYPE_CHECKING:
    from ..core.mf_quantum_graph import MFQuantumGraph


class VertexBCType(Enum):
    """Boundary condition types for vertices based on domain decomposition theory."""

    BOUNDARY_NEUMANN = (
        "boundary_neumann"  # Degree-1 vertices (∂G) - homogeneous Neumann-Kirchhoff
    )
    INTERIOR_CONTINUITY = (
        "interior_continuity"  # Degree>1 vertices (int(G)) - continuity conditions
    )
    INTERFACE_DIRICHLET = (
        "interface_dirichlet"  # Interface vertices (Γ) - fixed values in Dirichlet step
    )
    INTERFACE_NEUMANN = (
        "interface_neumann"  # Interface vertices (Γ) - flux correction in Neumann step
    )


class NeumannNeumannStep(Enum):
    """Iteration step type in Neumann-Neumann domain decomposition."""

    DIRICHLET_STEP = "dirichlet_step"  # Interface gets Dirichlet BCs (fixed values)
    NEUMANN_STEP = "neumann_step"  # Interface gets Neumann BCs (flux correction)


class EdgeBCType(Enum):
    """Edge boundary condition types based on endpoint vertex classifications."""

    NN = "neumann_neumann"  # Both endpoints have Neumann conditions
    NC = "neumann_continuity"  # Out=Neumann, In=Continuity/Dirichlet
    CN = "continuity_neumann"  # Out=Continuity/Dirichlet, In=Neumann
    CC = "continuity_continuity"  # Both endpoints have Continuity/Dirichlet conditions


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
        self.vertex_weights = mfqg.vertex_weights

        # Compute vertex degrees for boundary condition classification
        self.vertex_degrees = np.zeros(self.vertices, dtype=int)
        for edge in self.edges:
            self.vertex_degrees[edge.out] += 1
            self.vertex_degrees[edge.in_] += 1

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

    def _classify_vertex_bc_type(
        self, vertex_id: int, step: NeumannNeumannStep
    ) -> VertexBCType:
        """
        Classify vertex boundary condition type based on domain decomposition theory and iteration step.

        Based on the Hungarian theory:
        - Degree-1 vertices (∂G): Always homogeneous Neumann-Kirchhoff conditions (zero flux)
        - Interior vertices (int(G)):
          * Dirichlet step: Interface Dirichlet conditions (fixed values)
          * Neumann step: Interface Neumann conditions (flux correction)

        Args:
            vertex_id: The vertex index to classify
            step: Whether we're in Dirichlet step or Neumann step of the iteration

        Returns:
            VertexBCType indicating the boundary condition type for this step
        """
        degree = self.vertex_degrees[vertex_id]

        if degree == 1:
            # Degree-1 vertices are boundary vertices (∂G)
            # They ALWAYS get homogeneous Neumann-Kirchhoff conditions (zero flux)
            # This is independent of the iteration step
            return VertexBCType.BOUNDARY_NEUMANN
        else:
            # Interior vertices (degree > 1) are interface vertices (Γ)
            # Their boundary condition type depends on the iteration step:
            if step == NeumannNeumannStep.DIRICHLET_STEP:
                # In Dirichlet step: interface vertices get fixed values (Dirichlet)
                return VertexBCType.INTERFACE_DIRICHLET
            else:  # NEUMANN_STEP
                # In Neumann step: interface vertices get flux correction (Neumann)
                return VertexBCType.INTERFACE_NEUMANN

    def _classify_edge_bc_type(self, edge, step: NeumannNeumannStep) -> EdgeBCType:
        """
        Classify edge boundary condition type based on endpoint vertex types and iteration step.

        This determines which neural network model should be used for this edge
        in the domain decomposition framework.

        Args:
            edge: QGEdge instance to classify
            step: Whether we're in Dirichlet step or Neumann step of the iteration

        Returns:
            EdgeBCType indicating the edge boundary condition combination
        """
        out_bc_type = self._classify_vertex_bc_type(edge.out, step)
        in_bc_type = self._classify_vertex_bc_type(edge.in_, step)

        # Map vertex BC types to edge BC types based on actual boundary conditions
        # Note: BOUNDARY_NEUMANN (degree-1) is always Neumann regardless of step
        # INTERFACE_DIRICHLET/INTERFACE_NEUMANN depends on the iteration step

        out_is_neumann = (
            out_bc_type == VertexBCType.BOUNDARY_NEUMANN
            or out_bc_type == VertexBCType.INTERFACE_NEUMANN
        )
        out_is_dirichlet = out_bc_type == VertexBCType.INTERFACE_DIRICHLET

        in_is_neumann = (
            in_bc_type == VertexBCType.BOUNDARY_NEUMANN
            or in_bc_type == VertexBCType.INTERFACE_NEUMANN
        )
        in_is_dirichlet = in_bc_type == VertexBCType.INTERFACE_DIRICHLET

        if out_is_neumann and in_is_neumann:
            return EdgeBCType.NN  # Both endpoints are Neumann
        elif out_is_neumann and in_is_dirichlet:
            return EdgeBCType.NC  # Out=Neumann, In=Dirichlet (mapped to NC)
        elif out_is_dirichlet and in_is_neumann:
            return EdgeBCType.CN  # Out=Dirichlet, In=Neumann (mapped to CN)
        else:  # Both Dirichlet
            return EdgeBCType.CC  # Both endpoints are Dirichlet (mapped to CC)

    def solve(
        self,
        rhs: np.ndarray,
        step: NeumannNeumannStep = NeumannNeumannStep.DIRICHLET_STEP,
    ) -> np.ndarray:
        """
        Apply the Neumann-Neumann preconditioner with step-aware boundary condition classification.

        This method solves local problems on each edge using boundary conditions
        appropriate for the current iteration step (Dirichlet or Neumann).

        Args:
            rhs: Right-hand side vector
            step: Which step of Neumann-Neumann iteration (Dirichlet or Neumann step)

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
            # Classify edge boundary condition type for neural network model selection
            # This is now step-aware: same edge can have different BC types depending on iteration step
            edge_bc_type = self._classify_edge_bc_type(edge, step)

            # Create local right-hand side
            local_rhs = np.zeros(self.N)

            # Set boundary conditions from global rhs, scaled by vertex weights
            local_rhs[self.N - 2] = (
                rhs[edge.out] * self.vertex_weights[edge.out]
            )  # Left boundary
            local_rhs[self.N - 1] = (
                rhs[edge.in_] * self.vertex_weights[edge.in_]
            )  # Right boundary

            # 🚀 NEURAL NETWORK INTEGRATION POINT:
            # Here we have all information needed for neural network inference:
            # - edge: contains coefficient functions c(x), v(x), f(x)
            # - local_rhs: boundary condition values
            # - edge_bc_type: which of 4 models to use (NN, NC, CN, CC)
            # - step: Dirichlet step or Neumann step of the iteration
            # - self.N: discretization points
            #
            # For now, continue with original finite element solver
            # TODO: Replace with neural network inference based on edge_bc_type AND step
            # Current step: {step.value}, Edge type: {edge_bc_type.value}
            # if self.use_neural_networks:
            #     local_solution = self._neural_network_solve(edge, local_rhs, edge_bc_type, step)
            # else:
            #     local_solution = spsolve(local_matrix, local_rhs)

            # Solve local Neumann problem
            try:
                local_solution = spsolve(local_matrix, local_rhs)
                # Check for NaN or infinite results
                if not np.all(np.isfinite(local_solution)):
                    logging.warning(
                        f"FALLBACK: Non-finite solution encountered on edge {edge}. "
                        f"Matrix condition: {np.linalg.cond(local_matrix.toarray()) if hasattr(local_matrix, 'toarray') else 'unknown'}. "
                        f"Using least squares fallback."
                    )
                    # Fallback: use least squares solution for singular systems
                    from scipy.sparse.linalg import lsqr

                    local_solution = lsqr(local_matrix, local_rhs)[0]
            except Exception as e:
                logging.error(
                    f"FALLBACK: Exception in local solve for edge {edge}: {e}. "
                    f"Matrix shape: {local_matrix.shape}, RHS shape: {local_rhs.shape}. "
                    f"Using simple boundary value approximation."
                )
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

    def as_linear_operator(self) -> LinearOperator:
        """
        Create a SciPy LinearOperator wrapper for this preconditioner.

        This overrides the base class method to handle the step parameter
        required by the Neumann-Neumann solve method.

        Returns:
            LinearOperator that applies the preconditioner with default DIRICHLET_STEP
        """
        if not self.is_initialized:
            raise RuntimeError("Preconditioner must be computed before use")

        def matvec(x):
            # Apply Neumann-Neumann preconditioner with default Dirichlet step
            # In a full implementation, the step type would be provided by the outer iteration
            return self.solve(x, NeumannNeumannStep.DIRICHLET_STEP)

        return LinearOperator(shape=self.shape, matvec=matvec, dtype=np.float64)

    def __repr__(self) -> str:
        """String representation of the Neumann-Neumann preconditioner."""
        if self.is_initialized:
            return (
                f"NeumannNeumannPreconditioner(size={self._size}, "
                f"edges={len(self.edges)}, initialized=True)"
            )
        else:
            return "NeumannNeumannPreconditioner(uninitialized)"
