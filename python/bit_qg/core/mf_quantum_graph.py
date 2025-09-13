"""
Multi-scale Finite Element Quantum Graph implementation.

This module contains the MFQuantumGraph class which implements finite element
methods on quantum graphs using the Schur complement approach.
"""

import numpy as np
from scipy import sparse

from .qgedge import QGEdge


class MFQuantumGraph:
    """
    Multi-scale Finite Element Quantum Graph solver.

    This is the Python equivalent of the C++ MFQuantumGraph class. It implements
    finite element discretization on quantum graphs with custom boundary conditions
    using the Schur complement method.

    The system is split into interior (I) and graph (G) degrees of freedom:
    - AII: Interior-interior coupling matrix
    - AIG: Interior-graph coupling matrix
    - AGG: Graph-graph coupling matrix
    - bI: Interior load vector
    - bG: Graph load vector

    The solve method implements the Schur complement:
    AGG * rhs - AIG.T @ solver.solve(AIG @ rhs)
    """

    def __init__(self, N: int, vertices: int, edges: list[QGEdge]) -> None:
        """
        Initialize the quantum graph finite element system.

        Args:
            N: Number of discretization points along each edge
            vertices: Number of vertices in the graph
            edges: List of QGEdge objects defining the graph structure
        """
        self.N = N
        self.vertices = vertices
        self.edges = edges
        self.vertex_weights = [0] * vertices

        # Matrix dimensions
        size_I = len(edges) * (N - 2)  # Interior degrees of freedom
        size_G = vertices  # Graph/vertex degrees of freedom

        # Initialize triplet lists for sparse matrix construction
        coefficients_II = []
        coefficients_IG = []
        coefficients_GG = []

        # Initialize load vectors
        self.bI = np.zeros(size_I)
        self.bG = np.zeros(size_G)

        # Accumulate vertex stiffness and load contributions
        vertex_stiffness = np.zeros(size_G)

        # Count vertex degrees (number of incident edges)
        for edge in edges:
            self.vertex_weights[edge.out] += 1
            self.vertex_weights[edge.in_] += 1

        # Discretization points on [0,1]
        x = np.linspace(0, 1, N)
        h = x[1] - x[0]  # Grid spacing

        # Process each edge to build matrices
        for i, edge in enumerate(edges):
            out = edge.out
            in_ = edge.in_

            # Evaluate edge functions at discretization points
            cx = np.array([edge.c(xi) for xi in x])
            vx = np.array([edge.v(xi) for xi in x])
            fx = np.array([edge.f(xi) for xi in x])

            edge_offset = i * (N - 2)

            # Build AII matrix (interior-interior coupling)
            # First interior point
            coefficients_II.append((
                edge_offset, edge_offset,
                (cx[0] + 2*cx[1] + cx[2])/(2*h) + h*vx[1]
            ))
            coefficients_II.append((
                edge_offset, edge_offset + 1,
                -(cx[1] + cx[2])/(2*h)
            ))

            # Middle interior points
            for j in range(1, N - 3):
                idx = edge_offset + j
                coefficients_II.append((idx, idx - 1, -(cx[j] + cx[j+1])/(2*h)))
                coefficients_II.append((
                    idx, idx,
                    (cx[j] + 2*cx[j+1] + cx[j+2])/(2*h) + h*vx[j+1]
                ))
                coefficients_II.append((idx, idx + 1, -(cx[j+1] + cx[j+2])/(2*h)))

            # Last interior point
            last_idx = edge_offset + N - 3
            coefficients_II.append((
                last_idx, last_idx - 1,
                -(cx[N-3] + cx[N-2])/(2*h)
            ))
            coefficients_II.append((
                last_idx, last_idx,
                (cx[N-3] + 2*cx[N-2] + cx[N-1])/(2*h) + h*vx[N-2]
            ))

            # Build AIG matrix (interior-graph coupling)
            lcoeff = -(cx[0] + cx[1])/(2*h)
            rcoeff = -(cx[N-2] + cx[N-1])/(2*h)
            coefficients_IG.append((edge_offset, out, lcoeff))
            coefficients_IG.append((edge_offset + N - 3, in_, rcoeff))

            # Accumulate vertex stiffness contributions
            vertex_stiffness[out] += -lcoeff + vx[0]*h/2
            vertex_stiffness[in_] += -rcoeff + vx[N-1]*h/2

            # Build load vectors
            self.bI[edge_offset:edge_offset + N - 2] = h * fx[1:N-1]
            self.bG[out] += fx[0] * h/2
            self.bG[in_] += fx[N-1] * h/2

        # Build AGG matrix (graph-graph coupling)
        for i in range(len(vertex_stiffness)):
            coefficients_GG.append((i, i, vertex_stiffness[i]))

        # Convert triplets to sparse matrices
        self.AII = sparse.csr_matrix(
            ([coeff for _, _, coeff in coefficients_II],
             ([row for row, _, _ in coefficients_II],
              [col for _, col, _ in coefficients_II])),
            shape=(size_I, size_I)
        )

        self.AIG = sparse.csr_matrix(
            ([coeff for _, _, coeff in coefficients_IG],
             ([row for row, _, _ in coefficients_IG],
              [col for _, col, _ in coefficients_IG])),
            shape=(size_I, size_G)
        )

        self.AGG = sparse.csr_matrix(
            ([coeff for _, _, coeff in coefficients_GG],
             ([row for row, _, _ in coefficients_GG],
              [col for _, col, _ in coefficients_GG])),
            shape=(size_G, size_G)
        )

        # Factor AII matrix for Schur complement solves
        from scipy.sparse.linalg import splu
        # Convert to CSC format to avoid efficiency warning
        self.solver = splu(self.AII.tocsc())

        # Pre-compute Schur complement right-hand side
        self.bG -= self.AIG.T @ self.solver.solve(self.bI)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve the Schur complement system using the factored AII matrix.

        This implements: AGG * rhs - AIG.T @ AII^(-1) @ (AIG @ rhs)

        Args:
            rhs: Right-hand side vector of length vertices

        Returns:
            Solution vector of length vertices
        """
        return self.AGG @ rhs - self.AIG.T @ self.solver.solve(self.AIG @ rhs)

    def __matmul__(self, rhs: np.ndarray) -> np.ndarray:
        """Matrix-vector multiplication operator (Python equivalent of operator*)."""
        return self.solve(rhs)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the linear operator."""
        return (self.vertices, self.vertices)

    def __repr__(self) -> str:
        """String representation."""
        return (f"MFQuantumGraph(N={self.N}, vertices={self.vertices}, "
                f"edges={len(self.edges)})")
