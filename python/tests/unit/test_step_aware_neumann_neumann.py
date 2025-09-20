"""
Test step-aware boundary condition classification in Neumann-Neumann preconditioner.

This module tests the step-aware boundary condition classification functionality
that enables proper neural network model selection based on iteration step.
"""

import numpy as np

from bit_qg.core import MFQuantumGraph, QGEdge
from bit_qg.preconditioners.neumann_neumann import (
    EdgeBCType,
    NeumannNeumannPreconditioner,
    NeumannNeumannStep,
    VertexBCType,
)


class TestStepAwareBoundaryConditions:
    """Test step-aware boundary condition classification."""

    def test_step_aware_vertex_classification(self):
        """Test that vertex classification depends on iteration step."""

        # Create simple graph: 0 -- 1 -- 2 (degrees: [1, 2, 1])
        def c_func(x):
            return 1.0

        def v_func(x):
            return 0.0

        def f_func(x):
            return 0.0

        edges = [
            QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func),
            QGEdge(out=1, in_=2, c=c_func, v=v_func, f=f_func),
        ]
        mfqg = MFQuantumGraph(N=9, vertices=3, edges=edges)

        preconditioner = NeumannNeumannPreconditioner()
        preconditioner.compute(mfqg)

        # Test boundary vertices (degree 1) - should be BOUNDARY_NEUMANN in both steps
        for step in [
            NeumannNeumannStep.DIRICHLET_STEP,
            NeumannNeumannStep.NEUMANN_STEP,
        ]:
            vertex_0_bc = preconditioner._classify_vertex_bc_type(0, step)
            vertex_2_bc = preconditioner._classify_vertex_bc_type(2, step)

            assert vertex_0_bc == VertexBCType.BOUNDARY_NEUMANN
            assert vertex_2_bc == VertexBCType.BOUNDARY_NEUMANN

        # Test interior vertex (degree > 1) - should depend on step
        vertex_1_dirichlet = preconditioner._classify_vertex_bc_type(
            1, NeumannNeumannStep.DIRICHLET_STEP
        )
        vertex_1_neumann = preconditioner._classify_vertex_bc_type(
            1, NeumannNeumannStep.NEUMANN_STEP
        )

        assert vertex_1_dirichlet == VertexBCType.INTERFACE_DIRICHLET
        assert vertex_1_neumann == VertexBCType.INTERFACE_NEUMANN

    def test_step_aware_edge_classification(self):
        """Test that edge classification depends on iteration step."""

        # Create simple graph: 0 -- 1 -- 2 (degrees: [1, 2, 1])
        def c_func(x):
            return 1.0

        def v_func(x):
            return 0.0

        def f_func(x):
            return 0.0

        edges = [
            QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func),
            QGEdge(out=1, in_=2, c=c_func, v=v_func, f=f_func),
        ]
        mfqg = MFQuantumGraph(N=9, vertices=3, edges=edges)

        preconditioner = NeumannNeumannPreconditioner()
        preconditioner.compute(mfqg)

        edge_0_1 = edges[0]  # 0 -> 1: boundary -> interior
        edge_1_2 = edges[1]  # 1 -> 2: interior -> boundary

        # Test Dirichlet step
        edge_0_1_dirichlet = preconditioner._classify_edge_bc_type(
            edge_0_1, NeumannNeumannStep.DIRICHLET_STEP
        )
        edge_1_2_dirichlet = preconditioner._classify_edge_bc_type(
            edge_1_2, NeumannNeumannStep.DIRICHLET_STEP
        )

        # In Dirichlet step: vertex 1 gets Dirichlet conditions
        # Edge 0->1: Neumann (vertex 0) -> Dirichlet (vertex 1) = NC
        # Edge 1->2: Dirichlet (vertex 1) -> Neumann (vertex 2) = CN
        assert edge_0_1_dirichlet == EdgeBCType.NC
        assert edge_1_2_dirichlet == EdgeBCType.CN

        # Test Neumann step
        edge_0_1_neumann = preconditioner._classify_edge_bc_type(
            edge_0_1, NeumannNeumannStep.NEUMANN_STEP
        )
        edge_1_2_neumann = preconditioner._classify_edge_bc_type(
            edge_1_2, NeumannNeumannStep.NEUMANN_STEP
        )

        # In Neumann step: vertex 1 gets Neumann conditions
        # Edge 0->1: Neumann (vertex 0) -> Neumann (vertex 1) = NN
        # Edge 1->2: Neumann (vertex 1) -> Neumann (vertex 2) = NN
        assert edge_0_1_neumann == EdgeBCType.NN
        assert edge_1_2_neumann == EdgeBCType.NN

    def test_step_aware_solve_compatibility(self):
        """Test that step-aware solve method maintains backward compatibility."""

        # Create simple test system
        def c_func(x):
            return 1.0

        def v_func(x):
            return 0.0

        def f_func(x):
            return 0.0

        edges = [QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func)]
        mfqg = MFQuantumGraph(N=7, vertices=2, edges=edges)

        preconditioner = NeumannNeumannPreconditioner()
        preconditioner.compute(mfqg)

        rhs = np.array([1.0, 1.0])

        # Test that default step is DIRICHLET_STEP
        sol_default = preconditioner.solve(rhs)
        sol_dirichlet = preconditioner.solve(rhs, NeumannNeumannStep.DIRICHLET_STEP)

        np.testing.assert_allclose(sol_default, sol_dirichlet, rtol=1e-14)

        # Test that both steps produce valid results
        sol_neumann = preconditioner.solve(rhs, NeumannNeumannStep.NEUMANN_STEP)

        assert np.all(np.isfinite(sol_dirichlet))
        assert np.all(np.isfinite(sol_neumann))
        assert np.linalg.norm(sol_dirichlet) > 0
        assert np.linalg.norm(sol_neumann) > 0

    def test_linear_operator_step_aware_compatibility(self):
        """Test that LinearOperator works with step-aware solve method."""

        # Create simple test system
        def c_func(x):
            return 1.0

        def v_func(x):
            return 0.0

        def f_func(x):
            return 0.0

        edges = [QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func)]
        mfqg = MFQuantumGraph(N=7, vertices=2, edges=edges)

        preconditioner = NeumannNeumannPreconditioner()
        preconditioner.compute(mfqg)

        rhs = np.array([1.0, 1.0])

        # Test LinearOperator uses default DIRICHLET_STEP
        linear_op = preconditioner.as_linear_operator()
        sol_linear_op = linear_op @ rhs
        sol_dirichlet = preconditioner.solve(rhs, NeumannNeumannStep.DIRICHLET_STEP)

        np.testing.assert_allclose(sol_linear_op, sol_dirichlet, rtol=1e-14)

        # Test LinearOperator properties
        assert linear_op.shape == (mfqg.vertices, mfqg.vertices)
        assert linear_op.dtype == np.float64
