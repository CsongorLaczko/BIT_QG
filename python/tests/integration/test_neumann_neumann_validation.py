"""
Integration tests for Neumann-Neumann preconditioner validation.

This module contains tests that validate the mathematical properties and
correctness of the Neumann-Neumann preconditioner implementation.
"""

import numpy as np
import pytest

from bit_qg.core import MFQuantumGraph, QGEdge
from bit_qg.preconditioners import NeumannNeumannPreconditioner


class TestNeumannNeumannValidation:
    """Validation tests for Neumann-Neumann preconditioner mathematical properties."""

    def test_preconditioner_symmetry(self):
        """Test that the preconditioner produces symmetric-like behavior."""

        # Create test system
        def c_func(x: float) -> float:
            return 1.0

        def v_func(x: float) -> float:
            return 0.0

        def f_func(x: float) -> float:
            return 1.0

        edge = QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func)
        qg = MFQuantumGraph(N=7, vertices=2, edges=[edge])

        precond = NeumannNeumannPreconditioner()
        precond.compute(qg)

        # Test that applying the preconditioner to symmetric inputs gives reasonable results
        rhs1 = np.array([1.0, 1.0])
        rhs2 = np.array([2.0, 2.0])

        result1 = precond.solve(rhs1)
        result2 = precond.solve(rhs2)

        # Results should scale linearly for this simple case
        np.testing.assert_allclose(result2, 2.0 * result1, rtol=1e-10)

    def test_preconditioner_consistency_multiple_edges(self):
        """Test preconditioner behavior with multiple edges."""

        # Create test system with two edges
        def c_func(x: float) -> float:
            return 1.0

        def v_func(x: float) -> float:
            return 0.0

        def f_func(x: float) -> float:
            return 1.0

        edge1 = QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func)
        edge2 = QGEdge(out=1, in_=2, c=c_func, v=v_func, f=f_func)
        qg = MFQuantumGraph(N=5, vertices=3, edges=[edge1, edge2])

        precond = NeumannNeumannPreconditioner()
        precond.compute(qg)

        # Test solve
        rhs = np.array([1.0, 0.0, 1.0])
        result = precond.solve(rhs)

        # Result should be finite and reasonable
        assert np.all(np.isfinite(result))
        assert result.shape == (3,)
        # Middle vertex (vertex 1) should be affected by both edges
        assert result[1] != 0.0

    def test_zero_rhs_gives_zero_result(self):
        """Test that zero right-hand side gives zero result."""

        def c_func(x: float) -> float:
            return 1.0

        def v_func(x: float) -> float:
            return 0.0

        def f_func(x: float) -> float:
            return 1.0

        edge = QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func)
        qg = MFQuantumGraph(N=5, vertices=2, edges=[edge])

        precond = NeumannNeumannPreconditioner()
        precond.compute(qg)

        # Test with zero RHS
        rhs = np.array([0.0, 0.0])
        result = precond.solve(rhs)

        np.testing.assert_allclose(result, np.zeros(2), atol=1e-14)

    def test_preconditioner_different_n_values(self):
        """Test that preconditioner works with different discretization sizes."""

        def c_func(x: float) -> float:
            return 1.0

        def v_func(x: float) -> float:
            return 0.0

        def f_func(x: float) -> float:
            return 1.0

        rhs = np.array([1.0, 1.0])

        for N in [4, 5, 6, 7, 10]:
            edge = QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func)
            qg = MFQuantumGraph(N=N, vertices=2, edges=[edge])

            precond = NeumannNeumannPreconditioner()
            precond.compute(qg)

            result = precond.solve(rhs)

            # All results should be finite
            assert np.all(np.isfinite(result))
            assert result.shape == (2,)

    def test_coefficient_function_effects(self):
        """Test that different coefficient functions produce different results."""

        def c_func1(x: float) -> float:
            return 1.0

        def c_func2(x: float) -> float:
            return 2.0

        def v_func(x: float) -> float:
            return 0.0

        def f_func(x: float) -> float:
            return 1.0

        rhs = np.array([1.0, 1.0])

        # Test with c=1
        edge1 = QGEdge(out=0, in_=1, c=c_func1, v=v_func, f=f_func)
        qg1 = MFQuantumGraph(N=5, vertices=2, edges=[edge1])
        precond1 = NeumannNeumannPreconditioner()
        precond1.compute(qg1)
        result1 = precond1.solve(rhs)

        # Test with c=2
        edge2 = QGEdge(out=0, in_=1, c=c_func2, v=v_func, f=f_func)
        qg2 = MFQuantumGraph(N=5, vertices=2, edges=[edge2])
        precond2 = NeumannNeumannPreconditioner()
        precond2.compute(qg2)
        result2 = precond2.solve(rhs)

        # Results should be different (not equal)
        assert not np.allclose(result1, result2, rtol=1e-10)

    def test_matrix_assembly_properties(self):
        """Test properties of the assembled local matrices."""

        def c_func(x: float) -> float:
            return 1.0

        def v_func(x: float) -> float:
            return 0.0

        def f_func(x: float) -> float:
            return 1.0

        edge = QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func)
        qg = MFQuantumGraph(N=7, vertices=2, edges=[edge])

        precond = NeumannNeumannPreconditioner()
        precond.compute(qg)

        # Check that we have the right number of local matrices
        assert len(precond.neumann_solvers) == 1

        local_matrix = precond.neumann_solvers[0]

        # Matrix should be square and of size N
        assert local_matrix.shape == (qg.N, qg.N)

        # Matrix should be sparse
        assert hasattr(local_matrix, "nnz")

        # For this simple case with constant coefficients, check some properties
        matrix_dense = local_matrix.toarray()

        # Check that boundary rows have non-zero diagonal entries
        assert matrix_dense[qg.N - 2, qg.N - 2] != 0  # Left boundary
        assert matrix_dense[qg.N - 1, qg.N - 1] != 0  # Right boundary
