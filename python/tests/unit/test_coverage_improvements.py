"""
Additional tests to improve coverage for edge cases and error conditions.
"""

import numpy as np
import pytest

from bit_qg.core import MFQuantumGraph, QGEdge
from bit_qg.preconditioners.diagonal import DiagonalPreconditioner
from bit_qg.preconditioners.neumann_neumann import NeumannNeumannPreconditioner


class TestCoverageImprovements:
    """Tests to cover missing lines and edge cases."""

    def test_neumann_neumann_invalid_graph_missing_attributes(self):
        """Test error handling for invalid graph without required attributes."""
        precond = NeumannNeumannPreconditioner()

        # Test with object missing edges attribute
        class InvalidGraph1:
            vertex_weights = [1.0, 1.0]

        with pytest.raises(
            ValueError, match="Quantum graph must have edges and vertex_weights"
        ):
            precond.compute(InvalidGraph1())

        # Test with object missing vertex_weights attribute
        class InvalidGraph2:
            edges = []

        with pytest.raises(
            ValueError, match="Quantum graph must have edges and vertex_weights"
        ):
            precond.compute(InvalidGraph2())

    def test_neumann_neumann_empty_graph(self):
        """Test error handling for graph with empty edges or vertex_weights."""
        precond = NeumannNeumannPreconditioner()

        # Test with empty edges
        class EmptyEdgesGraph:
            edges = []
            vertex_weights = [1.0, 1.0]

        with pytest.raises(
            ValueError, match="Quantum graph must have edges and vertex_weights"
        ):
            precond.compute(EmptyEdgesGraph())

        # Test with empty vertex_weights
        def c_func(x):
            return 1.0

        def v_func(x):
            return 0.0

        def f_func(x):
            return 0.0

        class EmptyWeightsGraph:
            edges = [QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func)]
            vertex_weights = []

        with pytest.raises(
            ValueError, match="Quantum graph must have edges and vertex_weights"
        ):
            precond.compute(EmptyWeightsGraph())

    def test_neumann_neumann_zero_vertex_weights(self):
        """Test handling of zero vertex weights."""

        def c_func(x):
            return 1.0

        def v_func(x):
            return 0.0

        def f_func(x):
            return 0.0

        # Create graph with zero vertex weight
        class ZeroWeightGraph:
            def __init__(self):
                self.vertices = 2
                self.N = 5
                self.edges = [QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func)]
                self.vertex_weights = [0.0, 1.0]  # Zero weight for vertex 0

        precond = NeumannNeumannPreconditioner()
        precond.compute(ZeroWeightGraph())

        # Check that zero weight is handled (converted to 1.0)
        assert precond.vertex_weights[0] == 1.0
        assert precond.vertex_weights[1] == 1.0

    def test_diagonal_preconditioner_error_conditions(self):
        """Test error conditions for diagonal preconditioner."""
        precond = DiagonalPreconditioner()

        # Test solve before compute
        with pytest.raises(
            RuntimeError, match="Preconditioner must be computed before use"
        ):
            precond.solve(np.array([1.0, 1.0]))

        # Test as_linear_operator before compute
        with pytest.raises(
            RuntimeError, match="Preconditioner must be computed before use"
        ):
            precond.as_linear_operator()

    def test_diagonal_preconditioner_wrong_rhs_size(self):
        """Test diagonal preconditioner with wrong RHS size."""

        def c_func(x):
            return 1.0

        def v_func(x):
            return 0.0

        def f_func(x):
            return 0.0

        edge = QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func)
        qg = MFQuantumGraph(N=5, vertices=2, edges=[edge])

        precond = DiagonalPreconditioner()
        precond.compute(qg)

        # Test with wrong size RHS
        wrong_size_rhs = np.array([1.0, 1.0, 1.0])  # Size 3 instead of 2
        with pytest.raises(ValueError, match="RHS size .* does not match system size"):
            precond.solve(wrong_size_rhs)

    def test_graph_io_edge_case(self):
        """Test edge case in graph_io module."""
        from bit_qg.utils.graph_io import GraphLoader

        loader = GraphLoader()

        # Test with non-existent file (should raise FileNotFoundError)
        with pytest.raises(FileNotFoundError):
            loader.load_from_adjacency_matrix("nonexistent_file.txt")
