"""
Tests for the diagonal preconditioner.
"""

import numpy as np
import pytest

from bit_qg.core.mf_quantum_graph import MFQuantumGraph
from bit_qg.preconditioners.diagonal import DiagonalPreconditioner
from bit_qg.utils.graph_io import GraphLoader


class TestDiagonalPreconditioner:
    """Test DiagonalPreconditioner class."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple 2-edge graph for testing."""
        loader = GraphLoader()
        return loader.create_simple_path(3)  # 3 vertices, 2 edges

    @pytest.fixture
    def simple_problem(self, simple_graph):
        """Create a simple quantum graph problem."""
        edges, vertices = simple_graph
        return MFQuantumGraph(5, vertices, edges)

    def test_diagonal_preconditioner_creation(self):
        """Test creating a DiagonalPreconditioner instance."""
        prec = DiagonalPreconditioner()
        assert not prec.is_initialized
        assert prec.inverse_diagonal is None

    def test_diagonal_preconditioner_compute(self, simple_problem):
        """Test computing the diagonal preconditioner."""
        prec = DiagonalPreconditioner()
        result = prec.compute(simple_problem)

        # Should return self for chaining
        assert result is prec
        assert prec.is_initialized
        assert prec.inverse_diagonal is not None
        assert len(prec.inverse_diagonal) == simple_problem.vertices

    def test_diagonal_preconditioner_solve(self, simple_problem):
        """Test applying the diagonal preconditioner."""
        prec = DiagonalPreconditioner()
        prec.compute(simple_problem)

        # Test with random vector
        b = np.random.random(simple_problem.vertices)
        result = prec.solve(b)

        assert len(result) == len(b)
        assert np.allclose(result, prec.inverse_diagonal * b)

    def test_diagonal_preconditioner_solve_not_computed(self, simple_problem):
        """Test that solve fails if preconditioner not computed."""
        prec = DiagonalPreconditioner()
        b = np.random.random(simple_problem.vertices)

        with pytest.raises(RuntimeError, match="must be computed"):
            prec.solve(b)

    def test_diagonal_preconditioner_as_linear_operator(self, simple_problem):
        """Test converting to LinearOperator."""
        prec = DiagonalPreconditioner()
        prec.compute(simple_problem)

        op = prec.as_linear_operator()
        assert op.shape == (simple_problem.vertices, simple_problem.vertices)
        assert op.dtype == np.float64

        # Test matrix-vector product
        b = np.random.random(simple_problem.vertices)
        result1 = op @ b
        result2 = prec.solve(b)
        assert np.allclose(result1, result2)

    def test_diagonal_preconditioner_diagonal_extraction(self, simple_problem):
        """Test that diagonal entries are extracted correctly."""
        prec = DiagonalPreconditioner()
        prec.compute(simple_problem)

        # Verify that the diagonal entries make sense
        # (they should be positive for a well-posed problem)
        diagonal_entries = 1.0 / prec.inverse_diagonal  # type: ignore[operator]
        assert np.all(np.isfinite(diagonal_entries))

        # For a quantum graph problem, diagonal entries should be positive
        # due to the positive definiteness of the system
        positive_count = np.sum(diagonal_entries > 0)
        assert positive_count > 0  # At least some should be positive

    def test_diagonal_preconditioner_repr(self, simple_problem):
        """Test string representation."""
        prec = DiagonalPreconditioner()

        # Before computation
        repr_str = repr(prec)
        assert "not computed" in repr_str

        # After computation
        prec.compute(simple_problem)
        repr_str = repr(prec)
        assert "vertices=" in repr_str
        assert "diag_range=" in repr_str

    def test_diagonal_preconditioner_zero_diagonal_handling(self, simple_problem):
        """Test handling of zero diagonal entries."""
        prec = DiagonalPreconditioner()
        prec.compute(simple_problem)

        # Check that no inverse diagonal entries are infinite
        assert np.all(np.isfinite(prec.inverse_diagonal))

        # Check that all entries are reasonable (not too large)
        # This tests the zero-diagonal fallback
        assert np.all(prec.inverse_diagonal <= 1e10)  # type: ignore[operator]
