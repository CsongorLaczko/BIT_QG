"""
Unit tests for core quantum graph data structures.
"""

import numpy as np

from bit_qg.core import MFQuantumGraph, QGEdge


class TestQGEdge:
    """Test cases for QGEdge class."""

    def test_qgedge_creation(self):
        """Test basic QGEdge creation."""

        def c_func(x: float) -> float:
            return 1.0

        def v_func(x: float) -> float:
            return x * x

        def f_func(x: float) -> float:
            return np.sin(np.pi * x)

        edge = QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func)

        assert edge.out == 0
        assert edge.in_ == 1
        assert edge.c(0.5) == 1.0
        assert edge.v(0.5) == 0.25
        assert abs(edge.f(0.5) - 1.0) < 1e-10  # sin(π/2) = 1

    def test_qgedge_equality(self):
        """Test QGEdge equality comparison."""

        def c_func(x: float) -> float:
            return 1.0

        def v_func(x: float) -> float:
            return x

        def f_func(x: float) -> float:
            return 0.0

        edge1 = QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func)
        edge2 = QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func)
        edge3 = QGEdge(out=1, in_=0, c=c_func, v=v_func, f=f_func)

        assert edge1 == edge2
        assert edge1 != edge3

    def test_qgedge_equality_with_non_edge(self):
        """Test QGEdge equality with non-QGEdge objects."""

        def dummy_func(x: float) -> float:
            return 0.0

        edge = QGEdge(out=0, in_=1, c=dummy_func, v=dummy_func, f=dummy_func)

        # Should return NotImplemented when comparing with non-QGEdge
        assert edge.__eq__("not an edge") == NotImplemented
        assert edge.__eq__(42) == NotImplemented
        assert edge.__eq__(None) == NotImplemented

    def test_qgedge_hash(self):
        """Test QGEdge hashing for use in sets and dictionaries."""

        def dummy_func(x: float) -> float:
            return 0.0

        edge1 = QGEdge(out=0, in_=1, c=dummy_func, v=dummy_func, f=dummy_func)
        edge2 = QGEdge(out=0, in_=1, c=dummy_func, v=dummy_func, f=dummy_func)
        edge3 = QGEdge(out=1, in_=0, c=dummy_func, v=dummy_func, f=dummy_func)

        # Same edges should have same hash
        assert hash(edge1) == hash(edge2)
        # Different edges should have different hash (very likely)
        assert hash(edge1) != hash(edge3)

        # Should be usable in sets and dictionaries
        edge_set = {edge1, edge2, edge3}
        assert len(edge_set) == 2  # edge1 and edge2 are equal, so only 2 unique

        edge_dict = {edge1: "value1", edge3: "value3"}
        assert len(edge_dict) == 2
        assert edge_dict[edge2] == "value1"  # edge2 == edge1, so same key

    def test_qgedge_repr(self):
        """Test QGEdge string representation."""

        def dummy_func(x: float) -> float:
            return 0.0

        edge = QGEdge(out=2, in_=3, c=dummy_func, v=dummy_func, f=dummy_func)
        assert "QGEdge(out=2, in_=3)" in repr(edge)


class TestMFQuantumGraph:
    """Test cases for MFQuantumGraph class."""

    def test_simple_graph_creation(self):
        """Test creation of a simple 2-vertex, 1-edge graph."""

        def c_func(x: float) -> float:
            return 1.0  # constant coefficient

        def v_func(x: float) -> float:
            return 0.0  # zero potential

        def f_func(x: float) -> float:
            return 1.0  # constant force

        edge = QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func)
        qg = MFQuantumGraph(N=5, vertices=2, edges=[edge])

        assert qg.N == 5
        assert qg.vertices == 2
        assert len(qg.edges) == 1
        assert qg.vertex_weights == [1, 1]  # Each vertex has degree 1

        # Check matrix dimensions
        assert qg.AII.shape == (3, 3)  # (N-2) interior points per edge
        assert qg.AIG.shape == (3, 2)  # Interior to graph coupling
        assert qg.AGG.shape == (2, 2)  # Graph vertices

        # Check that solve method works
        rhs = np.ones(2)
        solution = qg.solve(rhs)
        assert solution.shape == (2,)

    def test_shape_property(self):
        """Test the shape property."""

        def dummy_func(x: float) -> float:
            return 1.0

        edge = QGEdge(out=0, in_=1, c=dummy_func, v=dummy_func, f=dummy_func)
        qg = MFQuantumGraph(N=10, vertices=3, edges=[edge])

        assert qg.shape == (3, 3)

    def test_matmul_operator(self):
        """Test the @ operator for matrix-vector multiplication."""

        def dummy_func(x: float) -> float:
            return 1.0

        edge = QGEdge(out=0, in_=1, c=dummy_func, v=dummy_func, f=dummy_func)
        qg = MFQuantumGraph(N=5, vertices=2, edges=[edge])

        rhs = np.array([1.0, 2.0])
        result1 = qg @ rhs
        result2 = qg.solve(rhs)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_repr(self):
        """Test string representation."""

        def dummy_func(x: float) -> float:
            return 1.0

        edge = QGEdge(out=0, in_=1, c=dummy_func, v=dummy_func, f=dummy_func)
        qg = MFQuantumGraph(N=5, vertices=2, edges=[edge])

        repr_str = repr(qg)
        assert "MFQuantumGraph" in repr_str
        assert "N=5" in repr_str
        assert "vertices=2" in repr_str
        assert "edges=1" in repr_str
