"""
Unit tests for graph I/O utilities.
"""

import tempfile
from pathlib import Path

import pytest

from bit_qg.core import QGEdge
from bit_qg.utils import GraphLoader, load_quantum_graph


class TestGraphLoader:
    """Test the GraphLoader class functionality."""

    def test_graph_loader_creation(self):
        """Test that GraphLoader can be created."""
        loader = GraphLoader()
        assert loader is not None

    def test_default_functions(self):
        """Test that default coefficient functions work correctly."""
        loader = GraphLoader()

        # Test default c function
        c_val = loader.default_c_function(0.5)
        assert isinstance(c_val, float)
        assert c_val > 0

        # Test default v function
        v_val = loader.default_v_function(0.5)
        assert isinstance(v_val, float)
        assert v_val >= 0

        # Test default f function
        f_val = loader.default_f_function(0.0)
        assert isinstance(f_val, float)
        assert f_val > 0

    def test_create_simple_path(self):
        """Test creating a simple path graph."""
        loader = GraphLoader()

        # Test valid path
        edges, vertices = loader.create_simple_path(3)
        assert vertices == 3
        assert len(edges) == 2

        # Check edge connections
        assert edges[0].out == 0 and edges[0].in_ == 1
        assert edges[1].out == 1 and edges[1].in_ == 2

        # Test invalid path
        with pytest.raises(ValueError, match="Path must have at least 2 vertices"):
            loader.create_simple_path(1)

    def test_create_complete_graph(self):
        """Test creating a complete graph."""
        loader = GraphLoader()

        # Test complete graph with 3 vertices
        edges, vertices = loader.create_complete_graph(3)
        assert vertices == 3
        assert len(edges) == 3  # Complete graph: n*(n-1)/2 edges

        # Check all edges exist
        edge_pairs = {(edge.out, edge.in_) for edge in edges}
        expected_pairs = {(0, 1), (0, 2), (1, 2)}
        assert edge_pairs == expected_pairs

        # Test invalid input
        with pytest.raises(
            ValueError, match="Complete graph must have at least 2 vertices"
        ):
            loader.create_complete_graph(1)

    def test_load_from_adjacency_matrix_simple(self):
        """Test loading from a simple adjacency matrix file."""
        loader = GraphLoader()

        # Create a temporary adjacency matrix file
        matrix_content = "0,1,1\n0,0,1\n0,0,0\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(matrix_content)
            temp_path = f.name

        try:
            edges, vertices = loader.load_from_adjacency_matrix(temp_path)

            assert vertices == 3
            assert len(edges) == 3  # Three edges: 0->1, 0->2, and 1->2

            # Check edge connections
            edge_pairs = {(edge.out, edge.in_) for edge in edges}
            expected_pairs = {(0, 1), (0, 2), (1, 2)}
            assert edge_pairs == expected_pairs

            # Check that edges have the right functions
            for edge in edges:
                assert callable(edge.c)
                assert callable(edge.v)
                assert callable(edge.f)

        finally:
            Path(temp_path).unlink()

    def test_load_from_adjacency_matrix_errors(self):
        """Test error handling in adjacency matrix loading."""
        loader = GraphLoader()

        # Test file not found
        with pytest.raises(FileNotFoundError):
            loader.load_from_adjacency_matrix("nonexistent_file.txt")

        # Test invalid matrix format
        invalid_content = "1,2,x\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(invalid_content)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid adjacency matrix format"):
                loader.load_from_adjacency_matrix(temp_path)
        finally:
            Path(temp_path).unlink()

        # Test non-square matrix
        nonsquare_content = "0,1\n0,0,1\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(nonsquare_content)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid adjacency matrix format"):
                loader.load_from_adjacency_matrix(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_from_adjacency_matrix_custom_functions(self):
        """Test loading with custom coefficient functions."""
        loader = GraphLoader()

        def custom_c(x):
            return 2.0

        def custom_v(x):
            return 0.1

        def custom_f(x):
            return 0.5

        matrix_content = "0,1\n0,0\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(matrix_content)
            temp_path = f.name

        try:
            edges, vertices = loader.load_from_adjacency_matrix(
                temp_path, c_func=custom_c, v_func=custom_v, f_func=custom_f
            )

            assert len(edges) == 1
            edge = edges[0]

            # Test that custom functions are used
            assert edge.c(0.5) == 2.0
            assert edge.v(0.5) == 0.1
            assert edge.f(0.5) == 0.5

        finally:
            Path(temp_path).unlink()

    def test_load_from_graph_file_integration(self):
        """Test loading from the actual graph files in the repository."""
        loader = GraphLoader()

        # Test with a known graph file (if it exists)
        try:
            # Try to load a small test graph
            edges, vertices = loader.load_from_graph_file(
                "dorogovtsev_goltsev_mendes_1"
            )

            # This should be a 3-vertex graph with 3 edges
            assert vertices == 3
            assert len(edges) == 3

            # Check that edges are valid QGEdge objects
            for edge in edges:
                assert isinstance(edge, QGEdge)
                assert callable(edge.c)
                assert callable(edge.v)
                assert callable(edge.f)

        except FileNotFoundError:
            # Skip test if graphs directory is not available
            pytest.skip("Graph files not available for testing")


class TestLoadQuantumGraphFunction:
    """Test the convenience load_quantum_graph function."""

    def test_load_quantum_graph_function(self):
        """Test the convenience function for loading graphs."""
        try:
            edges, vertices = load_quantum_graph("dorogovtsev_goltsev_mendes_1")

            assert isinstance(edges, list)
            assert isinstance(vertices, int)
            assert vertices > 0
            assert len(edges) >= 0

            # Check that edges are valid QGEdge objects
            for edge in edges:
                assert isinstance(edge, QGEdge)

        except FileNotFoundError:
            # Skip test if graphs directory is not available
            pytest.skip("Graph files not available for testing")
