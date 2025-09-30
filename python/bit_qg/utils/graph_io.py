"""
Graph loading and I/O utilities for quantum graphs.

This module provides utilities for loading quantum graphs from files and converting
between different graph representations, corresponding to the Example class in the
C++ implementation.
"""

import csv
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import cast

from ..core import QGEdge, EdgeFunction


class GraphLoader:
    """
    Utility class for loading quantum graphs from various file formats.

    This class provides functionality similar to the C++ Example class, loading
    graphs from adjacency matrix files and constructing QGEdge lists with
    predefined coefficient functions.
    """

    def __init__(self):
        """Initialize the graph loader."""
        pass

    @staticmethod
    def default_c_function(x: float) -> float:
        """
        Default conductivity function: c(x) = 1/(1+exp(-25*(x-0.5))) + 1.

        This matches the C++ implementation's default coefficient function.

        Args:
            x: Position along the edge [0, 1]

        Returns:
            Conductivity value
        """
        return 1.0 / (1.0 + math.exp(-25 * (x - 0.5))) + 1.0

    @staticmethod
    def default_v_function(x: float) -> float:
        """
        Default potential function: v(x) = 0.05/0.2^2 * |x-0.5-0.2|^2 + 0.05.

        This matches the C++ implementation's default potential function.

        Args:
            x: Position along the edge [0, 1]

        Returns:
            Potential value
        """
        return 0.05 / (0.2**2) * (abs(x - 0.5) - 0.2) ** 2 + 0.05

    @staticmethod
    def default_f_function(x: float) -> float:
        """
        Default source function: f(x) = exp(-(x-0.0)^2 * 250 * 4).

        This matches the C++ implementation's default source function.

        Args:
            x: Position along the edge [0, 1]

        Returns:
            Source value
        """
        return 1.0 * math.exp(-((x - 0.0) ** 2) * 250 * 4)

    def load_from_adjacency_matrix(
        self,
        filepath: str | Path,
        c_func: Callable[[float], float] | None = None,
        v_func: Callable[[float], float] | None = None,
        f_func: Callable[[float], float] | None = None,
    ) -> tuple[list[QGEdge], int]:
        """
        Load a quantum graph from an adjacency matrix file.

        The file should contain a comma-separated adjacency matrix where
        entry (i,j) = 1 indicates an edge from vertex i to vertex j.
        Only the upper triangular part is used (undirected graphs).

        Args:
            filepath: Path to the adjacency matrix file
            c_func: Conductivity function (uses default if None)
            v_func: Potential function (uses default if None)
            f_func: Source function (uses default if None)

        Returns:
            Tuple of (edges, vertices) where edges is a list of QGEdge objects
            and vertices is the number of vertices

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Graph file not found: {filepath}")

        # Use default functions if not provided
        if c_func is None:
            c_func = self.default_c_function
        if v_func is None:
            v_func = self.default_v_function
        if f_func is None:
            f_func = self.default_f_function

        edges = []
        vertices = 0

        try:
            with open(filepath, newline="") as f:
                reader = csv.reader(f)
                adjacency_matrix = []

                for row in reader:
                    # Convert string entries to integers
                    matrix_row = [int(x.strip()) for x in row if x.strip()]
                    if matrix_row:  # Skip empty rows
                        adjacency_matrix.append(matrix_row)

                vertices = len(adjacency_matrix)

                if vertices == 0:
                    raise ValueError("Empty adjacency matrix")

                # Validate matrix is square
                for i, row in enumerate(adjacency_matrix):
                    if len(row) != vertices:
                        raise ValueError(
                            f"Row {i} has {len(row)} entries, expected {vertices}"
                        )

                # Extract edges from upper triangular matrix
                for i in range(vertices):
                    for j in range(i + 1, vertices):  # Only upper triangular
                        if adjacency_matrix[i][j] == 1:
                            # Create edge from vertex i to vertex j
                            edge = QGEdge(out=i, in_=j, c=cast(EdgeFunction, c_func), v=cast(EdgeFunction, v_func), f=cast(EdgeFunction, f_func))
                            edges.append(edge)

        except (ValueError, IndexError) as e:
            logging.error(
                f"FALLBACK: Error loading graph file - File: {filepath}, Error: {e}, "
                f"Matrix size: {len(adjacency_matrix) if 'adjacency_matrix' in locals() else 'unknown'}"
            )
            raise ValueError(
                f"Invalid adjacency matrix format in {filepath}: {e}"
            ) from e

        return edges, vertices

    def load_from_graph_file(
        self, graph_name: str, graphs_dir: str | Path | None = None
    ) -> tuple[list[QGEdge], int]:
        """
        Load a quantum graph from the standard graphs directory.

        This method provides the same interface as the C++ Example constructor
        that takes a filename string.

        Args:
            graph_name: Name of the graph file (without .txt extension)
            graphs_dir: Directory containing graph files (defaults to ../graphs)

        Returns:
            Tuple of (edges, vertices) where edges is a list of QGEdge objects
            and vertices is the number of vertices
        """
        if graphs_dir is None:
            # Default to ../graphs relative to this file
            current_dir = Path(__file__).parent
            graphs_dir = current_dir.parent.parent.parent / "graphs"

        graphs_dir = Path(graphs_dir)
        filepath = graphs_dir / f"{graph_name}.txt"

        return self.load_from_adjacency_matrix(filepath)

    def create_simple_path(self, num_vertices: int) -> tuple[list[QGEdge], int]:
        """
        Create a simple path graph with default coefficient functions.

        Args:
            num_vertices: Number of vertices in the path

        Returns:
            Tuple of (edges, vertices) for a path graph 0-1-2-...-n
        """
        if num_vertices < 2:
            raise ValueError("Path must have at least 2 vertices")

        edges = []
        for i in range(num_vertices - 1):
            edge = QGEdge(
                out=i,
                in_=i + 1,
                c=self.default_c_function,
                v=self.default_v_function,
                f=self.default_f_function,
            )
            edges.append(edge)

        return edges, num_vertices

    def create_complete_graph(self, num_vertices: int) -> tuple[list[QGEdge], int]:
        """
        Create a complete graph with default coefficient functions.

        Args:
            num_vertices: Number of vertices

        Returns:
            Tuple of (edges, vertices) for a complete graph
        """
        if num_vertices < 2:
            raise ValueError("Complete graph must have at least 2 vertices")

        edges = []
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                edge = QGEdge(
                    out=i,
                    in_=j,
                    c=self.default_c_function,
                    v=self.default_v_function,
                    f=self.default_f_function,
                )
                edges.append(edge)

        return edges, num_vertices


def load_quantum_graph(
    graph_name: str, graphs_dir: str | Path | None = None
) -> tuple[list[QGEdge], int]:
    """
    Convenience function to load a quantum graph from file.

    This provides a simple interface similar to the C++ Example constructor.

    Args:
        graph_name: Name of the graph file (without .txt extension)
        graphs_dir: Directory containing graph files (defaults to ../graphs)

    Returns:
        Tuple of (edges, vertices) where edges is a list of QGEdge objects
        and vertices is the number of vertices
    """
    loader = GraphLoader()
    return loader.load_from_graph_file(graph_name, graphs_dir)
