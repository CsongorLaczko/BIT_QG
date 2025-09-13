"""
Pytest configuration and fixtures for BIT_QG tests.
"""

from collections.abc import Callable

import numpy as np
import pytest


@pytest.fixture
def simple_graph_data() -> tuple[int, list[tuple[int, int]]]:
    """
    Provides a simple test graph with 3 vertices and 2 edges.

    Returns:
        (vertices, edges) where edges are (out, in) tuples
    """
    vertices = 3
    edges = [(0, 1), (1, 2)]
    return vertices, edges


@pytest.fixture
def test_functions() -> tuple[Callable[[float], float], ...]:
    """
    Provides simple test functions for quantum graph edges.

    Returns:
        (c_func, v_func, f_func) - coefficient, potential, and force functions
    """
    def c_func(x: float) -> float:
        return 1.0  # constant coefficient

    def v_func(x: float) -> float:
        return x * x  # quadratic potential

    def f_func(x: float) -> float:
        return np.sin(np.pi * x)  # sine force

    return c_func, v_func, f_func


@pytest.fixture
def tolerance() -> float:
    """Standard numerical tolerance for test comparisons."""
    return 1e-12


@pytest.fixture
def discretization_size() -> int:
    """Standard discretization size for testing."""
    return 10
