"""
Quantum Graph Edge implementation.

This module contains the QGEdge class which represents an edge in a quantum graph
with associated coefficient, potential, and force functions.
"""

from typing import Protocol


class EdgeFunction(Protocol):
    """Protocol for edge functions that take a float and return a float."""

    def __call__(self, x: float) -> float:
        """Apply the function to input x."""
        ...


class QGEdge:
    """
    Represents an edge in a quantum graph with associated functions.

    This is the Python equivalent of the C++ QGEdge class. Instead of function
    pointers, it uses Python callable objects which can be functions, lambdas,
    or any callable object.

    Attributes:
        out: Output vertex index
        in_: Input vertex index (renamed from 'in' to avoid Python keyword)
        c: Coefficient function c(x)
        v: Potential function v(x)
        f: Force function f(x)
    """

    def __init__(
        self,
        out: int,
        in_: int,
        c: EdgeFunction,
        v: EdgeFunction,
        f: EdgeFunction,
    ) -> None:
        """
        Initialize a quantum graph edge.

        Args:
            out: Output vertex index
            in_: Input vertex index
            c: Coefficient function c(x)
            v: Potential function v(x)
            f: Force function f(x)
        """
        self.out = out
        self.in_ = in_  # Renamed from 'in' to avoid Python keyword
        self.c = c
        self.v = v
        self.f = f

    def __repr__(self) -> str:
        """String representation of the edge."""
        return f"QGEdge(out={self.out}, in_={self.in_})"

    def __eq__(self, other: object) -> bool:
        """
        Check equality based on vertex indices.

        Note: Functions are not compared as they may not be easily comparable.
        """
        if not isinstance(other, QGEdge):
            return NotImplemented
        return self.out == other.out and self.in_ == other.in_

    def __hash__(self) -> int:
        """Hash based on vertex indices."""
        return hash((self.out, self.in_))
