"""
BIT_QG: Quantum Graph Finite Element Methods

A Python library for quantum graph computations using finite element methods.
This is a port of the original C++ implementation using Eigen to Python using
SciPy/NumPy with future PyTorch integration for neural networks.
"""

__version__ = "0.1.0"
__author__ = "BIT_QG Team"

from .core import MFQuantumGraph, QGEdge
from .preconditioners import (
    DegreePreconditioner,
    PolynomialPreconditioner,
    PreconditionerBase,
)

__all__ = [
    "QGEdge",
    "MFQuantumGraph",
    "PreconditionerBase",
    "DegreePreconditioner",
    "PolynomialPreconditioner",
]
