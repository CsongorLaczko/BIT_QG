"""
Custom preconditioners for iterative linear solvers.

This module provides various preconditioners for quantum graph finite element
systems, compatible with SciPy's iterative solvers.
"""

from .base import PreconditionerBase
from .degree import DegreePreconditioner
from .polynomial import PolynomialPreconditioner

__all__ = [
    "PreconditionerBase",
    "DegreePreconditioner",
    "PolynomialPreconditioner",
]
