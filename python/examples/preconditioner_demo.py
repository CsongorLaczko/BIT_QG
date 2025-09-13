#!/usr/bin/env python3
"""
Example demonstrating quantum graph preconditioners with SciPy iterative solvers.

This script shows how to use the ported preconditioners with SciPy's conjugate
gradient solver, comparing performance and iterations.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from bit_qg import (
    DegreePreconditioner,
    MFQuantumGraph,
    PolynomialPreconditioner,
    QGEdge,
)


def create_example_system():
    """Create a simple quantum graph system for testing."""
    # Define edge functions
    def c_func(x: float) -> float:
        return 1.0  # constant coefficient

    def v_func(x: float) -> float:
        return x * x  # quadratic potential

    def f_func(x: float) -> float:
        return np.sin(np.pi * x)  # sine force

    # Create edges for a simple graph: 0 -> 1 -> 2
    edges = [
        QGEdge(out=0, in_=1, c=c_func, v=v_func, f=f_func),
        QGEdge(out=1, in_=2, c=c_func, v=v_func, f=f_func),
    ]

    # Create quantum graph
    qg = MFQuantumGraph(N=10, vertices=3, edges=edges)
    return qg


def solve_with_preconditioner(qg, preconditioner, rhs):
    """Solve system using CG with given preconditioner."""
    print(f"\nSolving with {type(preconditioner).__name__}:")

    # Compute preconditioner
    preconditioner.compute(qg)
    print(f"  Preconditioner size: {preconditioner.size}")

    # Create system matrix as LinearOperator
    def matvec(x):
        return qg @ x

    A = LinearOperator(shape=qg.shape, matvec=matvec, dtype=np.float64)

    # Get preconditioner as LinearOperator
    M = preconditioner.as_linear_operator()

    # Solve with conjugate gradient
    iteration_count = [0]

    def callback(x):
        iteration_count[0] += 1

    solution, info = cg(A, rhs, M=M, callback=callback, atol=1e-8, maxiter=100)

    print(f"  Iterations: {iteration_count[0]}")
    print(f"  Convergence info: {info}")
    print(f"  Solution norm: {np.linalg.norm(solution):.6f}")

    # Verify solution
    residual = A.matvec(solution) - rhs
    print(f"  Residual norm: {np.linalg.norm(residual):.2e}")

    return solution, iteration_count[0], info


def main():
    """Main example function."""
    print("Quantum Graph Preconditioner Example")
    print("=" * 40)

    # Create test system
    qg = create_example_system()
    print(f"System size: {qg.shape}")
    print(f"Graph vertices: {qg.vertices}")
    print(f"Graph edges: {len(qg.edges)}")

    # Create right-hand side
    rhs = np.ones(qg.shape[0])
    print(f"RHS norm: {np.linalg.norm(rhs):.6f}")

    # Test different preconditioners
    results = {}

    # Degree preconditioner
    degree_precond = DegreePreconditioner()
    sol1, iter1, info1 = solve_with_preconditioner(qg, degree_precond, rhs)
    results['Degree'] = (iter1, info1)

    # Polynomial preconditioner
    poly_precond = PolynomialPreconditioner()
    sol2, iter2, info2 = solve_with_preconditioner(qg, poly_precond, rhs)
    results['Polynomial'] = (iter2, info2)

    # No preconditioner (for comparison)
    print("\nSolving without preconditioner:")
    def matvec(x):
        return qg @ x

    A = LinearOperator(shape=qg.shape, matvec=matvec, dtype=np.float64)

    iteration_count = [0]
    def callback(x):
        iteration_count[0] += 1

    sol3, info3 = cg(A, rhs, callback=callback, atol=1e-8, maxiter=100)
    results['None'] = (iteration_count[0], info3)

    print(f"  Iterations: {iteration_count[0]}")
    print(f"  Convergence info: {info3}")
    print(f"  Solution norm: {np.linalg.norm(sol3):.6f}")

    residual = A.matvec(sol3) - rhs
    print(f"  Residual norm: {np.linalg.norm(residual):.2e}")

    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print("=" * 40)
    for name, (iters, info) in results.items():
        status = "Converged" if info == 0 else "Failed"
        print(f"{name:12}: {iters:3d} iterations, {status}")

    # Verify solutions are similar
    print("\nSolution differences:")
    print(f"  |sol_degree - sol_none|: {np.linalg.norm(sol1 - sol3):.2e}")
    print(f"  |sol_poly - sol_none|:   {np.linalg.norm(sol2 - sol3):.2e}")
    print(f"  |sol_degree - sol_poly|: {np.linalg.norm(sol1 - sol2):.2e}")


if __name__ == "__main__":
    main()
