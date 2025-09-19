#!/usr/bin/env python3
"""
Debug script to investigate C++ vs Python numerical differences.

This script analyzes the key differences:
1. Tolerance settings
2. Iteration counting
3. Initial guess differences
4. Matrix assembly precision
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from bit_qg.core.mf_quantum_graph import MFQuantumGraph
from bit_qg.utils import GraphLoader


def debug_solver_behavior():
    """Debug the exact solver behavior differences."""
    print("=== C++ vs Python Numerical Differences Investigation ===\n")
    
    # Load same test case as C++ run
    loader = GraphLoader()
    edges, vertices = loader.load_from_graph_file("dorogovtsev_goltsev_mendes_2")
    
    # Use same parameters as C++ test
    log_n = 4
    N = 2**log_n - 1 + 2  # C++ formula: pow(2, logN) - 1 + 2
    print("Graph: dorogovtsev_goltsev_mendes_2")
    print(f"Vertices: {vertices}")
    print(f"Edges: {len(edges)}")
    print(f"N (discretization): {N}")
    print()
    
    # Create the problem
    problem = MFQuantumGraph(N, vertices, edges)
    
    print(f"Matrix shape: {problem.shape}")
    print(f"RHS norm: {np.linalg.norm(problem.bG)}")
    print()
    
    # Check tolerances
    cpp_tolerance = np.sqrt(2.2204e-16)
    print(f"C++ tolerance: {cpp_tolerance:.2e}")
    print(f"Machine epsilon: {np.finfo(float).eps:.2e}")
    print()
    
    # Test identity preconditioner first (like C++ Vanilla)
    print("=== Identity Preconditioner Test ===")
    
    # Create LinearOperator for the problem
    A_op = LinearOperator(problem.shape, matvec=problem.solve, dtype=np.float64)
    
    # Test with different initial guesses
    zero_guess = np.zeros_like(problem.bG)
    random_guess = np.random.rand(len(problem.bG)) * 1e-6
    
    print("1. Zero initial guess:")
    solution1, info1 = cg(
        A_op, 
        problem.bG, 
        x0=zero_guess,
        rtol=cpp_tolerance,
        atol=0.0,
        maxiter=1000
    )
    residual1 = problem.bG - problem.solve(solution1)
    print(f"   Info: {info1}, Residual norm: {np.linalg.norm(residual1):.2e}")
    
    print("2. Random initial guess:")
    solution2, info2 = cg(
        A_op, 
        problem.bG, 
        x0=random_guess,
        rtol=cpp_tolerance,
        atol=0.0,
        maxiter=1000
    )
    residual2 = problem.bG - problem.solve(solution2)
    print(f"   Info: {info2}, Residual norm: {np.linalg.norm(residual2):.2e}")
    
    # Check if the problem is trivial (already solved)
    print("\n3. Testing if problem is trivial:")
    initial_residual = np.linalg.norm(problem.bG - problem.solve(zero_guess))
    print(f"   Initial residual with zero guess: {initial_residual:.2e}")
    
    if initial_residual < cpp_tolerance:
        print("   *** PROBLEM IS TRIVIAL - Zero is already the solution! ***")
        print("   This explains why C++ shows 0 iterations.")
    
    # Test with much looser tolerance
    print("\n4. Testing with looser tolerance (1e-6):")
    solution3, info3 = cg(
        A_op, 
        problem.bG, 
        x0=zero_guess,
        rtol=1e-6,
        atol=0.0,
        maxiter=1000
    )
    residual3 = problem.bG - problem.solve(solution3)
    print(f"   Info: {info3}, Residual norm: {np.linalg.norm(residual3):.2e}")
    
    # Check matrix properties
    print("\n=== Matrix Analysis ===")
    print(f"Matrix type: {type(problem)}")
    print(f"bG (RHS) min/max: {np.min(problem.bG):.2e} / {np.max(problem.bG):.2e}")
    
    # Try direct solution check
    direct_solve = problem.solve(problem.bG)
    direct_residual = problem.bG - problem.solve(direct_solve)
    print(f"Direct solve residual: {np.linalg.norm(direct_residual):.2e}")
    
    print("\n=== Analysis Summary ===")
    print("Key findings:")
    if initial_residual < cpp_tolerance:
        print("- The linear system Ax = b has x = 0 as solution (trivial case)")
        print("- C++ Eigen detects this immediately (0 iterations)")
        print("- Python SciPy still performs 1 iteration before detecting convergence")
        print("- Both results are mathematically correct")
    else:
        print("- Need further investigation of solver differences")


if __name__ == "__main__":
    debug_solver_behavior()