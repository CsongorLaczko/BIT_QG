#!/usr/bin/env python3
"""
Challenging Validation Framework
==============================

Create challenging test cases that actually stress iterative solvers
with meaningful iteration counts (10-50+ iterations).
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))


def create_extreme_conductivity(x: float) -> float:
    """Extreme conductivity variation that creates ill-conditioned matrices."""
    return 1e-3 + 1e6 * x**10  # Factor of 10^9 variation


def create_oscillatory_source(x: float) -> float:
    """Highly oscillatory source function."""
    return math.sin(100 * math.pi * x) * math.cos(50 * math.pi * x)


def create_discontinuous_conductivity(x: float) -> float:
    """Discontinuous conductivity with large jump."""
    return 1e-3 if x < 0.5 else 1e3  # Factor of 10^6 jump


def create_boundary_source(x: float) -> float:
    """Source concentrated near boundaries."""
    eps = 0.05
    if x < eps:
        return 1000.0 * math.exp(-((x - 0.02)**2) / (eps**2))
    elif x > 1 - eps:
        return 1000.0 * math.exp(-((x - 0.98)**2) / (eps**2))
    return 0.1


def run_challenging_test(
    graph_name: str,
    size: int,
    log_n: int,
    solver: str,
    c_func,
    f_func,
    test_name: str
):
    """Run a single challenging test case."""
    
    from bit_qg.core import MFQuantumGraph
    from bit_qg.utils import GraphLoader
    from bit_qg.preconditioners import (
        DegreePreconditioner, DiagonalPreconditioner,
        PolynomialPreconditioner, NeumannNeumannPreconditioner
    )
    import scipy.sparse.linalg as spla
    
    # Load graph with challenging functions
    root_dir = Path(__file__).parent.parent
    graphs_dir = root_dir / "graphs"
    
    loader = GraphLoader()
    graph_file = graphs_dir / f"{graph_name}_{size}.txt"
    edges, vertices = loader.load_from_adjacency_matrix(
        graph_file,
        c_func=c_func,
        v_func=lambda x: 0.05,  # Simple potential
        f_func=f_func
    )
    
    N = 2**log_n - 1 + 2
    mfqg = MFQuantumGraph(N, vertices, edges)
    
    # Analyze matrix properties
    A_dense = mfqg.AGG.toarray()
    condition_number = np.linalg.cond(A_dense)
    matrix_size = mfqg.AGG.shape
    rhs_norm = np.linalg.norm(mfqg.bG)
    
    print(f"\n🔥 Test: {test_name}")
    print(f"   Graph: {graph_name}_{size}, N={N}")
    print(f"   Condition number: {condition_number:.2e}")
    print(f"   Matrix size: {matrix_size}")
    print(f"   RHS norm: {rhs_norm:.2e}")
    print("   " + "="*60)
    
    results = {}
    
    # Test all preconditioners
    preconditioner_classes = {
        "identity": None,
        "degree": DegreePreconditioner,
        "diagonal": DiagonalPreconditioner,
        "polynomial": PolynomialPreconditioner,
        "neumann_neumann": NeumannNeumannPreconditioner
    }
    
    for prec_name, prec_class in preconditioner_classes.items():
        try:
            # Setup preconditioner
            if prec_class is not None:
                preconditioner = prec_class()
                preconditioner.compute(mfqg)
                M = preconditioner.as_linear_operator()
            else:
                M = None
            
            # Count iterations
            iteration_count = [0]
            
            def iteration_callback(x):
                iteration_count[0] += 1
            
            # Solve with timing
            start_time = time.time()
            if solver.lower() == "cg":
                solution, info = spla.cg(
                    mfqg.AGG, mfqg.bG, M=M,
                    rtol=1.49e-08, maxiter=1000, atol=0.0,
                    callback=iteration_callback
                )
            else:  # bicgstab
                solution, info = spla.bicgstab(
                    mfqg.AGG, mfqg.bG, M=M,
                    rtol=1.49e-08, maxiter=1000, atol=0.0,
                    callback=iteration_callback
                )
            runtime = time.time() - start_time
            
            # Calculate residual
            residual = np.linalg.norm(mfqg.AGG @ solution - mfqg.bG)
            
            results[prec_name] = {
                "iterations": iteration_count[0],
                "runtime": runtime,
                "residual": residual,
                "converged": info == 0,
                "info": info
            }
            
            # Print result
            status = "✅" if info == 0 else "❌"
            iterations_str = f"{iteration_count[0]:2d}" if iteration_count[0] > 0 else "❌"
            print(f"   {prec_name:15} {iterations_str} iter {status} (residual: {residual:.2e})")
            
        except Exception as e:
            results[prec_name] = {
                "iterations": -1,
                "runtime": 0.0,
                "residual": float('inf'),
                "converged": False,
                "error": str(e)
            }
            print(f"   {prec_name:15} ❌ ERROR: {str(e)[:50]}")
    
    return {
        "condition_number": condition_number,
        "matrix_size": matrix_size,
        "rhs_norm": rhs_norm,
        "results": results
    }


def main():
    """Run challenging validation tests."""
    print("🔥 Challenging Validation Framework")
    print("==================================")
    print("Goal: Create test cases with meaningful iteration counts")
    print("to properly evaluate preconditioner effectiveness.\n")
    
    # Test case configurations
    test_cases = [
        {
            "name": "extreme_oscillatory",
            "description": "Extreme conductivity + Oscillatory source", 
            "c_func": create_extreme_conductivity,
            "f_func": create_oscillatory_source
        },
        {
            "name": "discontinuous_boundary",
            "description": "Discontinuous conductivity + Boundary source",
            "c_func": create_discontinuous_conductivity, 
            "f_func": create_boundary_source
        }
    ]
    
    # Run tests
    all_results = {}
    
    for test_case in test_cases:
        print(f"\n🎯 Running: {test_case['description']}")
        
        # Test with different parameters
        for solver in ["cg", "bicgstab"]:
            for log_n in [4, 5]:  # N=17, 33
                test_name = f"{test_case['name']}_{solver}_logN{log_n}"
                
                result = run_challenging_test(
                    "dorogovtsev_goltsev_mendes", 4,  # Use larger graph
                    log_n, solver.upper(),
                    test_case["c_func"],
                    test_case["f_func"],
                    test_name
                )
                
                all_results[test_name] = result
    
    # Summary
    print("\n" + "="*70)
    print("📊 CHALLENGING VALIDATION SUMMARY")
    print("="*70)
    
    high_iteration_tests = []
    for test_name, result in all_results.items():
        successful_precs = []
        for prec_name, prec_result in result["results"].items():
            if prec_result["converged"] and prec_result["iterations"] >= 10:
                successful_precs.append((prec_name, prec_result["iterations"]))
        
        if successful_precs:
            high_iteration_tests.append((test_name, result["condition_number"], successful_precs))
    
    if high_iteration_tests:
        print(f"🎯 SUCCESS: Found {len(high_iteration_tests)} test cases with high iteration counts!")
        for test_name, condition_num, precs in high_iteration_tests:
            print(f"\n   {test_name}")
            print(f"   Condition: {condition_num:.2e}")
            for prec_name, iterations in precs:
                print(f"   - {prec_name}: {iterations} iterations")
    else:
        print("⚠️  No test cases achieved high iteration counts (≥10)")
        
        # Show best results
        best_iterations = {}
        for test_name, result in all_results.items():
            for prec_name, prec_result in result["results"].items():
                if prec_result["converged"]:
                    key = f"{test_name}_{prec_name}"
                    best_iterations[key] = prec_result["iterations"]
        
        if best_iterations:
            max_iter = max(best_iterations.values())
            best_test = max(best_iterations.items(), key=lambda x: x[1])
            print(f"   Best result: {best_test[0]} with {best_test[1]} iterations")
    
    print("\n✅ Challenging validation complete!")


if __name__ == "__main__":
    main()