#!/usr/bin/env python3
"""
Challenging Validation Test Cases
=================================

Create test scenarios that stress iterative solvers with:
1. Ill-conditioned matrices (high condition numbers)
2. Random ill-conditioned problems
3. Discontinuous coefficients
4. High aspect ratio problems
5. Nearly singular problems

Goal: Generate problems requiring 20-100+ iterations to test preconditioner effectiveness.
"""

import math
import time
import numpy as np
from pathlib import Path
from typing import Callable, Tuple, Dict, Any

# Add current directory to path to import our modules
import sys
sys.path.append(str(Path(__file__).parent))

def create_ill_conditioned_coefficients():
    """Create coefficient functions that result in ill-conditioned matrices."""
    
    # Test case 1: Extreme conductivity variations
    def extreme_conductivity(x: float) -> float:
        """Conductivity varies by factor of 10^6 across the domain."""
        return 1e-3 + 1e6 * x**10  # Very small at x=0, very large at x=1
    
    # Test case 2: Discontinuous conductivity
    def discontinuous_conductivity(x: float) -> float:
        """Jump discontinuity in conductivity."""
        return 1e-3 if x < 0.5 else 1e3  # 10^6 ratio
    
    # Test case 3: Oscillatory conductivity
    def oscillatory_conductivity(x: float) -> float:
        """Rapidly oscillating conductivity."""
        return 1.0 + 100.0 * math.sin(50 * math.pi * x)**2
    
    # Test case 4: Boundary layer conductivity
    def boundary_layer_conductivity(x: float) -> float:
        """Boundary layer with thin high-conductivity region."""
        eps = 0.01
        if x < eps or x > 1 - eps:
            return 1e6  # High conductivity in boundary layers
        return 1.0      # Normal conductivity in interior
    
    return {
        "extreme": extreme_conductivity,
        "discontinuous": discontinuous_conductivity,
        "oscillatory": oscillatory_conductivity,
        "boundary_layer": boundary_layer_conductivity
    }


def create_challenging_source_functions():
    """Create source functions that challenge the solvers."""
    
    def oscillatory_source(x: float) -> float:
        """Highly oscillatory source function."""
        return math.sin(100 * math.pi * x) * math.cos(50 * math.pi * x)
    
    def boundary_source(x: float) -> float:
        """Source concentrated near boundaries."""
        eps = 0.05
        if x < eps:
            return 1000.0 * math.exp(-((x - 0.02)**2) / (eps**2))
        elif x > 1 - eps:
            return 1000.0 * math.exp(-((x - 0.98)**2) / (eps**2))
        return 0.1
    
    def random_source(x: float) -> float:
        """Pseudo-random source (deterministic but irregular)."""
        # Use multiple sine waves with incommensurable frequencies
        return (math.sin(37 * math.pi * x) + 
                0.5 * math.sin(73 * math.pi * x) + 
                0.25 * math.sin(157 * math.pi * x))
    
    return {
        "oscillatory": oscillatory_source,
        "boundary": boundary_source,
        "random": random_source
    }


def test_challenging_scenarios():
    """Test various challenging scenarios."""
    from bit_qg.core import MFQuantumGraph
    from bit_qg.utils import GraphLoader
    from bit_qg.preconditioners import (
        DegreePreconditioner, DiagonalPreconditioner,
        PolynomialPreconditioner, NeumannNeumannPreconditioner
    )
    import scipy.sparse.linalg as spla
    
    print("🔥 Testing Challenging Scenarios for High Iteration Counts")
    print("=" * 65)
    
    # Get challenging coefficients and sources
    challenging_c = create_ill_conditioned_coefficients()
    challenging_f = create_challenging_source_functions()
    
    # Use larger graph and higher discretization
    graph_file = "dorogovtsev_goltsev_mendes_4.txt"
    log_n = 5  # N = 33 points per edge
    
    print(f"📊 Graph: {graph_file}")
    print(f"📐 Discretization: logN={log_n}, N={2**log_n - 1 + 2} points per edge")
    print()
    
    results = {}
    
    # Test combinations of challenging coefficients and sources
    test_cases = [
        ("extreme", "oscillatory", "Extreme conductivity + Oscillatory source"),
        ("discontinuous", "boundary", "Discontinuous conductivity + Boundary source"),
        ("oscillatory", "random", "Oscillatory conductivity + Random source"),
        ("boundary_layer", "oscillatory", "Boundary layer + Oscillatory source"),
    ]
    
    for c_name, f_name, description in test_cases:
        print(f"🧪 {description}")
        print("-" * 50)
        
        try:
            # Load graph with challenging functions
            loader = GraphLoader()
            graph_path = Path(__file__).parent.parent / "graphs" / graph_file
            edges, vertices = loader.load_from_adjacency_matrix(
                graph_path,
                c_func=challenging_c[c_name],
                v_func=lambda x: 0.05,  # Simple potential to isolate conductivity effects
                f_func=challenging_f[f_name]
            )
            
            N = 2**log_n - 1 + 2
            mfqg = MFQuantumGraph(N, vertices, edges)
            
            # Analyze the matrix properties
            A_dense = mfqg.AGG.toarray()
            cond_num = np.linalg.cond(A_dense)
            rhs_norm = np.linalg.norm(mfqg.bG)
            
            print(f"Matrix size: {mfqg.AGG.shape}")
            print(f"Condition number: {cond_num:.2e}")
            print(f"RHS norm: {rhs_norm:.6e}")
            
            if cond_num > 1e12:
                print("⚠️  Matrix is nearly singular - may not converge")
                continue
            elif cond_num > 1e6:
                print("🔥 Ill-conditioned matrix - expect many iterations")
            elif cond_num > 1e3:
                print("🌡️  Moderately conditioned - expect some iterations")
            else:
                print("✅ Well-conditioned - may converge quickly")
            
            # Test all preconditioners
            preconditioners = [
                ("Identity", None),
                ("Degree", DegreePreconditioner()),
                ("Diagonal", DiagonalPreconditioner()),
                ("Polynomial", PolynomialPreconditioner()),
                ("Neumann-Neumann", NeumannNeumannPreconditioner())
            ]
            
            case_results = {}
            
            for prec_name, preconditioner in preconditioners:
                print(f"\n📈 {prec_name} Preconditioner:")
                
                try:
                    if preconditioner is not None:
                        preconditioner.compute(mfqg)
                        M = preconditioner.as_linear_operator()
                    else:
                        M = None
                    
                    # Test CG with iteration callback to count iterations
                    iteration_count = [0]
                    
                    def iteration_callback(x):
                        iteration_count[0] += 1
                        if iteration_count[0] % 10 == 0:
                            residual = np.linalg.norm(mfqg.AGG @ x - mfqg.bG)
                            print(f"     Iteration {iteration_count[0]}: residual = {residual:.2e}")
                    
                    start_time = time.time()
                    solution, info = spla.cg(
                        mfqg.AGG, mfqg.bG, M=M,
                        rtol=1.49e-08, maxiter=1000, atol=0.0,
                        callback=iteration_callback
                    )
                    runtime = time.time() - start_time
                    
                    residual = np.linalg.norm(mfqg.AGG @ solution - mfqg.bG)
                    
                    print(f"   CG: {iteration_count[0]} iterations, info={info}")
                    print(f"   Runtime: {runtime:.6f}s")
                    print(f"   Final residual: {residual:.2e}")
                    
                    case_results[prec_name] = {
                        "iterations": iteration_count[0],
                        "runtime": runtime,
                        "residual": residual,
                        "converged": info == 0
                    }
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    case_results[prec_name] = {"error": str(e)}
            
            results[f"{c_name}_{f_name}"] = {
                "description": description,
                "condition_number": cond_num,
                "matrix_size": mfqg.AGG.shape,
                "rhs_norm": rhs_norm,
                "preconditioner_results": case_results
            }
            
            print()
            
        except Exception as e:
            print(f"❌ Failed to create test case: {e}")
            print()
    
    return results


def create_random_ill_conditioned_problems():
    """Create random problems with controlled condition numbers."""
    from bit_qg.core import MFQuantumGraph
    from bit_qg.utils import GraphLoader
    from bit_qg.preconditioners import DegreePreconditioner
    import scipy.sparse.linalg as spla
    
    print("🎲 Testing Random Ill-Conditioned Problems")
    print("=" * 45)
    
    # Load a standard graph
    loader = GraphLoader()
    graph_path = Path(__file__).parent.parent / "graphs" / "dorogovtsev_goltsev_mendes_2.txt"
    edges, vertices = loader.load_from_adjacency_matrix(graph_path)
    
    N = 17  # logN=4
    mfqg = MFQuantumGraph(N, vertices, edges)
    
    print(f"Base matrix size: {mfqg.AGG.shape}")
    
    # Create random problems with different condition numbers
    np.random.seed(42)  # Reproducible results
    
    condition_targets = [1e2, 1e4, 1e6, 1e8]
    
    for target_cond in condition_targets:
        print(f"\n🎯 Target condition number: {target_cond:.1e}")
        
        # Create ill-conditioned right-hand side
        # Method: Create RHS in direction of smallest eigenvector
        try:
            A = mfqg.AGG.toarray()
            
            # Get smallest eigenvalue and eigenvector
            eigenvals, eigenvecs = np.linalg.eigh(A)
            min_idx = np.argmin(eigenvals)
            max_idx = np.argmax(eigenvals)
            
            # Create RHS as combination of smallest and largest eigenvectors
            # to make the problem ill-conditioned
            alpha = 1.0 / math.sqrt(target_cond)
            rhs_ill = alpha * eigenvecs[:, min_idx] + eigenvecs[:, max_idx]
            rhs_ill = rhs_ill / np.linalg.norm(rhs_ill)  # Normalize
            
            # Verify the condition number
            actual_cond = np.linalg.cond(A)
            print(f"Actual matrix condition number: {actual_cond:.2e}")
            
            # Test with identity and degree preconditioner
            for prec_name, use_prec in [("Identity", False), ("Degree", True)]:
                print(f"\n   📈 {prec_name}:")
                
                if use_prec:
                    preconditioner = DegreePreconditioner()
                    preconditioner.compute(mfqg)
                    M = preconditioner.as_linear_operator()
                else:
                    M = None
                
                iteration_count = [0]
                
                def callback(x):
                    iteration_count[0] += 1
                
                start_time = time.time()
                solution, info = spla.cg(
                    mfqg.AGG, rhs_ill, M=M,
                    rtol=1.49e-08, maxiter=1000,
                    callback=callback
                )
                runtime = time.time() - start_time
                
                residual = np.linalg.norm(mfqg.AGG @ solution - rhs_ill)
                
                print(f"     Iterations: {iteration_count[0]}")
                print(f"     Runtime: {runtime:.6f}s")
                print(f"     Converged: {info == 0}")
                print(f"     Residual: {residual:.2e}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")


def summarize_results(results: Dict[str, Any]):
    """Summarize the challenging test results."""
    print("\n📋 Summary of Challenging Test Results")
    print("=" * 50)
    
    for case_name, case_data in results.items():
        print(f"\n🧪 {case_data['description']}")
        print(f"   Condition number: {case_data['condition_number']:.2e}")
        
        # Find the preconditioner with most iterations
        max_iterations = 0
        best_prec = "None"
        worst_prec = "None"
        min_iterations = float('inf')
        
        for prec_name, prec_results in case_data['preconditioner_results'].items():
            if 'iterations' in prec_results:
                iters = prec_results['iterations']
                if iters > max_iterations:
                    max_iterations = iters
                    worst_prec = prec_name
                if iters < min_iterations:
                    min_iterations = iters
                    best_prec = prec_name
        
        if max_iterations > 0:
            print(f"   Max iterations: {max_iterations} ({worst_prec})")
            print(f"   Min iterations: {min_iterations} ({best_prec})")
            if max_iterations >= 20:
                print("   ✅ Successfully created challenging problem!")
            elif max_iterations >= 5:
                print("   ⚠️  Moderately challenging problem")
            else:
                print("   ❌ Problem still too easy")
        else:
            print("   ❌ All preconditioners failed or converged instantly")


if __name__ == "__main__":
    # Test challenging scenarios
    results = test_challenging_scenarios()
    
    # Test random ill-conditioned problems
    create_random_ill_conditioned_problems()
    
    # Summarize results
    summarize_results(results)
    
    print("\n🎯 Goal: Find test cases with 20-100+ iterations to properly evaluate preconditioners")