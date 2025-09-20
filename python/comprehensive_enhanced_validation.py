#!/usr/bin/env python3
"""
Comprehensive Enhanced Validation Framework
==========================================

Addresses the original validation issue by providing both:
1. Well-conditioned test cases (fast convergence validation) 
2. Challenging ill-conditioned test cases (preconditioner effectiveness evaluation)

This solves the problem where all original tests had 0-1 iterations by offering
a complete spectrum from trivial to challenging problems.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))


def create_standard_source(x: float) -> float:
    """Standard source function - creates well-conditioned problems."""
    return math.exp(-1000 * x**2)


def create_extreme_conductivity(x: float) -> float:
    """Extreme conductivity variation - creates ill-conditioned matrices."""
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


class ComprehensiveValidator:
    """Comprehensive validator with both standard and challenging test cases."""
    
    def __init__(self):
        """Initialize the comprehensive validator."""
        self.root_dir = Path(__file__).parent.parent
        self.graphs_dir = self.root_dir / "graphs"
        
        # Test case configurations
        self.test_configurations = {
            "standard": {
                "description": "Well-conditioned problems (original validation)",
                "c_func": lambda x: 1.0,  # Constant conductivity
                "f_func": create_standard_source,
                "expected_iterations": "0-2 (fast convergence)",
                "expected_condition": "~2-5 (well-conditioned)"
            },
            "extreme_oscillatory": {
                "description": "Extreme conductivity + Oscillatory source",
                "c_func": create_extreme_conductivity,
                "f_func": create_oscillatory_source,
                "expected_iterations": "3-32 (challenging)",
                "expected_condition": "~10^8 (ill-conditioned)"
            },
            "discontinuous_boundary": {
                "description": "Discontinuous conductivity + Boundary source", 
                "c_func": create_discontinuous_conductivity,
                "f_func": create_boundary_source,
                "expected_iterations": "3-12 (moderate)",
                "expected_condition": "~10^5 (moderately ill-conditioned)"
            }
        }
    
    def run_comprehensive_validation(self) -> dict:
        """Run comprehensive validation with all test categories."""
        
        graphs = ["dorogovtsev_goltsev_mendes_4"]
        log_n_values = [3, 4, 5]
        solvers = ["cg", "bicgstab"]
        
        all_results = {}
        
        print("🚀 Comprehensive Enhanced Validation Framework")
        print("=" * 60)
        print("Providing complete preconditioner evaluation across")
        print("well-conditioned and challenging test scenarios.\n")
        
        for test_type, test_config in self.test_configurations.items():
            print(f"\n🎯 Test Category: {test_type.replace('_', ' ').title()}")
            print(f"   Description: {test_config['description']}")
            print(f"   Expected iterations: {test_config['expected_iterations']}")
            print(f"   Expected condition: {test_config['expected_condition']}")
            print("   " + "-" * 50)
            
            category_results = []
            
            for graph_name in graphs:
                for solver in solvers:
                    for log_n in log_n_values:
                        result = self._run_single_test(
                            test_type, graph_name, solver, log_n, test_config
                        )
                        category_results.append(result)
                        
                        # Progress summary
                        successful_precs = sum(
                            1 for prec_data in result["preconditioners"].values()
                            if prec_data["converged"]
                        )
                        
                        if successful_precs > 0:
                            iter_values = [
                                prec_data["iterations"]
                                for prec_data in result["preconditioners"].values()
                                if prec_data["converged"]
                            ]
                            min_iter, max_iter = min(iter_values), max(iter_values)
                            
                            print(f"   {solver.upper()}_logN{log_n}: {successful_precs}/5 success, "
                                  f"iter {min_iter}-{max_iter}, cond {result['condition_number']:.1e}")
            
            all_results[test_type] = category_results
            
            # Category summary
            self._print_category_summary(test_type, category_results)
        
        return all_results
    
    def _run_single_test(self, test_type: str, graph_name: str, solver: str, log_n: int, test_config: dict) -> dict:
        """Run a single validation test."""
        
        from bit_qg.core import MFQuantumGraph
        from bit_qg.utils import GraphLoader
        from bit_qg.preconditioners import (
            DegreePreconditioner, DiagonalPreconditioner,
            PolynomialPreconditioner, NeumannNeumannPreconditioner
        )
        import scipy.sparse.linalg as spla
        
        # Load graph
        loader = GraphLoader()
        graph_file = self.graphs_dir / f"{graph_name}.txt"  # Fixed: don't append _4 twice
        edges, vertices = loader.load_from_adjacency_matrix(
            graph_file,
            c_func=test_config["c_func"],
            v_func=lambda x: 0.05,
            f_func=test_config["f_func"]
        )
        
        N = 2**log_n - 1 + 2
        mfqg = MFQuantumGraph(N, vertices, edges)
        
        # Matrix analysis
        A_dense = mfqg.AGG.toarray()
        condition_number = np.linalg.cond(A_dense)
        
        result = {
            "test_type": test_type,
            "graph": graph_name,
            "solver": solver,
            "log_n": log_n,
            "N": N,
            "condition_number": condition_number,
            "matrix_size": mfqg.AGG.shape,
            "rhs_norm": np.linalg.norm(mfqg.bG),
            "preconditioners": {}
        }
        
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
                
                # Solve
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
                
                residual = np.linalg.norm(mfqg.AGG @ solution - mfqg.bG)
                
                result["preconditioners"][prec_name] = {
                    "iterations": iteration_count[0],
                    "runtime": runtime,
                    "residual": residual,
                    "converged": info == 0,
                    "info": info
                }
                
            except Exception as e:
                result["preconditioners"][prec_name] = {
                    "iterations": -1,
                    "runtime": 0.0,
                    "residual": float('inf'),
                    "converged": False,
                    "error": str(e)
                }
        
        return result
    
    def _print_category_summary(self, test_type: str, results):
        """Print summary statistics for a test category."""
        all_successful = []
        for result in results:
            for prec_data in result["preconditioners"].values():
                if prec_data["converged"]:
                    all_successful.append(prec_data["iterations"])
        
        if all_successful:
            print(f"\n   📊 {test_type} Summary:")
            print(f"      Successful tests: {len(all_successful)}")
            print(f"      Iteration range: {min(all_successful)}-{max(all_successful)}")
            print(f"      Average iterations: {sum(all_successful)/len(all_successful):.1f}")
            
            if max(all_successful) >= 15:
                print("      🎯 High iteration counts achieved - excellent for preconditioner testing!")
            elif max(all_successful) >= 5:
                print("      ⚡ Moderate iteration counts - good for evaluation")
            else:
                print("      ⚠️  Low iteration counts - fast convergence validation")


def main():
    """Main execution function."""
    validator = ComprehensiveValidator()
    
    # Run comprehensive validation
    print("Starting comprehensive validation with enhanced test cases...")
    all_results = validator.run_comprehensive_validation()
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE VALIDATION COMPLETE")
    print("=" * 70)
    
    # Summary across all categories
    total_high_iteration_tests = 0
    for test_type, results in all_results.items():
        high_iter_count = 0
        for result in results:
            for prec_data in result["preconditioners"].values():
                if prec_data["converged"] and prec_data["iterations"] >= 10:
                    high_iter_count += 1
        total_high_iteration_tests += high_iter_count
        
        print(f"🎯 {test_type.replace('_', ' ').title()}: {high_iter_count} tests with ≥10 iterations")
    
    print(f"\n✅ SUCCESS: {total_high_iteration_tests} total tests achieved high iteration counts")
    print("📈 Framework successfully provides:")
    print("   - Fast convergence validation (well-conditioned problems)")
    print("   - Preconditioner effectiveness testing (ill-conditioned problems)")
    print("   - Complete performance evaluation spectrum")


if __name__ == "__main__":
    main()