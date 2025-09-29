#!/usr/bin/env python3
"""
Generate correct reference results for regression testing.

This script generates baseline data with properly computed preconditioners.
"""

import json
import time
from pathlib import Path
import sys

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from bit_qg.utils.graph_io import load_quantum_graph
from bit_qg.benchmarks.benchmarking import QuantumGraphBenchmark
from bit_qg.preconditioners.degree import DegreePreconditioner
from bit_qg.preconditioners.diagonal import DiagonalPreconditioner
from bit_qg.preconditioners.polynomial import PolynomialPreconditioner
from bit_qg.preconditioners.neumann_neumann import NeumannNeumannPreconditioner


def generate_reference_results():
    """Generate reference results for small test cases."""
    
    # Test parameters
    test_cases = [
        ("dorogovtsev_goltsev_mendes_1", 3),  # Small graph, logN=3 (N=9 points)
    ]
    
    results = {}
    
    for graph_name, logN in test_cases:
        print(f"Processing {graph_name}, logN={logN}")
        
        try:
            # Load graph
            edges, vertices = load_quantum_graph(graph_name)
            N = 2**logN - 1 + 2
            
            # Create benchmark
            benchmark = QuantumGraphBenchmark(edges, vertices)
            
            # Test all solver/preconditioner combinations
            solvers = ['cg', 'bicgstab']
            preconditioners = [
                ('identity', None),
                ('degree', DegreePreconditioner()),
                ('diagonal', DiagonalPreconditioner()),
                ('polynomial', PolynomialPreconditioner()),
                ('neumann_neumann', NeumannNeumannPreconditioner()),
            ]
            
            graph_key = f"{graph_name}_logN{logN}"
            results[graph_key] = {}
            
            for solver in solvers:
                results[graph_key][solver] = {}
                
                for prec_name, prec_obj in preconditioners:
                    print(f"  Testing {solver} + {prec_name}")
                    
                    # Run benchmark (this will now properly compute the preconditioner)
                    result = benchmark.benchmark_single_run(N, solver, prec_obj)
                    
                    results[graph_key][solver][prec_name] = {
                        'assembly_time': result.assembly_time,
                        'runtime': result.solve_time,
                        'iterations': result.iterations,
                        'error': result.residual_norm,
                        'success': result.success
                    }
                    
                    print(f"    Result: {result.iterations} iter, {result.residual_norm:.2e} residual")
        
        except Exception as e:
            print(f"Error processing {graph_name}: {e}")
            continue
    
    # Save results
    output_file = Path(__file__).parent / "tests" / "regression" / "reference_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Reference results saved to {output_file}")
    return results


if __name__ == "__main__":
    results = generate_reference_results()
    
    # Print summary
    print("\nSummary:")
    for graph_key, graph_data in results.items():
        print(f"{graph_key}:")
        for solver, solver_data in graph_data.items():
            print(f"  {solver}:")
            for prec_name, prec_data in solver_data.items():
                status = "✅" if prec_data['success'] else "❌"
                print(f"    {prec_name}: {prec_data['iterations']} iter, {prec_data['error']:.2e} error {status}")