#!/usr/bin/env python3
"""
Reference data generator for regression testing.

This module generates baseline data with properly computed preconditioners
for use in regression testing validation.
"""

import json
import time
from pathlib import Path

from bit_qg.utils.graph_io import load_quantum_graph
from bit_qg.benchmarks.benchmarking import QuantumGraphBenchmark
from bit_qg.preconditioners.degree import DegreePreconditioner
from bit_qg.preconditioners.diagonal import DiagonalPreconditioner
from bit_qg.preconditioners.polynomial import PolynomialPreconditioner
from bit_qg.preconditioners.neumann_neumann import NeumannNeumannPreconditioner


def generate_reference_results():
    """
    Generate reference results for comprehensive validation.
    
    This function generates Python baseline results for comparison with C++ implementation.
    
    C++ Reference Results (for validation):
    - barabasi_albert_4 (logN=6): CG 3 iter, BiCGSTAB 4 iter (small graph)
    - barabasi_albert_100 (logN=6): CG 39/25/25/13/13 iter, BiCGSTAB 28/17/16/9/9 iter (medium)
    - barabasi_albert_500 (logN=6): CG 63/28/28/15/15 iter, BiCGSTAB 42/18/17/11/10 iter (large)
    
    Python typically shows ±1-2 iteration difference due to algorithmic conventions,
    which is within expected tolerance for iterative methods.
    """
    
    # Test parameters - expanded to include C++ validation cases
    test_cases = [
        ("dorogovtsev_goltsev_mendes_1", 3),  # Small graph, logN=3 (N=9 points)
        ("barabasi_albert_4", 6),             # Small Barabási-Albert, logN=6 (N=65 points)  
        ("barabasi_albert_100", 6),           # Medium Barabási-Albert, logN=6 (N=65 points)
        ("barabasi_albert_500", 6),           # Large Barabási-Albert, logN=6 (N=65 points)
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
    output_file = Path(__file__).parent.parent.parent / "tests" / "regression" / "reference_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Reference results saved to {output_file}")
    return results