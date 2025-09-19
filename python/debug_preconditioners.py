#!/usr/bin/env python3
"""
Debug script to investigate preconditioner failures.

This script focuses on understanding why custom preconditioners fail
on certain graphs while working on others.
"""

import sys
import traceback
from pathlib import Path

from bit_qg.benchmarks import QuantumGraphBenchmark
from bit_qg.preconditioners import (
    DegreePreconditioner,
    DiagonalPreconditioner,
    NeumannNeumannPreconditioner,
    PolynomialPreconditioner,
)
from bit_qg.utils import GraphLoader


def debug_preconditioner_failure():
    """Debug specific preconditioner failures on problematic graphs."""
    print("🔍 Debugging Preconditioner Failures")
    print("=" * 50)
    
    loader = GraphLoader()
    
    # Test the most problematic case: barabasi_albert_4 with degree preconditioner
    print("📊 Loading barabasi_albert_4 graph...")
    try:
        edges, vertices = loader.load_from_graph_file("barabasi_albert_4")
        print(f"   Vertices: {vertices}")
        print(f"   Edges: {len(edges)}")
        
        # Test with small problem size first
        N = 9  # logN = 3
        print(f"   Discretization: N = {N}")
        
        # Create benchmark
        benchmark = QuantumGraphBenchmark(edges, vertices)
        
        print("\n🧪 Testing each preconditioner individually...")
        
        preconditioners = {
            "degree": DegreePreconditioner,
            "diagonal": DiagonalPreconditioner, 
            "polynomial": PolynomialPreconditioner,
            "neumann_neumann": NeumannNeumannPreconditioner,
        }
        
        for prec_name, prec_class in preconditioners.items():
            print(f"\n--- Testing {prec_name.title()} Preconditioner ---")
            
            try:
                # Create preconditioner
                print("1. Creating preconditioner instance...")
                preconditioner = prec_class()
                print(f"   ✅ {prec_class.__name__} created successfully")
                
                # Create problem for preconditioner computation
                print("2. Creating problem and computing preconditioner...")
                from bit_qg.core.mf_quantum_graph import MFQuantumGraph
                problem = MFQuantumGraph(N, vertices, edges)
                preconditioner.compute(problem)
                print(f"   ✅ Problem created and preconditioner computed: shape {problem.shape}")
                
                # Test preconditioner solve method
                print("3. Testing preconditioner solve method...")
                test_vector = problem.bG.copy()
                print(f"   Test vector norm: {test_vector.dot(test_vector)**0.5:.2e}")
                
                result = preconditioner.solve(test_vector)
                print(f"   ✅ Solve successful: result norm {result.dot(result)**0.5:.2e}")
                
                # Test LinearOperator creation
                print("4. Testing LinearOperator interface...")
                lin_op = preconditioner.as_linear_operator()
                print(f"   ✅ LinearOperator created: shape {lin_op.shape}")
                
                # Test benchmark run
                print("5. Testing benchmark run...")
                bench_result = benchmark.benchmark_single_run(
                    n_points=N,
                    solver_name="cg",
                    preconditioner=preconditioner,
                    graph_name="barabasi_albert_4"
                )
                
                if bench_result.success:
                    print(f"   ✅ Benchmark successful:")
                    print(f"      Assembly time: {bench_result.assembly_time:.2e}s")
                    print(f"      Solve time: {bench_result.solve_time:.2e}s")
                    print(f"      Iterations: {bench_result.iterations}")
                    print(f"      Residual: {bench_result.residual_norm:.2e}")
                else:
                    print(f"   ❌ Benchmark failed:")
                    print(f"      Residual: {bench_result.residual_norm}")
                
            except Exception as e:
                print(f"   ❌ Error in {prec_name}: {e}")
                print(f"   Traceback: {traceback.format_exc()}")
                
                # Try to identify the specific failure point
                try:
                    print(f"   🔍 Attempting to isolate failure...")
                    preconditioner = prec_class(vertices)
                    print(f"      ✅ Preconditioner creation OK")
                    
                    # Test with minimal vector
                    import numpy as np
                    test_vec = np.ones(vertices) * 0.1
                    result = preconditioner.solve(test_vec)
                    print(f"      ✅ Simple solve OK")
                    
                except Exception as inner_e:
                    print(f"      ❌ Inner error: {inner_e}")
        
        print("\n📈 Testing on working graph for comparison...")
        print("Loading dorogovtsev_goltsev_mendes_1...")
        
        edges_working, vertices_working = loader.load_from_graph_file("dorogovtsev_goltsev_mendes_1")
        benchmark_working = QuantumGraphBenchmark(edges_working, vertices_working)
        
        print(f"   Vertices: {vertices_working}, Edges: {len(edges_working)}")
        
        # Test degree preconditioner on working graph
        try:
            from bit_qg.core.mf_quantum_graph import MFQuantumGraph
            temp_problem = MFQuantumGraph(N, vertices_working, edges_working)
            
            preconditioner = DegreePreconditioner()
            preconditioner.compute(temp_problem)
            result = benchmark_working.benchmark_single_run(
                n_points=N,
                solver_name="cg", 
                preconditioner=preconditioner,
                graph_name="dorogovtsev_goltsev_mendes_1"
            )
            print(f"   ✅ Degree preconditioner works on dgm_1: success={result.success}")
        except Exception as e:
            print(f"   ❌ Degree preconditioner also fails on dgm_1: {e}")
        
    except Exception as e:
        print(f"❌ Failed to load graph: {e}")
        return
    
    print(f"\n🎯 Analysis complete. Check output above for failure patterns.")


if __name__ == "__main__":
    debug_preconditioner_failure()