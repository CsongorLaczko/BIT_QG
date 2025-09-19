#!/usr/bin/env python3
"""
C++ Validation Interface for measure_nn.cpp

This script replicates the exact interface and behavior of the C++ measure_nn
executable, allowing for direct comparison of results between C++ and Python
implementations.

Usage: python cpp_validation.py <graph> <size> <logN> [runs]

This matches the C++ interface:
  ./build/measure_nn <graph> <size> <logN> [runs]
"""

import sys

# Add the bit_qg package to path
sys.path.append('python')

from bit_qg.benchmarks import QuantumGraphBenchmark
from bit_qg.utils import GraphLoader


def cpp_format_output(solver_name: str, preconditioner_name: str, summary):
    """
    Format output to match C++ measure_nn.cpp exact format.
    
    C++ format: "assembly time: <time> runtime: <time> iterations: <int> error: <float>"
    """
    if preconditioner_name == "identity":
        print("Vanilla")
    else:
        # Map Python class names to C++ output names
        cpp_names = {
            "DegreePreconditioner": "Degree",
            "DiagonalPreconditioner": "Diagonal", 
            "PolynomialPreconditioner": "Polynomial",
            "NeumannNeumannPreconditioner": "Neumann-Neumann"
        }
        print(cpp_names.get(preconditioner_name, preconditioner_name))
    
    print(f"assembly time: {summary.mean_assembly_time:.6e} "
          f"runtime: {summary.mean_solve_time:.6e} "
          f"iterations: {summary.mean_iterations:.0f} "
          f"error: {summary.mean_residual_norm:.6e}")


def main():
    """Main function matching C++ measure_nn.cpp interface."""
    if len(sys.argv) < 4:
        print("Usage: python cpp_validation.py <graph> <size> <logN> [runs]")
        print("Example: python cpp_validation.py dorogovtsev_goltsev_mendes 1 3 1")
        sys.exit(1)
    
    # Parse command line arguments (matching C++ exactly)
    graph = sys.argv[1]
    size = int(sys.argv[2])
    log_n = int(sys.argv[3])
    runs = 1
    if len(sys.argv) == 5:
        runs = int(sys.argv[4])
    
    # Calculate N using C++ formula: N = pow(2, logN) - 1 + 2
    N = 2**log_n - 1 + 2
    
    # Construct filename: {graph}_{size} (without .txt extension)
    graph_name = f"{graph}_{size}"
    
    # Load graph using Python GraphLoader
    loader = GraphLoader()
    try:
        edges, vertices = loader.load_from_graph_file(graph_name)
    except Exception as e:
        print(f"Error loading graph: {e}")
        sys.exit(1)
    
    # Create benchmark instance
    benchmark = QuantumGraphBenchmark(edges, vertices)
    
    print("CG")  # Match C++ solver header
    
    # Run all preconditioners with exact C++ ordering and output format  
    # Note: C++ uses tolerance = sqrt(2.2204e-16), Python uses default solver tolerance
    summaries = benchmark.benchmark_all_preconditioners(
        n_points=N,
        n_runs=runs,
        solver_name="cg"
    )
    
    # Output results in exact C++ format
    for i, summary in enumerate(summaries):
        if i > 0:  # Add blank line between preconditioners (matches C++ output)
            print()
        
        cpp_format_output("cg", summary.preconditioner_name, summary)


if __name__ == "__main__":
    main()