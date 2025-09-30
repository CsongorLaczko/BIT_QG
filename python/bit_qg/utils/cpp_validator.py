#!/usr/bin/env python3
"""
C++ Validation Interface for measure_nn.cpp

This module provides functionality to replicate the exact interface and behavior 
of the C++ measure_nn executable, allowing for direct comparison of results 
between C++ and Python implementations.
"""

import sys
from pathlib import Path

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
            "NeumannNeumannPreconditioner": "Neumann-Neumann",
        }
        print(cpp_names.get(preconditioner_name, preconditioner_name))

    print(
        f"assembly time: {summary.mean_assembly_time:.6e} "
        f"runtime: {summary.mean_solve_time:.6e} "
        f"iterations: {summary.mean_iterations:.0f} "
        f"error: {summary.mean_residual_norm:.6e}"
    )


def run_cpp_validation(graph: str, size: int, log_n: int, runs: int = 1) -> bool:
    """
    Run C++ validation benchmark.
    
    Args:
        graph: Graph type name
        size: Graph size parameter
        log_n: Discretization parameter
        runs: Number of benchmark runs
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Calculate N using C++ formula: N = pow(2, logN) - 1 + 2
        N = 2**log_n - 1 + 2

        # Construct filename: {graph}_{size} (without .txt extension)
        graph_name = f"{graph}_{size}"

        # Load graph using Python GraphLoader
        loader = GraphLoader()
        edges, vertices = loader.load_from_graph_file(graph_name)

        # Create benchmark instance
        benchmark = QuantumGraphBenchmark(edges, vertices)

        print("CG")  # Match C++ solver header

        # Run all preconditioners with exact C++ ordering and output format
        # Note: C++ uses tolerance = sqrt(2.2204e-16), Python uses default solver tolerance
        summaries = benchmark.benchmark_all_preconditioners(
            n_points=N, n_runs=runs, solver_name="cg"
        )

        # Output results in exact C++ format
        for i, summary in enumerate(summaries):
            if i > 0:  # Add blank line between preconditioners (matches C++ output)
                print()

            cpp_format_output("cg", summary.preconditioner_name, summary)

        print("\nBiCGSTAB")  # Match C++ solver header

        # Run BiCGSTAB solver as well (matching full C++ behavior)
        summaries_bicgstab = benchmark.benchmark_all_preconditioners(
            n_points=N, n_runs=runs, solver_name="bicgstab"
        )

        # Output results in exact C++ format
        for i, summary in enumerate(summaries_bicgstab):
            if i > 0:  # Add blank line between preconditioners (matches C++ output)
                print()

            cpp_format_output("bicgstab", summary.preconditioner_name, summary)
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Make sure the graph file '../graphs/{graph}_{size}.txt' exists")
        return False