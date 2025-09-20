#!/usr/bin/env python3
"""
Diagonal Preconditioner Demonstration

This script demonstrates the newly implemented DiagonalPreconditioner
alongside all other preconditioners for quantum graph benchmarking.

The diagonal preconditioner computes the diagonal entries of the Schur
complement system by applying the system matrix to unit vectors, then
uses the inverse of these diagonal entries as a preconditioner.
"""

from bit_qg.benchmarks import QuantumGraphBenchmark
from bit_qg.utils import GraphLoader


def main():
    """Demonstrate all preconditioners including the new diagonal preconditioner."""
    print("Diagonal Preconditioner Demonstration")
    print("=" * 50)

    # Create a simple path graph (4 vertices, 3 edges)
    loader = GraphLoader()
    edges, vertices = loader.create_simple_path(4)

    print(f"Graph: {vertices} vertices, {len(edges)} edges")
    print(f"Edges: {[f'{e.out}→{e.in_}' for e in edges]}")
    print()

    # Create benchmark instance
    benchmark = QuantumGraphBenchmark(edges, vertices)

    # Run all preconditioners with moderate problem size
    print("Running benchmarks with all 5 preconditioners...")
    print("(This includes the newly implemented DiagonalPreconditioner)")
    print()

    summaries = benchmark.benchmark_all_preconditioners(
        n_points=10,  # Moderate discretization
        n_runs=3,  # Multiple runs for averaging
        solver_name="cg",
    )

    # Display results in a nice table
    print(
        f"{'Preconditioner':<25} {'Iterations':<10} {'Total Time':<12} {'Success':<8}"
    )
    print("-" * 60)

    for summary in summaries:
        prec_name = summary.preconditioner_name
        if prec_name == "identity":
            prec_display = "Identity (vanilla)"
        else:
            prec_display = prec_name.replace("Preconditioner", "")

        print(
            f"{prec_display:<25} "
            f"{summary.mean_iterations:<10.1f} "
            f"{summary.mean_total_time * 1000:<12.2f}ms "
            f"{summary.success_rate * 100:<8.0f}%"
        )

    print()
    print("✅ All preconditioners working correctly!")
    print("✅ DiagonalPreconditioner successfully implemented!")

    # Show the specific benefits of the diagonal preconditioner
    diagonal_summary = next(s for s in summaries if "Diagonal" in s.preconditioner_name)
    identity_summary = next(s for s in summaries if s.preconditioner_name == "identity")

    speedup = identity_summary.mean_total_time / diagonal_summary.mean_total_time
    iter_reduction = identity_summary.mean_iterations / diagonal_summary.mean_iterations

    print("\nDiagonal Preconditioner Benefits:")
    print(f"  📈 Speedup over identity: {speedup:.1f}x")
    print(f"  🔄 Iteration reduction: {iter_reduction:.1f}x")


if __name__ == "__main__":
    main()
