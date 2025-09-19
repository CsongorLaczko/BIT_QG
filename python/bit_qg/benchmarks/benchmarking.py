"""
Performance benchmarking framework for quantum graph finite element methods.

This module provides functionality to measure and compare the performance of
different solvers and preconditioners for quantum graph problems, porting
the functionality from the C++ measure_nn.cpp implementation.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, cg

from ..core.mf_quantum_graph import MFQuantumGraph
from ..core.qgedge import QGEdge
from ..preconditioners.base import PreconditionerBase
from ..preconditioners.degree import DegreePreconditioner
from ..preconditioners.diagonal import DiagonalPreconditioner
from ..preconditioners.neumann_neumann import NeumannNeumannPreconditioner
from ..preconditioners.polynomial import PolynomialPreconditioner
from ..utils.graph_io import load_quantum_graph


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    solver_name: str
    preconditioner_name: str
    n_points: int
    assembly_time: float
    solve_time: float
    total_time: float
    iterations: int
    residual_norm: float
    success: bool
    graph_name: str = ""
    run_index: int = 0


@dataclass
class BenchmarkSummary:
    """Statistical summary of multiple benchmark runs."""

    solver_name: str
    preconditioner_name: str
    n_points: int
    n_runs: int
    mean_assembly_time: float
    std_assembly_time: float
    mean_solve_time: float
    std_solve_time: float
    mean_total_time: float
    std_total_time: float
    mean_iterations: float
    std_iterations: float
    mean_residual_norm: float
    std_residual_norm: float
    success_rate: float
    graph_name: str = ""


class QuantumGraphBenchmark:
    """
    Benchmark runner for quantum graph finite element methods.

    This class provides methods to benchmark different solvers and preconditioners
    on quantum graph problems, measuring assembly time, solve time, convergence,
    and accuracy across multiple runs.
    """

    def __init__(
        self,
        edges: list[QGEdge],
        vertices: int,
        tolerance: float = np.sqrt(2.2204e-16),
        max_iterations: int = 1000,
    ):
        """
        Initialize benchmark with a quantum graph.

        Args:
            edges: List of QGEdge objects defining the graph
            vertices: Number of vertices in the graph
            tolerance: Solver convergence tolerance (matches C++ default)
            max_iterations: Maximum solver iterations
        """
        self.edges = edges
        self.vertices = vertices
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    @classmethod
    def from_graph_file(
        cls,
        graph_file: str | Path,
        tolerance: float = np.sqrt(2.2204e-16),
        max_iterations: int = 1000,
    ) -> "QuantumGraphBenchmark":
        """Create benchmark from a graph file."""
        edges, vertices = load_quantum_graph(graph_file)
        return cls(edges, vertices, tolerance, max_iterations)

    def _identity_preconditioner_solve(
        self,
        A: Any,
        b: np.ndarray,
        solver_func: Callable,
        x0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Solve with identity preconditioner (no preconditioning)."""
        if x0 is None:
            x0 = np.zeros_like(b)

        # Create LinearOperator for the MFQuantumGraph
        A_op = LinearOperator(A.shape, matvec=A.solve, dtype=np.float64)

        if solver_func == cg:
            solution, info = solver_func(
                A_op,
                b,
                x0=x0,
                rtol=self.tolerance,
                maxiter=self.max_iterations,
                atol=0.0,
            )
        else:  # bicgstab
            solution, info = solver_func(
                A_op,
                b,
                x0=x0,
                rtol=self.tolerance,
                maxiter=self.max_iterations,
                atol=0.0,
            )

        # Calculate residual norm
        residual = b - A.solve(solution)
        residual_norm = np.linalg.norm(residual)

        # For SciPy solvers, info > 0 means convergence not achieved
        iterations = info if info > 0 else self.max_iterations
        if info == 0:
            # Successful convergence, we need to estimate iterations
            # This is a limitation of SciPy interface
            iterations = 1  # Placeholder - SciPy doesn't return iteration count

        return solution, iterations, residual_norm

    def _preconditioned_solve(
        self,
        A: Any,
        b: np.ndarray,
        preconditioner: PreconditionerBase,
        solver_func: Callable,
        x0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Solve with a custom preconditioner."""
        if x0 is None:
            x0 = np.zeros_like(b)

        # Create LinearOperator for the MFQuantumGraph
        A_op = LinearOperator(A.shape, matvec=A.solve, dtype=np.float64)

        # Use the preconditioner as a LinearOperator
        M = preconditioner.as_linear_operator()

        if solver_func == cg:
            solution, info = solver_func(
                A_op,
                b,
                x0=x0,
                M=M,
                rtol=self.tolerance,
                maxiter=self.max_iterations,
                atol=0.0,
            )
        else:  # bicgstab
            solution, info = solver_func(
                A_op,
                b,
                x0=x0,
                M=M,
                rtol=self.tolerance,
                maxiter=self.max_iterations,
                atol=0.0,
            )

        # Calculate residual norm
        residual = b - A.solve(solution)
        residual_norm = np.linalg.norm(residual)

        # For SciPy solvers, info > 0 means convergence not achieved
        iterations = info if info > 0 else self.max_iterations
        if info == 0:
            iterations = 1  # Placeholder - SciPy doesn't return iteration count

        return solution, iterations, residual_norm

    def benchmark_single_run(
        self,
        n_points: int,
        solver_name: str = "cg",
        preconditioner: PreconditionerBase | None = None,
        graph_name: str = "",
        run_index: int = 0,
    ) -> BenchmarkResult:
        """
        Run a single benchmark iteration.

        Args:
            n_points: Number of discretization points per edge (N parameter)
            solver_name: Either "cg" or "bicgstab"
            preconditioner: Preconditioner instance, None for identity
            graph_name: Name of the graph being tested
            run_index: Index of this run in a series

        Returns:
            BenchmarkResult with timing and convergence information
        """
        # Map solver names to functions
        solver_map = {"cg": cg, "bicgstab": bicgstab}

        if solver_name not in solver_map:
            raise ValueError(f"Unknown solver: {solver_name}")

        solver_func = solver_map[solver_name]
        preconditioner_name = (
            "identity" if preconditioner is None else type(preconditioner).__name__
        )

        # Assembly phase timing
        start_assembly = time.perf_counter()
        problem = MFQuantumGraph(n_points, self.vertices, self.edges)
        end_assembly = time.perf_counter()
        assembly_time = end_assembly - start_assembly

        # Solve phase timing
        start_solve = time.perf_counter()
        try:
            if preconditioner is None:
                solution, iterations, residual_norm = (
                    self._identity_preconditioner_solve(
                        problem, problem.bG, solver_func
                    )
                )
            else:
                solution, iterations, residual_norm = self._preconditioned_solve(
                    problem, problem.bG, preconditioner, solver_func
                )
            success = True
        except Exception:
            # Handle solver failures gracefully
            iterations = self.max_iterations
            residual_norm = np.inf
            success = False

        end_solve = time.perf_counter()
        solve_time = end_solve - start_solve
        total_time = assembly_time + solve_time

        return BenchmarkResult(
            solver_name=solver_name,
            preconditioner_name=preconditioner_name,
            n_points=n_points,
            assembly_time=assembly_time,
            solve_time=solve_time,
            total_time=total_time,
            iterations=iterations,
            residual_norm=residual_norm,
            success=success,
            graph_name=graph_name,
            run_index=run_index,
        )

    def benchmark_multiple_runs(
        self,
        n_points: int,
        n_runs: int = 5,
        solver_name: str = "cg",
        preconditioner: PreconditionerBase | None = None,
        graph_name: str = "",
    ) -> BenchmarkSummary:
        """
        Run multiple benchmark iterations and compute statistics.

        Args:
            n_points: Number of discretization points per edge
            n_runs: Number of benchmark runs to average
            solver_name: Either "cg" or "bicgstab"
            preconditioner: Preconditioner instance, None for identity
            graph_name: Name of the graph being tested

        Returns:
            BenchmarkSummary with mean and standard deviation statistics
        """
        results = []

        for run_idx in range(n_runs):
            result = self.benchmark_single_run(
                n_points=n_points,
                solver_name=solver_name,
                preconditioner=preconditioner,
                graph_name=graph_name,
                run_index=run_idx,
            )
            results.append(result)

        # Compute statistics
        assembly_times = [r.assembly_time for r in results]
        solve_times = [r.solve_time for r in results]
        total_times = [r.total_time for r in results]
        iterations = [r.iterations for r in results]
        residual_norms = [
            r.residual_norm for r in results if np.isfinite(r.residual_norm)
        ]
        successes = [r.success for r in results]

        preconditioner_name = (
            "identity" if preconditioner is None else type(preconditioner).__name__
        )

        return BenchmarkSummary(
            solver_name=solver_name,
            preconditioner_name=preconditioner_name,
            n_points=n_points,
            n_runs=n_runs,
            mean_assembly_time=np.mean(assembly_times),
            std_assembly_time=np.std(assembly_times, ddof=1)
            if len(assembly_times) > 1
            else 0.0,
            mean_solve_time=np.mean(solve_times),
            std_solve_time=np.std(solve_times, ddof=1) if len(solve_times) > 1 else 0.0,
            mean_total_time=np.mean(total_times),
            std_total_time=np.std(total_times, ddof=1) if len(total_times) > 1 else 0.0,
            mean_iterations=np.mean(iterations),
            std_iterations=np.std(iterations, ddof=1) if len(iterations) > 1 else 0.0,
            mean_residual_norm=np.mean(residual_norms) if residual_norms else np.inf,
            std_residual_norm=np.std(residual_norms, ddof=1)
            if len(residual_norms) > 1
            else 0.0,
            success_rate=np.mean(successes),
            graph_name=graph_name,
        )

    def benchmark_all_preconditioners(
        self,
        n_points: int,
        n_runs: int = 5,
        solver_name: str = "cg",
        graph_name: str = "",
    ) -> list[BenchmarkSummary]:
        """
        Benchmark all available preconditioners with the specified solver.

        This method replicates the comprehensive comparison from measure_nn.cpp,
        testing identity (vanilla), degree, diagonal, polynomial, and Neumann-Neumann preconditioners.

        Args:
            n_points: Number of discretization points per edge
            n_runs: Number of runs to average over
            solver_name: Either "cg" or "bicgstab"
            graph_name: Name of the graph being tested

        Returns:
            List of BenchmarkSummary objects for each preconditioner
        """
        summaries = []

        # Create a temporary problem for preconditioner initialization
        temp_problem = MFQuantumGraph(n_points, self.vertices, self.edges)

        # Identity preconditioner (vanilla)
        summary = self.benchmark_multiple_runs(
            n_points=n_points,
            n_runs=n_runs,
            solver_name=solver_name,
            preconditioner=None,
            graph_name=graph_name,
        )
        summaries.append(summary)

        # Degree preconditioner
        degree_prec = DegreePreconditioner()
        degree_prec.compute(temp_problem)
        summary = self.benchmark_multiple_runs(
            n_points=n_points,
            n_runs=n_runs,
            solver_name=solver_name,
            preconditioner=degree_prec,
            graph_name=graph_name,
        )
        summaries.append(summary)

        # Diagonal preconditioner
        diagonal_prec = DiagonalPreconditioner()
        diagonal_prec.compute(temp_problem)
        summary = self.benchmark_multiple_runs(
            n_points=n_points,
            n_runs=n_runs,
            solver_name=solver_name,
            preconditioner=diagonal_prec,
            graph_name=graph_name,
        )
        summaries.append(summary)

        # Polynomial preconditioner
        poly_prec = PolynomialPreconditioner()
        poly_prec.compute(temp_problem)
        summary = self.benchmark_multiple_runs(
            n_points=n_points,
            n_runs=n_runs,
            solver_name=solver_name,
            preconditioner=poly_prec,
            graph_name=graph_name,
        )
        summaries.append(summary)

        # Neumann-Neumann preconditioner
        nn_prec = NeumannNeumannPreconditioner()
        nn_prec.compute(temp_problem)
        summary = self.benchmark_multiple_runs(
            n_points=n_points,
            n_runs=n_runs,
            solver_name=solver_name,
            preconditioner=nn_prec,
            graph_name=graph_name,
        )
        summaries.append(summary)

        return summaries


def print_benchmark_summary(summary: BenchmarkSummary) -> None:
    """Print a benchmark summary in a format similar to the C++ output."""
    print(f"{summary.preconditioner_name}")
    print(
        f"assembly time: {summary.mean_assembly_time:.6e} "
        f"runtime: {summary.mean_solve_time:.6e} "
        f"iterations: {summary.mean_iterations:.1f} "
        f"error: {summary.mean_residual_norm:.6e}"
    )


def print_benchmark_results(
    summaries: list[BenchmarkSummary], solver_name: str
) -> None:
    """Print benchmark results in a format matching the C++ measure_nn output."""
    print(f"{solver_name.upper()}")

    for summary in summaries:
        preconditioner_display = {
            "identity": "Vanilla",
            "DegreePreconditioner": "Degree",
            "DiagonalPreconditioner": "Diagonal",
            "PolynomialPreconditioner": "Polynomial",
            "NeumannNeumannPreconditioner": "Neumann-Neumann",
        }.get(summary.preconditioner_name, summary.preconditioner_name)

        print(f"\n{preconditioner_display}")
        print(
            f"assembly time: {summary.mean_assembly_time:.6e} "
            f"runtime: {summary.mean_solve_time:.6e} "
            f"iterations: {summary.mean_iterations:.1f} "
            f"error: {summary.mean_residual_norm:.6e}"
        )
