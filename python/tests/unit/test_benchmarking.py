"""
Tests for the benchmarking module.
"""

import numpy as np
import pytest

from bit_qg.benchmarks import BenchmarkResult, BenchmarkSummary, QuantumGraphBenchmark
from bit_qg.utils.graph_io import GraphLoader


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating a BenchmarkResult instance."""
        result = BenchmarkResult(
            solver_name="cg",
            preconditioner_name="identity",
            n_points=10,
            assembly_time=0.001,
            solve_time=0.002,
            total_time=0.003,
            iterations=5,
            residual_norm=1e-10,
            success=True,
        )

        assert result.solver_name == "cg"
        assert result.preconditioner_name == "identity"
        assert result.n_points == 10
        assert result.assembly_time == 0.001
        assert result.solve_time == 0.002
        assert result.total_time == 0.003
        assert result.iterations == 5
        assert result.residual_norm == 1e-10
        assert result.success is True


class TestBenchmarkSummary:
    """Test BenchmarkSummary dataclass."""

    def test_benchmark_summary_creation(self):
        """Test creating a BenchmarkSummary instance."""
        summary = BenchmarkSummary(
            solver_name="bicgstab",
            preconditioner_name="DegreePreconditioner",
            n_points=20,
            n_runs=3,
            mean_assembly_time=0.001,
            std_assembly_time=0.0001,
            mean_solve_time=0.005,
            std_solve_time=0.0005,
            mean_total_time=0.006,
            std_total_time=0.0006,
            mean_iterations=8.5,
            std_iterations=1.2,
            mean_residual_norm=1e-9,
            std_residual_norm=1e-10,
            success_rate=1.0,
        )

        assert summary.solver_name == "bicgstab"
        assert summary.preconditioner_name == "DegreePreconditioner"
        assert summary.n_points == 20
        assert summary.n_runs == 3
        assert summary.mean_assembly_time == 0.001
        assert summary.std_assembly_time == 0.0001
        assert summary.mean_solve_time == 0.005
        assert summary.std_solve_time == 0.0005
        assert summary.mean_total_time == 0.006
        assert summary.std_total_time == 0.0006
        assert summary.mean_iterations == 8.5
        assert summary.std_iterations == 1.2
        assert summary.mean_residual_norm == 1e-9
        assert summary.std_residual_norm == 1e-10
        assert summary.success_rate == 1.0


class TestQuantumGraphBenchmark:
    """Test QuantumGraphBenchmark class."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple 2-edge graph for testing."""
        loader = GraphLoader()
        return loader.create_simple_path(3)  # 3 vertices, 2 edges

    def test_benchmark_creation(self, simple_graph):
        """Test creating a QuantumGraphBenchmark instance."""
        edges, vertices = simple_graph
        benchmark = QuantumGraphBenchmark(edges, vertices)

        assert benchmark.edges == edges
        assert benchmark.vertices == vertices
        assert benchmark.tolerance == np.sqrt(2.2204e-16)
        assert benchmark.max_iterations == 1000

    def test_benchmark_creation_with_custom_params(self, simple_graph):
        """Test creating a benchmark with custom parameters."""
        edges, vertices = simple_graph
        benchmark = QuantumGraphBenchmark(
            edges, vertices, tolerance=1e-8, max_iterations=500
        )

        assert benchmark.tolerance == 1e-8
        assert benchmark.max_iterations == 500

    def test_benchmark_single_run_identity(self, simple_graph):
        """Test running a single benchmark with identity preconditioner."""
        edges, vertices = simple_graph
        benchmark = QuantumGraphBenchmark(edges, vertices, tolerance=1e-6)

        result = benchmark.benchmark_single_run(
            n_points=5, solver_name="cg", preconditioner=None
        )

        assert isinstance(result, BenchmarkResult)
        assert result.solver_name == "cg"
        assert result.preconditioner_name == "identity"
        assert result.n_points == 5
        assert result.assembly_time > 0
        assert result.solve_time > 0
        assert result.total_time > 0
        assert result.iterations > 0
        assert result.success is True

    def test_benchmark_single_run_bicgstab(self, simple_graph):
        """Test running a single benchmark with BiCGSTAB solver."""
        edges, vertices = simple_graph
        benchmark = QuantumGraphBenchmark(edges, vertices, tolerance=1e-6)

        result = benchmark.benchmark_single_run(
            n_points=5, solver_name="bicgstab", preconditioner=None
        )

        assert result.solver_name == "bicgstab"
        assert result.success is True

    def test_benchmark_invalid_solver(self, simple_graph):
        """Test that invalid solver names raise an error."""
        edges, vertices = simple_graph
        benchmark = QuantumGraphBenchmark(edges, vertices)

        with pytest.raises(ValueError, match="Unknown solver"):
            benchmark.benchmark_single_run(n_points=5, solver_name="invalid")

    def test_benchmark_multiple_runs(self, simple_graph):
        """Test running multiple benchmarks and getting statistics."""
        edges, vertices = simple_graph
        benchmark = QuantumGraphBenchmark(edges, vertices, tolerance=1e-6)

        summary = benchmark.benchmark_multiple_runs(
            n_points=5, n_runs=3, solver_name="cg", preconditioner=None
        )

        assert isinstance(summary, BenchmarkSummary)
        assert summary.solver_name == "cg"
        assert summary.preconditioner_name == "identity"
        assert summary.n_points == 5
        assert summary.n_runs == 3
        assert summary.mean_assembly_time > 0
        assert summary.mean_solve_time > 0
        assert summary.mean_total_time > 0
        assert summary.mean_iterations > 0
        assert summary.success_rate == 1.0

    def test_benchmark_all_preconditioners(self, simple_graph):
        """Test benchmarking all preconditioners."""
        edges, vertices = simple_graph
        benchmark = QuantumGraphBenchmark(edges, vertices, tolerance=1e-4)

        summaries = benchmark.benchmark_all_preconditioners(
            n_points=5, n_runs=2, solver_name="cg"
        )

        assert (
            len(summaries) == 5
        )  # identity, degree, diagonal, polynomial, neumann-neumann
        assert all(isinstance(s, BenchmarkSummary) for s in summaries)
        assert summaries[0].preconditioner_name == "identity"
        assert summaries[1].preconditioner_name == "DegreePreconditioner"
        assert summaries[2].preconditioner_name == "DiagonalPreconditioner"
        assert summaries[3].preconditioner_name == "PolynomialPreconditioner"
        assert summaries[4].preconditioner_name == "NeumannNeumannPreconditioner"

        # All should be successful on this simple problem
        assert all(s.success_rate > 0 for s in summaries)
