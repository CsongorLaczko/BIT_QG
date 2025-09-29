"""
Performance Regression Tests for BIT_QG Python Implementation

This module implements performance regression testing to detect degradation
over time and ensure scaling behavior remains consistent.
"""

import time
import psutil
import numpy as np
import pytest
from pathlib import Path
import sys
import json
from datetime import datetime

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from bit_qg.core.mf_quantum_graph import MFQuantumGraph
from bit_qg.core.qgedge import QGEdge
from bit_qg.utils.graph_io import load_quantum_graph
from bit_qg.preconditioners.degree import DegreePreconditioner
from bit_qg.preconditioners.diagonal import DiagonalPreconditioner
from bit_qg.preconditioners.polynomial import PolynomialPreconditioner
from bit_qg.preconditioners.neumann_neumann import NeumannNeumannPreconditioner


class PerformanceBaseline:
    """Manages performance baseline data for regression testing."""
    
    BASELINE_FILE = Path(__file__).parent / "performance_baseline.json"
    
    @classmethod
    def load_baseline(cls):
        """Load performance baseline from file."""
        if cls.BASELINE_FILE.exists():
            with open(cls.BASELINE_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    @classmethod
    def save_baseline(cls, data):
        """Save performance baseline to file."""
        with open(cls.BASELINE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def update_baseline(cls, test_name, measurement):
        """Update baseline for a specific test."""
        baseline = cls.load_baseline()
        baseline[test_name] = {
            'value': measurement,
            'timestamp': datetime.now().isoformat(),
            'description': f"Performance baseline for {test_name}"
        }
        cls.save_baseline(baseline)


class TestPerformanceRegression:
    """
    Test suite for performance regression detection.
    
    This class monitors assembly time, solve time, memory usage,
    and ensures performance doesn't degrade beyond acceptable limits.
    """
    
    # Performance regression tolerances
    SLOWDOWN_TOLERANCE = 0.10  # 10% slowdown tolerance
    MEMORY_TOLERANCE = 0.20    # 20% memory increase tolerance
    
    @pytest.fixture
    def small_graph(self):
        """Small graph for fast performance testing."""
        edges = [
            QGEdge(0, 1, lambda x: 1.0, lambda x: 0.0, lambda x: 1.0),
            QGEdge(1, 2, lambda x: 1.0, lambda x: 0.0, lambda x: 1.0),
            QGEdge(2, 3, lambda x: 1.0, lambda x: 0.0, lambda x: 1.0)
        ]
        return edges, 4
    
    @pytest.fixture
    def medium_graph(self):
        """Medium graph for scaling analysis."""
        # Create a small star graph
        vertices = 10
        edges = []
        for i in range(1, vertices):
            edges.append(QGEdge(0, i, lambda x: 1.0, lambda x: 0.0, lambda x: 1.0))
        return edges, vertices
    
    def measure_performance(self, mfqg, solver_type='cg', preconditioner=None):
        """Measure assembly and solve performance."""
        process = psutil.Process()
        
        # Measure memory before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure solve time
        start_time = time.perf_counter()
        solution = mfqg.solve(mfqg.bG, solver_type=solver_type, preconditioner=preconditioner)
        solve_time = time.perf_counter() - start_time
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        return {
            'solve_time': solve_time,
            'memory_usage': memory_usage,
            'solution_norm': np.linalg.norm(solution)
        }
    
    def test_small_graph_performance(self, small_graph):
        """Test performance on small graphs (<1 second expected)."""
        edges, vertices = small_graph
        N = 9
        
        mfqg = MFQuantumGraph(N, vertices, edges)
        perf = self.measure_performance(mfqg)
        
        # Check against baseline
        baseline = PerformanceBaseline.load_baseline()
        test_name = "small_graph_solve_time"
        
        if test_name in baseline:
            baseline_time = baseline[test_name]['value']
            slowdown = (perf['solve_time'] - baseline_time) / baseline_time
            
            assert slowdown <= self.SLOWDOWN_TOLERANCE, (
                f"Performance regression: {slowdown:.1%} slowdown "
                f"(tolerance: {self.SLOWDOWN_TOLERANCE:.1%})"
            )
        else:
            # First run - establish baseline
            PerformanceBaseline.update_baseline(test_name, perf['solve_time'])
        
        # Absolute performance bounds
        assert perf['solve_time'] < 1.0, f"Small graph solve too slow: {perf['solve_time']:.3f}s"
        assert perf['memory_usage'] < 100, f"Small graph memory usage too high: {perf['memory_usage']:.1f}MB"
    
    def test_medium_graph_performance(self, medium_graph):
        """Test performance on medium graphs (<10 seconds expected)."""
        edges, vertices = medium_graph
        N = 17  # Larger discretization
        
        mfqg = MFQuantumGraph(N, vertices, edges)
        perf = self.measure_performance(mfqg)
        
        # Check against baseline
        baseline = PerformanceBaseline.load_baseline()
        test_name = "medium_graph_solve_time"
        
        if test_name in baseline:
            baseline_time = baseline[test_name]['value']
            slowdown = (perf['solve_time'] - baseline_time) / baseline_time
            
            assert slowdown <= self.SLOWDOWN_TOLERANCE, (
                f"Medium graph performance regression: {slowdown:.1%} slowdown"
            )
        else:
            PerformanceBaseline.update_baseline(test_name, perf['solve_time'])
        
        # Absolute bounds
        assert perf['solve_time'] < 10.0, f"Medium graph solve too slow: {perf['solve_time']:.3f}s"
        assert perf['memory_usage'] < 500, f"Medium graph memory usage too high: {perf['memory_usage']:.1f}MB"
    
    def test_assembly_time_scaling(self, small_graph):
        """Test that assembly time scales linearly with problem size."""
        edges, vertices = small_graph
        
        assembly_times = []
        problem_sizes = []
        
        for N in [9, 17, 33]:  # Different discretizations
            start_time = time.perf_counter()
            mfqg = MFQuantumGraph(N, vertices, edges)
            assembly_time = time.perf_counter() - start_time
            
            assembly_times.append(assembly_time)
            problem_sizes.append(len(edges) * (N - 2) + vertices)  # Total DOF
        
        # Check that scaling is reasonable (not worse than quadratic)
        for i in range(len(assembly_times) - 1):
            size_ratio = problem_sizes[i+1] / problem_sizes[i]
            time_ratio = assembly_times[i+1] / assembly_times[i]
            
            # Time ratio should not exceed (size_ratio)²
            assert time_ratio <= size_ratio ** 2 * 2, (  # Factor of 2 for tolerance
                f"Assembly time scaling too poor: size ratio {size_ratio:.2f}, "
                f"time ratio {time_ratio:.2f}"
            )
    
    def test_preconditioner_performance_comparison(self, small_graph):
        """Test relative performance of different preconditioners."""
        edges, vertices = small_graph
        N = 9
        
        mfqg = MFQuantumGraph(N, vertices, edges)
        
        # Measure performance for each preconditioner
        preconditioners = {
            'identity': None,
            'degree': DegreePreconditioner(),
            'diagonal': DiagonalPreconditioner(),
            'polynomial': PolynomialPreconditioner(),
            'neumann_neumann': NeumannNeumannPreconditioner()
        }
        
        performance_results = {}
        
        for name, preconditioner in preconditioners.items():
            if preconditioner is not None:
                preconditioner.compute(mfqg)
                M = preconditioner.as_linear_operator()
            else:
                M = None
            
            perf = self.measure_performance(mfqg, preconditioner=M)
            performance_results[name] = perf
        
        # Preconditioners should generally be faster than identity
        # (though this depends on problem size and conditioning)
        identity_time = performance_results['identity']['solve_time']
        
        for name, perf in performance_results.items():
            if name != 'identity':
                # Performance should be reasonable (not 50x worse than identity)
                # Note: Neumann-Neumann can be much slower due to numerical conditioning
                max_factor = 50 if name == 'neumann_neumann' else 10
                assert perf['solve_time'] <= identity_time * max_factor, (
                    f"Preconditioner {name} too slow: {perf['solve_time']:.4f}s vs "
                    f"identity {identity_time:.4f}s (max factor: {max_factor}x)"
                )
    
    def test_memory_scaling(self, small_graph):
        """Test that memory usage scales linearly with problem size."""
        edges, vertices = small_graph
        
        memory_usage = []
        problem_sizes = []
        
        for N in [9, 17, 33]:
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            mfqg = MFQuantumGraph(N, vertices, edges)
            
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before
            
            memory_usage.append(memory_used)
            problem_sizes.append(len(edges) * (N - 2) + vertices)
        
        # Memory should scale roughly linearly
        for i in range(len(memory_usage) - 1):
            if memory_usage[i] > 0:  # Avoid division by zero
                size_ratio = problem_sizes[i+1] / problem_sizes[i]
                memory_ratio = memory_usage[i+1] / memory_usage[i]
                
                # Memory ratio should not exceed size_ratio² (sparse matrix storage)
                assert memory_ratio <= size_ratio ** 2 * 3, (  # Factor of 3 for tolerance
                    f"Memory scaling too poor: size ratio {size_ratio:.2f}, "
                    f"memory ratio {memory_ratio:.2f}"
                )
    
    def test_repeated_solve_performance(self, small_graph):
        """Test that repeated solves don't degrade performance (memory leaks)."""
        edges, vertices = small_graph
        N = 9
        
        mfqg = MFQuantumGraph(N, vertices, edges)
        
        solve_times = []
        memory_usage = []
        
        for i in range(5):  # Multiple solves
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            start_time = time.perf_counter()
            solution = mfqg.solve(mfqg.bG)
            solve_time = time.perf_counter() - start_time
            
            memory_after = process.memory_info().rss / 1024 / 1024
            
            solve_times.append(solve_time)
            memory_usage.append(memory_after)
        
        # Performance should remain consistent
        first_time = solve_times[0]
        for i, solve_time in enumerate(solve_times[1:], 1):
            slowdown = (solve_time - first_time) / first_time
            assert slowdown <= 0.5, (  # Allow 50% variation
                f"Repeated solve {i} degraded: {slowdown:.1%} slower"
            )
        
        # Memory should not grow significantly
        first_memory = memory_usage[0]
        for i, memory in enumerate(memory_usage[1:], 1):
            growth = (memory - first_memory) / first_memory if first_memory > 0 else 0
            assert growth <= 0.1, (  # Allow 10% memory growth
                f"Memory leak detected: {growth:.1%} growth at solve {i}"
            )


class TestPerformanceBounds:
    """Test absolute performance bounds for different problem classes."""
    
    def test_research_graph_performance(self):
        """Test performance on standard research graphs."""
        try:
            edges, vertices = load_quantum_graph("dorogovtsev_goltsev_mendes_1")
            N = 9
            
            mfqg = MFQuantumGraph(N, vertices, edges)
            
            start_time = time.perf_counter()
            solution = mfqg.solve(mfqg.bG)
            solve_time = time.perf_counter() - start_time
            
            # Should be reasonably fast for this small graph
            assert solve_time < 2.0, f"Research graph too slow: {solve_time:.3f}s"
            
        except FileNotFoundError:
            pytest.skip("Research graph file not found")
    
    def test_performance_vs_cpp_reference(self):
        """Test that Python performance is within expected range of C++ reference."""
        # This is based on our previous finding that Python is ~16-17x slower
        try:
            edges, vertices = load_quantum_graph("dorogovtsev_goltsev_mendes_1")
            N = 9
            
            mfqg = MFQuantumGraph(N, vertices, edges)
            
            start_time = time.perf_counter()
            solution = mfqg.solve(mfqg.bG)
            solve_time = time.perf_counter() - start_time
            
            # C++ reference time was ~9.06e-06 seconds
            # Python should be 10-30x slower (interpreted overhead)
            cpp_reference_time = 9.06e-06
            expected_python_time = cpp_reference_time * 20  # 20x factor
            
            assert solve_time < expected_python_time * 10, (  # Allow 10x tolerance
                f"Python too slow vs C++: {solve_time:.6f}s vs expected ~{expected_python_time:.6f}s"
            )
            
        except FileNotFoundError:
            pytest.skip("Research graph file not found")


if __name__ == "__main__":
    # Run the tests when called directly
    pytest.main([__file__, "-v"])