"""
End-to-End Workflow Tests for BIT_QG Python Implementation

This module implements comprehensive end-to-end testing of the complete
quantum graph workflow from graph loading to solution validation.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import time

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from bit_qg.core.mf_quantum_graph import MFQuantumGraph
from bit_qg.core.qgedge import QGEdge
from bit_qg.utils.graph_io import load_quantum_graph
from bit_qg.benchmarks.benchmarking import QuantumGraphBenchmark
from bit_qg.preconditioners.degree import DegreePreconditioner
from bit_qg.preconditioners.diagonal import DiagonalPreconditioner
from bit_qg.preconditioners.polynomial import PolynomialPreconditioner
from bit_qg.preconditioners.neumann_neumann import NeumannNeumannPreconditioner


class TestCompleteWorkflow:
    """Test complete workflow from graph loading to solution validation."""
    
    def test_research_graph_complete_workflow(self):
        """Test complete workflow on research graphs."""
        test_cases = [
            ("dorogovtsev_goltsev_mendes_1", 3),  # logN = 3
            ("dorogovtsev_goltsev_mendes_2", 3),
        ]
        
        for graph_name, logN in test_cases:
            try:
                # Step 1: Load graph
                edges, vertices = load_quantum_graph(graph_name)
                assert len(edges) > 0, f"No edges loaded for {graph_name}"
                assert vertices > 0, f"No vertices for {graph_name}"
                
                # Step 2: Create quantum graph system
                N = 2**logN - 1 + 2
                mfqg = MFQuantumGraph(N, vertices, edges)
                
                # Step 3: Verify system properties
                assert mfqg.shape[0] == vertices, "System size mismatch"
                assert mfqg.AGG.shape == (vertices, vertices), "AGG matrix wrong shape"
                
                # Step 4: Solve with all preconditioners
                preconditioners = [
                    (None, "identity"),
                    (DegreePreconditioner(), "degree"),
                    (DiagonalPreconditioner(), "diagonal"),
                    (PolynomialPreconditioner(), "polynomial"),
                    (NeumannNeumannPreconditioner(), "neumann_neumann")
                ]
                
                solutions = {}
                
                for preconditioner, name in preconditioners:
                    if preconditioner is not None:
                        preconditioner.compute(mfqg)
                        M = preconditioner.as_linear_operator()
                    else:
                        M = None
                    
                    # Test both solvers
                    for solver in ['cg', 'bicgstab']:
                        solution = mfqg.solve(mfqg.bG, solver_type=solver, preconditioner=M)
                        
                        # Validate solution
                        assert np.isfinite(solution).all(), f"Non-finite solution: {solver} {name}"
                        
                        residual = np.linalg.norm(mfqg.bG - mfqg.matvec(solution))
                        assert residual < 1e-8, f"Poor convergence: {solver} {name}, residual = {residual:.2e}"
                        
                        solutions[f"{solver}_{name}"] = solution
                
                # Step 5: Verify solution consistency
                # All solutions should be approximately the same
                reference_solution = solutions["cg_identity"]
                for key, solution in solutions.items():
                    relative_error = np.linalg.norm(solution - reference_solution) / np.linalg.norm(reference_solution)
                    assert relative_error < 1e-6, f"Solution inconsistency: {key}, error = {relative_error:.2e}"
                
            except FileNotFoundError:
                pytest.skip(f"Graph {graph_name} not found")


class TestResearchPaperReproduction:
    """Automated reproduction of published research results."""
    
    # Research paper reference data
    RESEARCH_REFERENCES = {
        ('dorogovtsev_goltsev_mendes_5', 6): {
            'cg': {'identity': 27, 'degree': 15, 'diagonal': 14, 'polynomial': 10, 'neumann_neumann': 11}
        },
        ('dorogovtsev_goltsev_mendes_6', 6): {
            'cg': {'identity': 36, 'degree': 15, 'diagonal': 14, 'polynomial': 12, 'neumann_neumann': 12}
        }
    }
    
    def test_research_paper_table_reproduction(self):
        """Automatically reproduce research paper table results."""
        for (graph_name, logN), expected_results in self.RESEARCH_REFERENCES.items():
            try:
                edges, vertices = load_quantum_graph(graph_name)
                N = 2**logN - 1 + 2
                
                benchmark = QuantumGraphBenchmark(edges, vertices)
                
                for solver, solver_expected in expected_results.items():
                    for preconditioner_name, expected_iterations in solver_expected.items():
                        
                        # Create preconditioner
                        if preconditioner_name == 'identity':
                            preconditioner = None
                        elif preconditioner_name == 'degree':
                            preconditioner = DegreePreconditioner()
                        elif preconditioner_name == 'diagonal':
                            preconditioner = DiagonalPreconditioner()
                        elif preconditioner_name == 'polynomial':
                            preconditioner = PolynomialPreconditioner()
                        elif preconditioner_name == 'neumann_neumann':
                            preconditioner = NeumannNeumannPreconditioner()
                        
                        # Run benchmark
                        result = benchmark.benchmark_single_run(N, solver, preconditioner)
                        
                        # Check iteration count
                        iteration_diff = abs(result.iterations - expected_iterations)
                        assert iteration_diff <= 2, (
                            f"Research reproduction failed: {graph_name} {solver} {preconditioner_name} "
                            f"expected {expected_iterations}, got {result.iterations}"
                        )
                        
                        # Check convergence (allow reasonable engineering tolerance)
                        assert result.residual_norm < 1e-6, (
                            f"Poor convergence in research reproduction: {result.residual_norm:.2e}"
                        )
                        
            except FileNotFoundError:
                pytest.skip(f"Research graph {graph_name} not found")


class TestRealWorldUseCases:
    """Test realistic quantum graph applications."""
    
    def test_multiscale_discretization(self):
        """Test problems with varying discretization levels."""
        try:
            edges, vertices = load_quantum_graph("dorogovtsev_goltsev_mendes_1")
            
            # Test different discretization levels
            discretizations = [9, 17, 33, 65]  # logN = 3, 4, 5, 6
            
            solutions = []
            residuals = []
            
            for N in discretizations:
                mfqg = MFQuantumGraph(N, vertices, edges)
                solution = mfqg.solve(mfqg.bG)
                residual = np.linalg.norm(mfqg.bG - mfqg.matvec(solution))
                
                solutions.append(solution)
                residuals.append(residual)
                
                # Each solution should converge
                assert residual < 1e-8, f"Poor convergence for N={N}: residual = {residual:.2e}"
            
            # Residuals should generally improve with finer discretization
            # (though this depends on the specific problem)
            assert all(r < 1e-6 for r in residuals), "Some discretizations failed to converge"
            
        except FileNotFoundError:
            pytest.skip("Test graph not found")
    
    def test_heterogeneous_edge_properties(self):
        """Test graphs with varying edge properties."""
        # Create a graph with different coefficient functions on different edges
        edges = [
            QGEdge(0, 1, lambda x: 1.0, lambda x: 1.0, lambda x: 1.0),      # Standard edge
            QGEdge(1, 2, lambda x: 2.0, lambda x: 0.5, lambda x: x),        # Different coefficients
            QGEdge(2, 3, lambda x: 0.5, lambda x: 2.0, lambda x: np.sin(x)) # Nonlinear source
        ]
        vertices = 4
        
        mfqg = MFQuantumGraph(9, vertices, edges)
        solution = mfqg.solve(mfqg.bG)
        
        # Should still converge despite heterogeneous properties
        residual = np.linalg.norm(mfqg.bG - mfqg.matvec(solution))
        assert residual < 1e-8, f"Heterogeneous edge test failed: residual = {residual:.2e}"
        assert np.isfinite(solution).all(), "Non-finite solution with heterogeneous edges"
    
    def test_parameter_sensitivity(self):
        """Test sensitivity to parameter variations."""
        try:
            edges, vertices = load_quantum_graph("dorogovtsev_goltsev_mendes_1")
            
            # Test different solver tolerances
            tolerances = [1e-6, 1e-8, 1e-10, 1e-12]
            
            for rtol in tolerances:
                mfqg = MFQuantumGraph(9, vertices, edges)
                solution = mfqg.solve(mfqg.bG, rtol=rtol)
                residual = np.linalg.norm(mfqg.bG - mfqg.matvec(solution))
                
                # Residual should be approximately at the requested tolerance
                assert residual <= rtol * 100, (  # Allow some factor for relative vs absolute tolerance
                    f"Tolerance not met: requested {rtol:.2e}, got {residual:.2e}"
                )
            
        except FileNotFoundError:
            pytest.skip("Test graph not found")


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    def test_invalid_graph_handling(self):
        """Test handling of invalid graph configurations."""
        # Empty graph should create zero-sized system
        mfqg = MFQuantumGraph(9, 0, [])
        assert mfqg.vertices == 0
        assert len(mfqg.edges) == 0
        
        # Very small graphs should work
        edges = [QGEdge(0, 1, lambda x: 1.0, lambda x: 1.0, lambda x: 1.0)]
        mfqg = MFQuantumGraph(9, 2, edges)
        solution = mfqg.solve(mfqg.bG)
        assert solution is not None
        assert np.isfinite(solution).all(), "Invalid handling of disconnected vertices"
    
    def test_solver_failure_recovery(self):
        """Test recovery from solver convergence failures."""
        # Create an ill-conditioned problem
        edges = [
            QGEdge(0, 1, lambda x: 1e-12, lambda x: 1e12, lambda x: 1.0)  # Very ill-conditioned
        ]
        vertices = 2
        
        mfqg = MFQuantumGraph(9, vertices, edges)
        
        # Solver might fail, but should handle gracefully
        try:
            solution = mfqg.solve(mfqg.bG, maxiter=10)  # Low iteration limit
            # If it succeeds, solution should be finite
            assert np.isfinite(solution).all(), "Non-finite solution from ill-conditioned problem"
        except (RuntimeError, Warning):
            # Failure is acceptable for ill-conditioned problems
            pass
    
    def test_output_validation(self):
        """Test validation of output data consistency."""
        try:
            edges, vertices = load_quantum_graph("dorogovtsev_goltsev_mendes_1")
            
            benchmark = QuantumGraphBenchmark(edges, vertices)
            result = benchmark.benchmark_single_run(9, 'cg', None)
            
            # Validate output data consistency
            assert result.n_points == 9, "Incorrect n_points in result"
            assert result.solver_name == 'cg', "Incorrect solver name"
            assert result.preconditioner_name == 'identity', "Incorrect preconditioner name"
            assert result.iterations > 0, "Invalid iteration count"
            assert result.total_time >= result.solve_time, "Invalid timing data"
            assert np.isfinite(result.residual_norm), "Non-finite residual norm"
            assert result.success, "Benchmark marked as failed despite successful completion"
            
        except FileNotFoundError:
            pytest.skip("Test graph not found")


if __name__ == "__main__":
    # Run the tests when called directly
    pytest.main([__file__, "-v"])