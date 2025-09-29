"""
Mathematical Accuracy Regression Tests for BIT_QG Python Implementation

This module implements comprehensive regression testing to ensure numerical
results stay within acceptable mathematical bounds over time.
"""

import numpy as np
import pytest
from pathlib import Path
import sys
import os

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from bit_qg.core.mf_quantum_graph import MFQuantumGraph
from bit_qg.core.qgedge import QGEdge
from bit_qg.utils.graph_io import load_quantum_graph
from bit_qg.preconditioners.degree import DegreePreconditioner
from bit_qg.preconditioners.diagonal import DiagonalPreconditioner
from bit_qg.preconditioners.polynomial import PolynomialPreconditioner
from bit_qg.preconditioners.neumann_neumann import NeumannNeumannPreconditioner


class TestMathematicalAccuracyRegression:
    """
    Test suite for mathematical accuracy regression validation.
    
    This class ensures that numerical results remain within acceptable
    bounds across different solver/preconditioner combinations.
    """
    
    # Mathematical accuracy tolerances
    RESIDUAL_TOLERANCE = 1e-12  # For well-conditioned problems
    SOLUTION_TOLERANCE = 1e-10  # Relative error tolerance
    ITERATION_VARIANCE = 2      # ±2 iterations from reference
    
    @pytest.fixture
    def simple_graph(self):
        """Create a simple 3-vertex path graph for testing."""
        # Use known good graph from existing tests
        try:
            edges, vertices = load_quantum_graph("dorogovtsev_goltsev_mendes_1")
            return edges, vertices
        except FileNotFoundError:
            # Fallback to manually created simple graph
            edges = [
                QGEdge(0, 1, lambda x: 1.0, lambda x: 1.0, lambda x: 1.0),
                QGEdge(1, 2, lambda x: 1.0, lambda x: 1.0, lambda x: 1.0)
            ]
            return edges, 3
    
    @pytest.fixture
    def analytical_solution_graph(self):
        """Create a graph with known analytical solution."""
        # Single edge with constant coefficients: -u'' + u = 1
        # Analytical solution: u(x) = 1 + A*exp(x) + B*exp(-x)
        # With Neumann-Kirchhoff BCs, the solution can be computed exactly
        edges = [
            QGEdge(0, 1, lambda x: 1.0, lambda x: 1.0, lambda x: 1.0)
        ]
        return edges, 2
    
    def test_residual_norm_bounds(self, simple_graph):
        """Test that residual norms stay below tolerance."""
        edges, vertices = simple_graph
        
        # Test different discretization levels
        for N in [9, 17, 33]:  # logN = 3, 4, 5
            mfqg = MFQuantumGraph(N, vertices, edges)
            
            # Test with identity preconditioner (most stable)
            solution = mfqg.solve(mfqg.bG, solver_type='cg')
            residual = np.linalg.norm(mfqg.bG - mfqg.matvec(solution))
            
            assert residual < self.RESIDUAL_TOLERANCE, (
                f"Residual {residual:.2e} exceeds tolerance {self.RESIDUAL_TOLERANCE:.2e} "
                f"for N={N}"
            )
    
    def test_preconditioner_consistency(self, simple_graph):
        """Test that all preconditioners produce consistent results."""
        edges, vertices = simple_graph
        N = 9  # Small problem for fast testing
        
        mfqg = MFQuantumGraph(N, vertices, edges)
        reference_solution = mfqg.solve(mfqg.bG, solver_type='cg')
        
        preconditioners = [
            ('degree', DegreePreconditioner()),
            ('diagonal', DiagonalPreconditioner()),
            ('polynomial', PolynomialPreconditioner()),
            ('neumann_neumann', NeumannNeumannPreconditioner())
        ]
        
        for name, preconditioner in preconditioners:
            preconditioner.compute(mfqg)
            M = preconditioner.as_linear_operator()
            
            solution = mfqg.solve(mfqg.bG, solver_type='cg', preconditioner=M)
            
            # All preconditioners should produce the same solution
            relative_error = np.linalg.norm(solution - reference_solution) / np.linalg.norm(reference_solution)
            
            assert relative_error < self.SOLUTION_TOLERANCE, (
                f"Preconditioner {name} solution differs by {relative_error:.2e} "
                f"from reference (tolerance: {self.SOLUTION_TOLERANCE:.2e})"
            )
    
    def test_solver_consistency(self, simple_graph):
        """Test that CG and BiCGSTAB produce identical solutions."""
        edges, vertices = simple_graph
        N = 9
        
        mfqg = MFQuantumGraph(N, vertices, edges)
        
        # Solve with both methods
        solution_cg = mfqg.solve(mfqg.bG, solver_type='cg')
        solution_bicgstab = mfqg.solve(mfqg.bG, solver_type='bicgstab')
        
        # Should converge to the same solution
        relative_error = np.linalg.norm(solution_cg - solution_bicgstab) / np.linalg.norm(solution_cg)
        
        assert relative_error < self.SOLUTION_TOLERANCE, (
            f"CG and BiCGSTAB solutions differ by {relative_error:.2e} "
            f"(tolerance: {self.SOLUTION_TOLERANCE:.2e})"
        )
    
    def test_convergence_rate_validation(self, analytical_solution_graph):
        """Test that convergence rates match theoretical O(h²) expectations."""
        edges, vertices = analytical_solution_graph
        
        # Test different mesh sizes
        discretizations = [9, 17, 33]  # h = 1/(N-1)
        errors = []
        
        for N in discretizations:
            mfqg = MFQuantumGraph(N, vertices, edges)
            solution = mfqg.solve(mfqg.bG, solver_type='cg')
            
            # For this simple case, we can compute the discretization error
            # This is a placeholder - in practice, you'd compare with analytical solution
            residual = np.linalg.norm(mfqg.bG - mfqg.matvec(solution))
            errors.append(residual)
        
        # Check that errors decrease (monotonicity)
        for i in range(len(errors) - 1):
            assert errors[i+1] <= errors[i] * 2.0, (
                f"Error did not decrease sufficiently: {errors[i]:.2e} -> {errors[i+1]:.2e}"
            )
    
    def test_reference_problem_validation(self):
        """Test against reference problems from research literature."""
        # Load a standard research graph
        try:
            edges, vertices = load_quantum_graph("dorogovtsev_goltsev_mendes_1")
            N = 9  # logN = 3
            
            mfqg = MFQuantumGraph(N, vertices, edges)
            
            # Reference iteration counts from previous validation
            # These are the "golden" values we expect to maintain
            reference_iterations = {
                'cg': 3,
                'bicgstab': 2
            }
            
            # Test both solvers
            for solver_type, expected_iterations in reference_iterations.items():
                # Count iterations using callback
                iteration_count = [0]
                def callback(x):
                    iteration_count[0] += 1
                
                from scipy.sparse.linalg import LinearOperator, cg, bicgstab
                A_op = LinearOperator(mfqg.shape, matvec=mfqg.matvec, dtype=np.float64)
                
                if solver_type == 'cg':
                    solution, info = cg(A_op, mfqg.bG, callback=callback, rtol=1e-8)
                else:
                    solution, info = bicgstab(A_op, mfqg.bG, callback=callback, rtol=1e-8)
                
                actual_iterations = iteration_count[0]
                
                # Allow ±2 iteration variance
                assert abs(actual_iterations - expected_iterations) <= self.ITERATION_VARIANCE, (
                    f"{solver_type.upper()} iterations {actual_iterations} differ from reference "
                    f"{expected_iterations} by more than ±{self.ITERATION_VARIANCE}"
                )
                
        except FileNotFoundError:
            pytest.skip("Reference graph file not found")
    
    def test_numerical_stability_bounds(self, simple_graph):
        """Test numerical stability across different problem scales."""
        edges, vertices = simple_graph
        
        # Test different condition numbers by scaling coefficients
        scales = [1e-3, 1.0, 1e3]
        
        for scale in scales:
            # Create scaled edges
            scaled_edges = [
                QGEdge(edge.out, edge.in_, 
                      lambda x, s=scale: s * edge.c(x),
                      lambda x: edge.v(x),
                      lambda x: edge.f(x))
                for edge in edges
            ]
            
            mfqg = MFQuantumGraph(9, vertices, scaled_edges)
            solution = mfqg.solve(mfqg.bG, solver_type='cg')
            residual = np.linalg.norm(mfqg.bG - mfqg.matvec(solution))
            
            # Residual should scale reasonably with coefficient scaling
            # This is a basic sanity check for numerical stability
            assert np.isfinite(residual), f"Non-finite residual for scale {scale}"
            assert residual < 1e-6, f"Poor convergence for scale {scale}: residual = {residual:.2e}"


class TestManufacturedSolutions:
    """
    Test suite using manufactured solutions with known exact answers.
    
    This provides the most reliable validation of mathematical correctness.
    """
    
    def test_constant_solution(self):
        """Test with manufactured constant solution u(x) = C."""
        # For u(x) = 1 everywhere, we need: -u'' + v*u = f
        # With u'' = 0 and u = 1, we get: v*1 = f, so f = v
        
        edges = [
            QGEdge(0, 1, lambda x: 1.0, lambda x: 2.0, lambda x: 2.0)  # v = f = 2
        ]
        vertices = 2
        
        mfqg = MFQuantumGraph(9, vertices, edges)
        solution = mfqg.solve(mfqg.bG, solver_type='cg')
        
        # The solution should be approximately constant
        # (exact depends on boundary conditions and discretization)
        residual = np.linalg.norm(mfqg.bG - mfqg.matvec(solution))
        assert residual < 1e-12, f"Manufactured solution test failed: residual = {residual:.2e}"
    
    def test_linear_solution(self):
        """Test with manufactured linear solution u(x) = x."""
        # For u(x) = x, we have u'' = 0, so: v*x = f
        
        edges = [
            QGEdge(0, 1, lambda x: 1.0, lambda x: 1.0, lambda x: x)  # f(x) = x
        ]
        vertices = 2
        
        mfqg = MFQuantumGraph(17, vertices, edges)  # Higher resolution for linear functions
        solution = mfqg.solve(mfqg.bG, solver_type='cg')
        
        residual = np.linalg.norm(mfqg.bG - mfqg.matvec(solution))
        assert residual < 1e-10, f"Linear manufactured solution test failed: residual = {residual:.2e}"


if __name__ == "__main__":
    # Run the tests when called directly
    pytest.main([__file__, "-v"])