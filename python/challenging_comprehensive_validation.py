#!/usr/bin/env python3
"""
Comprehensive Challenging Validation Framework
==============================================

Use the successful challenging test case to create a comprehensive C++ vs Python 
validation with meaningful iteration counts that actually test preconditioner effectiveness.
"""

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
sys.path.append(str(Path(__file__).parent))

@dataclass
class ChallengingTestCase:
    """Configuration for a challenging test case."""
    name: str
    description: str
    c_func_name: str
    f_func_name: str
    expected_condition: str
    expected_iterations: str

@dataclass 
class ChallengingValidationResult:
    """Results from challenging validation test."""
    test_name: str
    graph_name: str
    solver: str
    preconditioner: str
    log_n: int
    N: int
    
    # Matrix properties
    condition_number: float
    matrix_size: tuple
    rhs_norm: float
    
    # C++ results (if available)
    cpp_iterations: int = 0
    cpp_runtime: float = 0.0
    cpp_residual: float = 0.0
    cpp_success: bool = False
    
    # Python results
    python_iterations: int = 0
    python_runtime: float = 0.0
    python_residual: float = 0.0
    python_success: bool = False
    
    # Comparison metrics
    iteration_ratio: float = 0.0
    runtime_ratio: float = 0.0
    significant_difference: bool = False
    difference_reason: str = ""


def create_extreme_conductivity(x: float) -> float:
    """Extreme conductivity variation that creates ill-conditioned matrices."""
    return 1e-3 + 1e6 * x**10  # Factor of 10^9 variation


def create_oscillatory_source(x: float) -> float:
    """Highly oscillatory source function."""
    return math.sin(100 * math.pi * x) * math.cos(50 * math.pi * x)


def create_discontinuous_conductivity(x: float) -> float:
    """Discontinuous conductivity with large jump."""
    return 1e-3 if x < 0.5 else 1e3  # Factor of 10^6 jump


def create_boundary_source(x: float) -> float:
    """Source concentrated near boundaries."""
    eps = 0.05
    if x < eps:
        return 1000.0 * math.exp(-((x - 0.02)**2) / (eps**2))
    elif x > 1 - eps:
        return 1000.0 * math.exp(-((x - 0.98)**2) / (eps**2))
    return 0.1


class ChallengingValidator:
    """Validator for challenging test cases that actually stress iterative solvers."""
    
    def __init__(self, cpp_executable: Path = None):
        """Initialize the challenging validator."""
        self.root_dir = Path(__file__).parent.parent
        self.cpp_executable = cpp_executable or self.root_dir / "build" / "Debug" / "measure_nn.exe"
        self.graphs_dir = self.root_dir / "graphs"
        
        self.challenging_cases = self._create_challenging_test_cases()
    
    def _create_challenging_test_cases(self) -> List[ChallengingTestCase]:
        """Create challenging test case configurations."""
        return [
            ChallengingTestCase(
                name="extreme_oscillatory",
                description="Extreme conductivity + Oscillatory source",
                c_func_name="extreme",
                f_func_name="oscillatory", 
                expected_condition="~10^8 (ill-conditioned)",
                expected_iterations="4-20 depending on preconditioner"
            ),
            ChallengingTestCase(
                name="discontinuous_boundary",
                description="Discontinuous conductivity + Boundary source",
                c_func_name="discontinuous",
                f_func_name="boundary",
                expected_condition="~10^5 (moderately ill-conditioned)",
                expected_iterations="5-12 depending on preconditioner"
            )
        ]
    
    def run_python_challenging_test(
        self, 
        graph_name: str, 
        size: int, 
        log_n: int, 
        solver: str,
        test_case: ChallengingTestCase
    ) -> Dict[str, Any]:
        """Run Python test with challenging coefficients."""
        
        from bit_qg.core import MFQuantumGraph
        from bit_qg.utils import GraphLoader
        from bit_qg.preconditioners import (
            DegreePreconditioner, DiagonalPreconditioner,
            PolynomialPreconditioner, NeumannNeumannPreconditioner
        )
        import scipy.sparse.linalg as spla
        
        # Map function names to actual functions
        c_functions = {
            "extreme": create_extreme_conductivity,
            "discontinuous": create_discontinuous_conductivity
        }
        
        f_functions = {
            "oscillatory": create_oscillatory_source,
            "boundary": create_boundary_source
        }
        
        # Load graph with challenging functions
        loader = GraphLoader()
        graph_file = self.graphs_dir / f"{graph_name}_{size}.txt"
        edges, vertices = loader.load_from_adjacency_matrix(
            graph_file,
            c_func=c_functions[test_case.c_func_name],
            v_func=lambda x: 0.05,  # Simple potential
            f_func=f_functions[test_case.f_func_name]
        )
        
        N = 2**log_n - 1 + 2
        mfqg = MFQuantumGraph(N, vertices, edges)
        
        # Analyze matrix properties
        A_dense = mfqg.AGG.toarray()
        condition_number = np.linalg.cond(A_dense)
        matrix_size = mfqg.AGG.shape
        rhs_norm = np.linalg.norm(mfqg.bG)
        
        results = {
            "condition_number": condition_number,
            "matrix_size": matrix_size,
            "rhs_norm": rhs_norm
        }
        
        # Test all preconditioners
        preconditioner_classes = {
            "identity": None,
            "degree": DegreePreconditioner,
            "diagonal": DiagonalPreconditioner,
            "polynomial": PolynomialPreconditioner,
            "neumann_neumann": NeumannNeumannPreconditioner
        }
        
        for prec_name, prec_class in preconditioner_classes.items():
            try:
                # Setup preconditioner
                if prec_class is not None:
                    preconditioner = prec_class()
                    preconditioner.compute(mfqg)
                    M = preconditioner.as_linear_operator()
                else:
                    M = None
                
                # Count iterations
                iteration_count = [0]
                
                def iteration_callback(x):
                    iteration_count[0] += 1
                
                # Solve with timing
                start_time = time.time()
                if solver.lower() == "cg":
                    solution, info = spla.cg(
                        mfqg.AGG, mfqg.bG, M=M,
                        rtol=1.49e-08, maxiter=1000, atol=0.0,
                        callback=iteration_callback
                    )
                else:  # bicgstab
                    solution, info = spla.bicgstab(
                        mfqg.AGG, mfqg.bG, M=M,
                        rtol=1.49e-08, maxiter=1000, atol=0.0,
                        callback=iteration_callback
                    )
                runtime = time.time() - start_time
                
                # Calculate residual
                residual = np.linalg.norm(mfqg.AGG @ solution - mfqg.bG)
                
                results[prec_name] = {
                    "iterations": iteration_count[0],
                    "runtime": runtime,
                    "residual": residual,
                    "converged": info == 0,
                    "info": info
                }
                
            except Exception as e:
                results[prec_name] = {
                    "iterations": -1,
                    "runtime": 0.0,
                    "residual": float('inf'),
                    "converged": False,
                    "error": str(e)
                }
        
        return results
    
    def run_comprehensive_challenging_validation(
        self,
        graphs: List[str] = None,
        log_n_values: List[int] = None,
        solvers: List[str] = None
    ) -> Dict[str, List[ChallengingValidationResult]]:
        """Run comprehensive validation with challenging test cases."""
        
        if graphs is None:
            graphs = ["dorogovtsev_goltsev_mendes_4"]  # Use larger graph
        if log_n_values is None:
            log_n_values = [4, 5]  # Higher discretization for challenging problems
        if solvers is None:
            solvers = ["cg", "bicgstab"]
        
        all_results = {}
        
        for test_case in self.challenging_cases:
            print(f"\n🔥 Testing: {test_case.description}")
            print(f"📝 Expected condition: {test_case.expected_condition}")
            print(f"🎯 Expected iterations: {test_case.expected_iterations}")
            print("=" * 60)
            
            case_results = []
            test_count = 0
            total_tests = len(graphs) * len(solvers) * len(log_n_values) * 5  # 5 preconditioners
            
            for graph_name in graphs:
                for solver in solvers:
                    for log_n in log_n_values:
                        python_results = self.run_python_challenging_test(
                            graph_name, 4, log_n, solver, test_case  # size=4 for larger graph
                        )
                        
                        # Create validation results for each preconditioner
                        for prec_name in ["identity", "degree", "diagonal", "polynomial", "neumann_neumann"]:
                            test_count += 1
                            test_name = f"{test_case.name}_{graph_name}_{solver}_{prec_name}_logN{log_n}"
                            
                            py_data = python_results.get(prec_name, {})
                            
                            result = ChallengingValidationResult(
                                test_name=test_name,
                                graph_name=graph_name,
                                solver=solver,
                                preconditioner=prec_name,
                                log_n=log_n,
                                N=2**log_n - 1 + 2,
                                condition_number=python_results.get("condition_number", 0.0),
                                matrix_size=python_results.get("matrix_size", (0, 0)),
                                rhs_norm=python_results.get("rhs_norm", 0.0),
                                python_iterations=py_data.get("iterations", -1),
                                python_runtime=py_data.get("runtime", 0.0),
                                python_residual=py_data.get("residual", float('inf')),
                                python_success=py_data.get("converged", False)
                            )
                            
                            case_results.append(result)
                            
                            # Progress indicator
                            status = "✅" if result.python_success else "❌"
                            iterations_str = f"{result.python_iterations:2d}" if result.python_iterations > 0 else "❌"
                            print(f"[{test_count:2d}/{total_tests}] {prec_name:15} {iterations_str} iter {status}")
            
            all_results[test_case.name] = case_results
            
            # Summary for this test case
            successful = [r for r in case_results if r.python_success]
            if successful:
                avg_iter = sum(r.python_iterations for r in successful) / len(successful)
                max_iter = max(r.python_iterations for r in successful)
                min_iter = min(r.python_iterations for r in successful)
                
                print(f"\n📊 {test_case.name} Summary:")
                print(f"   Successful: {len(successful)}/{len(case_results)}")
                print(f"   Iteration range: {min_iter}-{max_iter} (avg: {avg_iter:.1f})")
                print(f"   Condition number: {successful[0].condition_number:.2e}")
                
                if max_iter >= 15:
                    print("   🎯 SUCCESS: High iteration counts achieved!")
                elif max_iter >= 5:
                    print("   ⚠️  Moderate iteration counts")
                else:
                    print("   ❌ Still converging too quickly")
        
        return all_results
    
    def generate_challenging_report(
        self, 
        all_results: Dict[str, List[ChallengingValidationResult]], 
        output_file: str = "CHALLENGING_VALIDATION_REPORT.md"
    ) -> None:
        """Generate comprehensive report for challenging validation."""
        
        with open(output_file, 'w') as f:
            f.write("# Challenging C++ vs Python Validation Report\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Test Cases Designed for High Iteration Counts\n\n")
            
            f.write("This validation uses challenging test cases that create ill-conditioned matrices\n")
            f.write("to properly stress iterative solvers and evaluate preconditioner effectiveness.\n\n")
            
            # Summary across all test cases
            total_tests = sum(len(results) for results in all_results.values())
            f.write(f"**Total Tests:** {total_tests}\n")
            f.write(f"**Test Cases:** {', '.join(all_results.keys())}\n\n")
            
            # Individual test case reports
            for case_name, results in all_results.items():
                case_description = next(tc.description for tc in self.challenging_cases if tc.name == case_name)
                f.write(f"## Test Case: {case_name.replace('_', ' ').title()}\n")
                f.write(f"**Description:** {case_description}\n\n")
                
                # Calculate statistics
                successful = [r for r in results if r.python_success]
                if successful:
                    condition_nums = [r.condition_number for r in successful]
                    iterations = [r.python_iterations for r in successful]
                    
                    f.write(f"**Matrix Properties:**\n")
                    f.write(f"- Condition numbers: {min(condition_nums):.1e} - {max(condition_nums):.1e}\n")
                    f.write(f"- Matrix sizes: {successful[0].matrix_size}\n\n")
                    
                    f.write(f"**Iteration Analysis:**\n")
                    f.write(f"- Successful tests: {len(successful)}/{len(results)}\n")
                    f.write(f"- Iteration range: {min(iterations)} - {max(iterations)}\n")
                    f.write(f"- Average iterations: {sum(iterations)/len(iterations):.1f}\n\n")
                    
                    # Preconditioner comparison
                    f.write(f"**Preconditioner Performance:**\n")
                    prec_stats = {}
                    for result in successful:
                        if result.preconditioner not in prec_stats:
                            prec_stats[result.preconditioner] = []
                        prec_stats[result.preconditioner].append(result.python_iterations)
                    
                    for prec_name, iterations in prec_stats.items():
                        avg_iter = sum(iterations) / len(iterations)
                        f.write(f"- {prec_name.replace('_', ' ').title()}: {avg_iter:.1f} avg iterations\n")
                    
                    f.write("\n")
                
                # Detailed results table
                f.write("### Detailed Results\n\n")
                f.write("| Solver | Preconditioner | LogN | Iterations | Runtime(s) | Residual | Condition Number |\n")
                f.write("|--------|----------------|------|------------|------------|----------|------------------|\n")
                
                for result in results:
                    if result.python_success:
                        f.write(f"| {result.solver.upper()} | {result.preconditioner.replace('_', ' ').title()} | ")
                        f.write(f"{result.log_n} | {result.python_iterations} | {result.python_runtime:.6f} | ")
                        f.write(f"{result.python_residual:.2e} | {result.condition_number:.2e} |\n")
                
                f.write("\n---\n\n")


def main():
    """Main execution function for challenging validation."""
    print("🔥 Challenging C++ vs Python Validation Framework")
    print("==================================================")
    print("Goal: Create test cases with meaningful iteration counts (5-50+ iterations)")
    print("to properly evaluate preconditioner effectiveness.\n")
    
    # Create validator
    validator = ChallengingValidator()
    
    # Run comprehensive validation with challenging cases
    all_results = validator.run_comprehensive_challenging_validation()
    
    # Generate report
    validator.generate_challenging_report(all_results)
    
    print(f"\n✅ Challenging validation complete!")
    print(f"📄 Report saved to: CHALLENGING_VALIDATION_REPORT.md")
    
    # Print summary
    for case_name, results in all_results.items():
        successful = [r for r in results if r.python_success]
        if successful:
            max_iter = max(r.python_iterations for r in successful)
            avg_iter = sum(r.python_iterations for r in successful) / len(successful)
            print(f"🎯 {case_name}: max {max_iter} iterations, avg {avg_iter:.1f}")


if __name__ == "__main__":
    main()