#!/usr/bin/env python3
"""
Comprehensive validation suite comparing C++ and Python implementations.

This suite tests all combinations of:
- Solvers: CG, BiCGSTAB
- Preconditioners: Identity, Degree, Diagonal, Polynomial, Neumann-Neumann  
- Graphs: All available graphs in graphs/ directory
- Problem sizes: Multiple discretization levels

Generates detailed comparison tables and identifies significant differences.
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add Python package to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from bit_qg.benchmarks import QuantumGraphBenchmark
from bit_qg.preconditioners import (
    DegreePreconditioner,
    DiagonalPreconditioner, 
    NeumannNeumannPreconditioner,
    PolynomialPreconditioner,
)
from bit_qg.utils import GraphLoader


@dataclass
class TestCase:
    """Individual test case configuration."""
    graph_name: str
    size: int
    log_n: int
    solver: str
    preconditioner: str
    
    @property
    def n_points(self) -> int:
        """Calculate N using C++ formula: N = pow(2, logN) - 1 + 2"""
        return 2**self.log_n - 1 + 2
    
    @property 
    def test_id(self) -> str:
        """Unique identifier for this test case."""
        return f"{self.graph_name}_{self.size}_{self.solver}_{self.preconditioner}_logN{self.log_n}"


@dataclass
class ValidationResult:
    """Results from comparing C++ and Python outputs."""
    test_case: TestCase
    cpp_assembly_time: float
    cpp_runtime: float
    cpp_iterations: int
    cpp_error: float
    python_assembly_time: float
    python_runtime: float
    python_iterations: int
    python_error: float
    
    # Status flags
    cpp_success: bool = True
    python_success: bool = True
    significant_difference: bool = False
    difference_reason: str = ""
    
    # Derived comparison metrics (calculated in __post_init__)
    assembly_time_ratio: float = 0.0
    runtime_ratio: float = 0.0
    iteration_diff: int = 0
    error_ratio: float = 0.0
    max_error: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics and detect significant differences."""
        # Calculate ratios (handle division by zero)
        self.assembly_time_ratio = (
            self.python_assembly_time / self.cpp_assembly_time 
            if self.cpp_assembly_time > 0 else float('inf')
        )
        self.runtime_ratio = (
            self.python_runtime / self.cpp_runtime 
            if self.cpp_runtime > 0 else float('inf')
        )
        self.iteration_diff = self.python_iterations - self.cpp_iterations
        self.error_ratio = (
            self.python_error / self.cpp_error 
            if self.cpp_error > 0 else float('inf')
        )
        self.max_error = max(self.cpp_error, self.python_error)
        
        # Detect significant differences and provide reasons
        self._analyze_differences()
    
    def _analyze_differences(self):
        """Analyze results for significant differences and provide explanations."""
        reasons = []
        
        # Check iteration differences
        if abs(self.iteration_diff) > 2:
            self.significant_difference = True
            reasons.append(f"Large iteration difference: {self.iteration_diff}")
        elif abs(self.iteration_diff) == 1:
            reasons.append("Expected: C++ optimization vs SciPy behavior")
        
        # Check error magnitude differences (more lenient)
        if self.max_error > 1e-6:
            self.significant_difference = True
            reasons.append(f"High residual error: {self.max_error:.2e}")
        elif self.max_error > 1e-12:
            reasons.append("Normal tolerance-level residual")
        elif self.max_error > 1e-14:
            reasons.append("Machine precision residual")
        
        # Check convergence failures
        if not self.cpp_success:
            self.significant_difference = True
            reasons.append("C++ solver failed")
        if not self.python_success:
            self.significant_difference = True
            reasons.append("Python solver failed")
        
        # Check extreme performance differences
        if self.runtime_ratio > 100 or self.runtime_ratio < 0.01:
            reasons.append(f"Extreme runtime ratio: {self.runtime_ratio:.1f}x")
        
        self.difference_reason = "; ".join(reasons) if reasons else "No significant differences"


class ComprehensiveValidator:
    """Main validation suite comparing C++ and Python implementations."""
    
    def __init__(self, cpp_executable: Path = None):
        """Initialize validator with C++ executable path."""
        self.root_dir = Path(__file__).parent.parent  # Go up one level from python/ to project root
        self.cpp_executable = cpp_executable or self.root_dir / "build" / "Debug" / "measure_nn.exe"
        self.python_dir = self.root_dir / "python"
        self.graphs_dir = self.root_dir / "graphs"
        
        # Verify C++ executable exists
        if not self.cpp_executable.exists():
            raise FileNotFoundError(f"C++ executable not found: {self.cpp_executable}")
        
        # Load Python components
        self.graph_loader = GraphLoader()
        
        # Available test configurations
        self.solvers = ["cg", "bicgstab"]
        self.preconditioners = {
            "identity": None,
            "degree": DegreePreconditioner,
            "diagonal": DiagonalPreconditioner,
            "polynomial": PolynomialPreconditioner,
            "neumann_neumann": NeumannNeumannPreconditioner,
        }
        
    def discover_graphs(self) -> List[tuple[str, int]]:
        """Discover all available graph files."""
        graph_files = []
        for graph_file in self.graphs_dir.glob("*.txt"):
            if graph_file.name == "generate.py":
                continue
            
            # Parse filename: {graph_name}_{size}.txt
            stem = graph_file.stem
            if "_" in stem:
                parts = stem.split("_")
                if parts[-1].isdigit():
                    graph_name = "_".join(parts[:-1])
                    size = int(parts[-1])
                    graph_files.append((graph_name, size))
        
        return sorted(set(graph_files))
    
    def run_cpp_test(self, graph_name: str, size: int, log_n: int, solver: str = "cg") -> Dict:
        """Run a single C++ test case and parse results."""
        try:
            # Run C++ executable
            result = subprocess.run(
                [str(self.cpp_executable), graph_name, str(size), str(log_n), "1"],
                capture_output=True,
                text=True,
                cwd=self.root_dir,
                timeout=30
            )
            
            if result.returncode != 0:
                return {"success": False, "error": result.stderr}
            
            # Parse C++ output
            lines = result.stdout.strip().split('\n')
            cpp_results = {}
            
            current_solver = None
            current_preconditioner = None
            
            for line in lines:
                line = line.strip()
                if line == "CG":
                    current_solver = "cg"
                elif line == "BiCGSTAB":
                    current_solver = "bicgstab"
                elif line in ["Vanilla", "Degree", "Diagonal", "Polynomial", "Neumann-Neumann"]:
                    current_preconditioner = line.lower().replace("-", "_")
                elif line.startswith("assembly time:"):
                    # Parse: assembly time: 0.000275 runtime: 5.76e-05 iterations: 0 error: 0
                    parts = line.split()
                    assembly_time = float(parts[2])
                    runtime = float(parts[4])
                    iterations = int(parts[6])
                    error = float(parts[8])
                    
                    if current_solver and current_preconditioner:
                        if current_solver not in cpp_results:
                            cpp_results[current_solver] = {}
                        
                        # Map preconditioner names
                        prec_name = "identity" if current_preconditioner == "vanilla" else current_preconditioner
                        
                        cpp_results[current_solver][prec_name] = {
                            "assembly_time": assembly_time,
                            "runtime": runtime, 
                            "iterations": iterations,
                            "error": error,
                            "success": True
                        }
            
            return {"success": True, "results": cpp_results}
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "C++ execution timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_python_test(self, graph_name: str, size: int, log_n: int, solver: str) -> Dict:
        """Run a single Python test case and return results."""
        try:
            # Load graph
            graph_file = f"{graph_name}_{size}"
            edges, vertices = self.graph_loader.load_from_graph_file(graph_file)
            
            # Create benchmark
            benchmark = QuantumGraphBenchmark(edges, vertices)
            n_points = 2**log_n - 1 + 2
            
            # Create temporary problem for preconditioner computation
            from bit_qg.core import MFQuantumGraph
            temp_problem = MFQuantumGraph(n_points, vertices, edges)
            
            # Test all preconditioners
            python_results = {}
            
            for prec_name, prec_class in self.preconditioners.items():
                try:
                    # Create preconditioner instance
                    if prec_class is None:
                        preconditioner = None
                    else:
                        preconditioner = prec_class()
                        preconditioner.compute(temp_problem)
                    
                    # Run benchmark
                    result = benchmark.benchmark_single_run(
                        n_points=n_points,
                        solver_name=solver,
                        preconditioner=preconditioner,
                        graph_name=graph_file
                    )
                    
                    python_results[prec_name] = {
                        "assembly_time": result.assembly_time,
                        "runtime": result.solve_time,
                        "iterations": result.iterations,
                        "error": result.residual_norm,
                        "success": result.success
                    }
                    
                except Exception as e:
                    python_results[prec_name] = {
                        "assembly_time": 0.0,
                        "runtime": 0.0,
                        "iterations": 0,
                        "error": float('inf'),
                        "success": False,
                        "error_msg": str(e)
                    }
            
            return {"success": True, "results": python_results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_comprehensive_validation(self, log_n_values: List[int] = [3, 4]) -> List[ValidationResult]:
        """Run comprehensive validation across all test cases."""
        print("🔍 Starting Comprehensive C++ vs Python Validation Suite")
        print("=" * 60)
        
        # Discover available graphs
        graphs = self.discover_graphs()
        print(f"📊 Found {len(graphs)} graph files:")
        for graph_name, size in graphs:
            print(f"  - {graph_name}_{size}")
        print()
        
        # Generate all test cases
        test_cases = []
        for graph_name, size in graphs:
            for log_n in log_n_values:
                for solver in self.solvers:
                    for prec_name in self.preconditioners.keys():
                        test_cases.append(TestCase(
                            graph_name=graph_name,
                            size=size,
                            log_n=log_n,
                            solver=solver,
                            preconditioner=prec_name
                        ))
        
        print(f"🧪 Total test cases: {len(test_cases)}")
        print(f"📈 Testing discretization levels: {log_n_values} (N = {[2**x-1+2 for x in log_n_values]})")
        print()
        
        # Run validation
        validation_results = []
        failed_tests = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i:3d}/{len(test_cases)}] {test_case.test_id:<50}", end=" ")
            
            try:
                # Run C++ test
                cpp_result = self.run_cpp_test(
                    test_case.graph_name, test_case.size, test_case.log_n, test_case.solver
                )
                
                # Run Python test  
                python_result = self.run_python_test(
                    test_case.graph_name, test_case.size, test_case.log_n, test_case.solver
                )
                
                # Compare results
                if (cpp_result["success"] and python_result["success"] and
                    test_case.solver in cpp_result["results"] and
                    test_case.preconditioner in cpp_result["results"][test_case.solver] and
                    test_case.preconditioner in python_result["results"]):
                    
                    cpp_data = cpp_result["results"][test_case.solver][test_case.preconditioner]
                    python_data = python_result["results"][test_case.preconditioner]
                    
                    validation_result = ValidationResult(
                        test_case=test_case,
                        cpp_assembly_time=cpp_data["assembly_time"],
                        cpp_runtime=cpp_data["runtime"],
                        cpp_iterations=cpp_data["iterations"],
                        cpp_error=cpp_data["error"],
                        python_assembly_time=python_data["assembly_time"],
                        python_runtime=python_data["runtime"],
                        python_iterations=python_data["iterations"],
                        python_error=python_data["error"],
                        cpp_success=cpp_data["success"],
                        python_success=python_data["success"]
                    )
                    
                    validation_results.append(validation_result)
                    
                    # Status indicator
                    if validation_result.significant_difference:
                        print("⚠️  DIFF")
                    else:
                        print("✅ OK")
                else:
                    failed_tests.append((test_case, cpp_result, python_result))
                    print("❌ FAIL")
                    
            except Exception as e:
                failed_tests.append((test_case, str(e), None))
                print(f"💥 ERROR: {e}")
        
        print(f"\n📋 Completed: {len(validation_results)} successful, {len(failed_tests)} failed")
        return validation_results, failed_tests
    
    def generate_comparison_tables(self, validation_results: List[ValidationResult]) -> str:
        """Generate comprehensive comparison tables."""
        if not validation_results:
            return "No validation results to analyze."
        
        # Convert to DataFrame for analysis
        data = []
        for result in validation_results:
            data.append({
                "Graph": f"{result.test_case.graph_name}_{result.test_case.size}",
                "LogN": result.test_case.log_n,
                "N": result.test_case.n_points,
                "Solver": result.test_case.solver.upper(),
                "Preconditioner": result.test_case.preconditioner.replace("_", "-").title(),
                "C++_Assembly(s)": f"{result.cpp_assembly_time:.2e}",
                "Python_Assembly(s)": f"{result.python_assembly_time:.2e}",
                "Assembly_Ratio": f"{result.assembly_time_ratio:.1f}x",
                "C++_Runtime(s)": f"{result.cpp_runtime:.2e}",
                "Python_Runtime(s)": f"{result.python_runtime:.2e}",
                "Runtime_Ratio": f"{result.runtime_ratio:.1f}x",
                "C++_Iterations": result.cpp_iterations,
                "Python_Iterations": result.python_iterations,
                "Iter_Diff": result.iteration_diff,
                "C++_Error": f"{result.cpp_error:.2e}",
                "Python_Error": f"{result.python_error:.2e}",
                "Max_Error": f"{result.max_error:.2e}",
                "Significant": "DIFF" if result.significant_difference else "OK",
                "Reason": result.difference_reason
            })
        
        df = pd.DataFrame(data)
        
        # Generate comprehensive report
        report = []
        report.append("# Comprehensive C++ vs Python Validation Report")
        report.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Tests:** {len(validation_results)}")
        report.append("")
        
        # Summary statistics
        significant_count = sum(1 for r in validation_results if r.significant_difference)
        report.append("## Summary Statistics")
        report.append(f"- **Successful Tests:** {len(validation_results)}")
        report.append(f"- **Significant Differences:** {significant_count} ({100*significant_count/len(validation_results):.1f}%)")
        report.append(f"- **Graphs Tested:** {len(df['Graph'].unique())}")
        report.append(f"- **Solvers Tested:** {', '.join(df['Solver'].unique())}")
        report.append(f"- **Preconditioners Tested:** {', '.join(df['Preconditioner'].unique())}")
        report.append("")
        
        # Performance summary
        avg_assembly_ratio = df['Assembly_Ratio'].str.replace('x', '').astype(float).mean()
        avg_runtime_ratio = df['Runtime_Ratio'].str.replace('x', '').astype(float).mean()
        report.append("## Performance Summary")
        report.append(f"- **Average Assembly Time Ratio (Python/C++):** {avg_assembly_ratio:.1f}x")
        report.append(f"- **Average Runtime Ratio (Python/C++):** {avg_runtime_ratio:.1f}x")
        report.append("")
        
        # Detailed results table
        report.append("## Detailed Comparison Table")
        report.append("")
        report.append(df.to_markdown(index=False, tablefmt="pipe"))
        report.append("")
        
        # Significant differences analysis
        if significant_count > 0:
            significant_df = df[df['Significant'] == 'DIFF']
            report.append("## Significant Differences Analysis")
            report.append("")
            report.append(significant_df[['Graph', 'Solver', 'Preconditioner', 'Iter_Diff', 'Max_Error', 'Reason']].to_markdown(index=False, tablefmt="pipe"))
            report.append("")
        
        # Iteration difference patterns
        report.append("## Iteration Difference Patterns")
        iter_diff_counts = df['Iter_Diff'].value_counts().sort_index()
        for diff, count in iter_diff_counts.items():
            percentage = 100 * count / len(df)
            report.append(f"- **{diff:+d} iterations:** {count} cases ({percentage:.1f}%)")
        report.append("")
        
        # Error analysis
        report.append("## Error Analysis")
        max_errors = df['Max_Error'].str.replace('e', 'E').astype(float)
        report.append(f"- **Maximum Error:** {max_errors.max():.2e}")
        report.append(f"- **Mean Error:** {max_errors.mean():.2e}")
        report.append(f"- **Median Error:** {max_errors.median():.2e}")
        report.append(f"- **Errors > 1e-12:** {(max_errors > 1e-12).sum()} cases")
        report.append("")
        
        return "\n".join(report)


def main():
    """Main execution function."""
    print("🚀 C++ vs Python Comprehensive Validation Suite")
    print("=" * 50)
    
    try:
        # Initialize validator
        validator = ComprehensiveValidator()
        
        # Run comprehensive validation
        validation_results, failed_tests = validator.run_comprehensive_validation(
            log_n_values=[3, 4, 5]  # Test multiple problem sizes
        )
        
        if failed_tests:
            print(f"\n⚠️  {len(failed_tests)} tests failed:")
            for test_case, error, _ in failed_tests[:5]:  # Show first 5 failures
                print(f"   - {test_case.test_id}: {error}")
            if len(failed_tests) > 5:
                print(f"   ... and {len(failed_tests) - 5} more")
        
        # Generate report
        if validation_results:
            print("\n📊 Generating comparison report...")
            report = validator.generate_comparison_tables(validation_results)
            
            # Save report
            report_file = Path("COMPREHENSIVE_VALIDATION_REPORT.md")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"✅ Report saved to: {report_file}")
            print(f"📈 {len(validation_results)} test cases analyzed")
            
            # Show brief summary
            significant_count = sum(1 for r in validation_results if r.significant_difference)
            print(f"🎯 Significant differences: {significant_count}/{len(validation_results)} ({100*significant_count/len(validation_results):.1f}%)")
        else:
            print("❌ No successful validation results to report")
    
    except Exception as e:
        print(f"💥 Validation suite failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())