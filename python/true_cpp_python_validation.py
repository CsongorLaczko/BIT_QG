#!/usr/bin/env python3
"""
True C++ vs Python Validation with Identical Problems
====================================================

Compare C++ and Python implementations solving EXACTLY THE SAME mathematical problems
to verify mathematical equivalence. Both should have nearly identical iteration counts.
"""

import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))


class TrueCppPythonValidator:
    """Validator that compares C++ and Python on identical mathematical problems."""
    
    def __init__(self):
        """Initialize the validator."""
        self.root_dir = Path(__file__).parent.parent
        self.cpp_executable = self.root_dir / "build" / "Debug" / "measure_nn.exe"
        self.graphs_dir = self.root_dir / "graphs"
    
    def run_cpp_test(self, graph_name: str, size: int, log_n: int) -> dict:
        """Run C++ measure_nn on specified test case."""
        try:
            cmd = [str(self.cpp_executable), graph_name, str(size), str(log_n), "1"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.root_dir)
            
            if result.returncode != 0:
                return {"error": f"C++ execution failed: {result.stderr}"}
            
            return self._parse_cpp_output(result.stdout)
            
        except Exception as e:
            return {"error": f"C++ execution error: {str(e)}"}
    
    def _parse_cpp_output(self, output: str) -> dict:
        """Parse C++ measure_nn output."""
        results = {}
        lines = output.strip().split('\n')
        
        current_solver = None
        current_preconditioner = None
        
        for line in lines:
            line = line.strip()
            
            # Detect solver
            if line == "CG":
                current_solver = "CG"
                continue
            elif line == "BiCGSTAB":
                current_solver = "BICGSTAB"
                continue
            
            # Detect preconditioner
            elif line == "Vanilla":
                current_preconditioner = "identity"
                continue
            elif line == "Degree":
                current_preconditioner = "degree"
                continue
            elif line == "Diagonal":
                current_preconditioner = "diagonal"
                continue
            elif line == "Polynomial":
                current_preconditioner = "polynomial"
                continue
            elif line == "Neumann-Neumann":
                current_preconditioner = "neumann_neumann"
                continue
            
            # Parse results line
            elif "assembly time:" in line and current_solver and current_preconditioner:
                parts = line.split()
                try:
                    assembly_time = float(parts[2])
                    runtime = float(parts[4])
                    iterations = int(parts[6])
                    error = float(parts[8])
                    
                    key = f"{current_solver}_{current_preconditioner}"
                    results[key] = {
                        "solver": current_solver,
                        "preconditioner": current_preconditioner,
                        "assembly_time": assembly_time,
                        "runtime": runtime,
                        "iterations": iterations,
                        "error": error
                    }
                except (ValueError, IndexError):
                    continue
        
        return results
    
    def run_python_standard_test(self, graph_name: str, size: int, log_n: int) -> dict:
        """Run Python test with STANDARD coefficients (same as C++)."""
        
        import scipy.sparse.linalg as spla

        from bit_qg.core import MFQuantumGraph
        from bit_qg.preconditioners import (
            DegreePreconditioner,
            DiagonalPreconditioner,
            NeumannNeumannPreconditioner,
            PolynomialPreconditioner,
        )
        from bit_qg.utils import GraphLoader
        
        # Load graph with STANDARD coefficients (same as C++)
        loader = GraphLoader()
        graph_file = self.graphs_dir / f"{graph_name}_{size}.txt"
        edges, vertices = loader.load_from_adjacency_matrix(
            graph_file,
            c_func=lambda x: 1.0,  # Constant conductivity (same as C++)
            v_func=lambda x: 0.05, # Simple potential
            f_func=lambda x: math.exp(-1000 * x**2)  # Standard source (same as C++)
        )
        
        N = 2**log_n - 1 + 2
        
        # Time assembly
        start_time = time.time()
        mfqg = MFQuantumGraph(N, vertices, edges)
        assembly_time = time.time() - start_time
        
        # Analyze matrix properties
        A_dense = mfqg.AGG.toarray()
        condition_number = np.linalg.cond(A_dense)
        
        results = {
            "condition_number": condition_number,
            "matrix_size": mfqg.AGG.shape,
            "assembly_time": assembly_time
        }
        
        # Test all preconditioners with both solvers
        preconditioner_classes = {
            "identity": None,
            "degree": DegreePreconditioner,
            "diagonal": DiagonalPreconditioner,
            "polynomial": PolynomialPreconditioner,
            "neumann_neumann": NeumannNeumannPreconditioner
        }
        
        for solver_name in ["CG", "BICGSTAB"]:
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
                    if solver_name == "CG":
                        solution, info = spla.cg(
                            mfqg.AGG, mfqg.bG, M=M,
                            rtol=1.49e-08, maxiter=1000, atol=0.0,
                            callback=iteration_callback
                        )
                    else:  # BICGSTAB
                        solution, info = spla.bicgstab(
                            mfqg.AGG, mfqg.bG, M=M,
                            rtol=1.49e-08, maxiter=1000, atol=0.0,
                            callback=iteration_callback
                        )
                    runtime = time.time() - start_time
                    
                    # Calculate residual
                    residual = np.linalg.norm(mfqg.AGG @ solution - mfqg.bG)
                    
                    key = f"{solver_name}_{prec_name}"
                    results[key] = {
                        "solver": solver_name,
                        "preconditioner": prec_name,
                        "iterations": iteration_count[0],
                        "runtime": runtime,
                        "residual": residual,
                        "converged": info == 0,
                        "info": info
                    }
                    
                except Exception as e:
                    key = f"{solver_name}_{prec_name}"
                    results[key] = {
                        "solver": solver_name,
                        "preconditioner": prec_name,
                        "iterations": -1,
                        "runtime": 0.0,
                        "residual": float('inf'),
                        "converged": False,
                        "error": str(e)
                    }
        
        return results
    
    def generate_true_comparison_report(self, output_file: str = "TRUE_CPP_PYTHON_COMPARISON.md"):
        """Generate comparison report for IDENTICAL mathematical problems."""
        
        print("🔍 True C++ vs Python Validation (Identical Problems)")
        print("=" * 65)
        print("Comparing C++ and Python solving EXACTLY THE SAME problems")
        print("to verify mathematical equivalence.\n")
        
        # Test parameters
        test_params = [
            ("dorogovtsev_goltsev_mendes", 1, 3),
            ("dorogovtsev_goltsev_mendes", 1, 4),
            ("dorogovtsev_goltsev_mendes", 2, 3),
            ("dorogovtsev_goltsev_mendes", 4, 3),
            ("barabasi_albert", 4, 3),
        ]
        
        all_comparisons = []
        
        for graph_name, size, log_n in test_params:
            print(f"📊 Testing: {graph_name}_{size}, logN={log_n}, N={2**log_n - 1 + 2}")
            
            # Run both C++ and Python on IDENTICAL problems
            cpp_results = self.run_cpp_test(graph_name, size, log_n)
            python_results = self.run_python_standard_test(graph_name, size, log_n)
            
            if "error" in cpp_results:
                print(f"   ❌ C++ Error: {cpp_results['error']}")
                continue
                
            comparison = {
                "graph": f"{graph_name}_{size}",
                "log_n": log_n,
                "N": 2**log_n - 1 + 2,
                "python_condition": python_results.get("condition_number", 0),
                "cpp_results": cpp_results,
                "python_results": python_results
            }
            
            all_comparisons.append(comparison)
            
            # Check iteration differences
            significant_differences = []
            
            for solver in ["CG", "BICGSTAB"]:
                for prec in ["identity", "degree", "diagonal", "polynomial", "neumann_neumann"]:
                    cpp_key = f"{solver}_{prec}"
                    py_key = f"{solver}_{prec}"
                    
                    cpp_data = cpp_results.get(cpp_key)
                    py_data = python_results.get(py_key, {})
                    
                    if cpp_data and py_data.get("converged"):
                        cpp_iter = cpp_data["iterations"]
                        py_iter = py_data["iterations"]
                        iter_diff = abs(cpp_iter - py_iter)
                        
                        # Flag significant differences (>1 iteration difference is concerning)
                        if iter_diff > 1:
                            significant_differences.append({
                                "solver": solver,
                                "preconditioner": prec,
                                "cpp_iter": cpp_iter,
                                "py_iter": py_iter,
                                "difference": iter_diff
                            })
                        
                        print(f"   {solver} {prec:15}: C++({cpp_iter:2d}) vs Python({py_iter:2d}) - diff: {iter_diff}")
            
            if significant_differences:
                print(f"   ⚠️  {len(significant_differences)} significant iteration differences found!")
            else:
                print("   ✅ All iteration counts within expected tolerance (≤1 difference)")
            
            print(f"   Condition number: {python_results.get('condition_number', 0):.1e}\n")
        
        # Generate detailed report
        self._write_true_comparison_report(all_comparisons, output_file)
        
        # Summary analysis
        total_tests = 0
        significant_diffs = 0
        max_diff = 0
        
        for comparison in all_comparisons:
            for solver in ["CG", "BICGSTAB"]:
                for prec in ["identity", "degree", "diagonal", "polynomial", "neumann_neumann"]:
                    cpp_data = comparison["cpp_results"].get(f"{solver}_{prec}")
                    py_data = comparison["python_results"].get(f"{solver}_{prec}", {})
                    
                    if cpp_data and py_data.get("converged"):
                        total_tests += 1
                        iter_diff = abs(cpp_data["iterations"] - py_data["iterations"])
                        max_diff = max(max_diff, iter_diff)
                        
                        if iter_diff > 1:
                            significant_diffs += 1
        
        print("=" * 65)
        print("📈 VALIDATION SUMMARY")
        print("=" * 65)
        print(f"Total comparisons: {total_tests}")
        print(f"Significant differences (>1 iter): {significant_diffs}")
        print(f"Maximum iteration difference: {max_diff}")
        print(f"Success rate: {((total_tests - significant_diffs) / total_tests * 100):.1f}%")
        
        if significant_diffs == 0:
            print("✅ EXCELLENT: All iteration counts within tolerance!")
        elif significant_diffs < total_tests * 0.1:
            print("⚠️  ACCEPTABLE: Few significant differences found")
        else:
            print("❌ CONCERNING: Many significant iteration differences!")
        
        print(f"\n📄 Detailed report saved to: {output_file}")
        
        return all_comparisons
    
    def _write_true_comparison_report(self, all_comparisons, output_file: str):
        """Write detailed comparison report for identical problems."""
        
        with open(output_file, 'w') as f:
            f.write("# True C++ vs Python Comparison (Identical Problems)\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Overview\n\n")
            f.write("This report compares C++ and Python implementations solving **EXACTLY THE SAME**\n")
            f.write("mathematical problems to verify mathematical equivalence. Both use identical\n")
            f.write("coefficient functions and should produce nearly identical iteration counts.\n\n")
            
            # Detailed comparison table
            f.write("## Detailed Iteration Comparison\n\n")
            f.write("| Graph | LogN | N | Solver | Preconditioner | ")
            f.write("C++ Iterations | Python Iterations | Iteration Diff | ")
            f.write("C++ Runtime(s) | Python Runtime(s) | Runtime Ratio | ")
            f.write("C++ Error | Python Residual | Status |\n")
            f.write("|-------|------|---|--------|----------------|")
            f.write("----------------|-------------------|----------------|")
            f.write("----------------|-------------------|---------------|")
            f.write("-----------|-----------------|--------|\n")
            
            total_tests = 0
            significant_diffs = 0
            
            for comparison in all_comparisons:
                cpp_results = comparison["cpp_results"]
                python_results = comparison["python_results"]
                
                for solver in ["CG", "BICGSTAB"]:
                    for prec in ["identity", "degree", "diagonal", "polynomial", "neumann_neumann"]:
                        cpp_key = f"{solver}_{prec}"
                        py_key = f"{solver}_{prec}"
                        
                        cpp_data = cpp_results.get(cpp_key)
                        py_data = python_results.get(py_key, {})
                        
                        if cpp_data and py_data.get("converged"):
                            total_tests += 1
                            
                            cpp_iter = cpp_data["iterations"]
                            py_iter = py_data["iterations"]
                            iter_diff = abs(cpp_iter - py_iter)
                            runtime_ratio = py_data["runtime"] / max(cpp_data["runtime"], 1e-9)
                            
                            if iter_diff > 1:
                                significant_diffs += 1
                                status = "⚠️ DIFF"
                            else:
                                status = "✅ OK"
                            
                            f.write(f"| {comparison['graph']} | {comparison['log_n']} | {comparison['N']} | ")
                            f.write(f"{solver} | {prec.replace('_', ' ').title()} | ")
                            f.write(f"{cpp_iter} | {py_iter} | {iter_diff} | ")
                            f.write(f"{cpp_data['runtime']:.6f} | {py_data['runtime']:.6f} | {runtime_ratio:.1f}x | ")
                            f.write(f"{cpp_data['error']:.2e} | {py_data['residual']:.2e} | {status} |\n")
            
            # Analysis section
            f.write("\n## Analysis\n\n")
            f.write("### Summary Statistics\n")
            f.write(f"- **Total Comparisons**: {total_tests}\n")
            f.write(f"- **Significant Differences** (>1 iteration): {significant_diffs}\n")
            if total_tests > 0:
                f.write(f"- **Success Rate**: {((total_tests - significant_diffs) / total_tests * 100):.1f}%\n\n")
            else:
                f.write("- **Success Rate**: No valid comparisons found\n\n")
            
            if significant_diffs == 0:
                f.write("### ✅ **EXCELLENT VALIDATION RESULTS**\n")
                f.write("All iteration counts are within expected tolerance (≤1 difference).\n")
                f.write("This confirms perfect mathematical equivalence between C++ and Python implementations.\n\n")
            else:
                f.write("### ⚠️ **ITERATION DIFFERENCES DETECTED**\n")
                f.write(f"{significant_diffs} test cases show >1 iteration difference.\n")
                f.write("This indicates potential mathematical discrepancies that need investigation.\n\n")
            
            f.write("### Expected vs Actual Behavior\n")
            f.write("- **Expected**: 0-1 iteration difference (due to C++ optimization vs SciPy always-iterate)\n")
            f.write("- **Concerning**: >1 iteration difference suggests algorithmic differences\n")
            f.write("- **Mathematical Equivalence**: Both should converge to same solution accuracy\n\n")


def main():
    """Main execution function."""
    validator = TrueCppPythonValidator()
    
    # Check if C++ executable exists
    if not validator.cpp_executable.exists():
        print(f"❌ C++ executable not found: {validator.cpp_executable}")
        print("   Please build the C++ code first:")
        print("   cmake -S . -B build && cmake --build build")
        return
    
    # Generate true comparison
    validator.generate_true_comparison_report()


if __name__ == "__main__":
    main()