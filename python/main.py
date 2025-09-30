#!/usr/bin/env python3
"""
BIT_QG Python Implementation - Main Entry Point

This is the single entry point for all BIT_QG Python operations.

Usage:
    python main.py test [options]           # Run tests
    python main.py benchmark <graph> <size> <logN> [runs]  # C++ validation benchmark
    python main.py demo [preconditioner]    # Run demonstrations
    python main.py validate                 # Generate validation reference data
    python main.py --help                   # Show help

Examples:
    python main.py test --level 1                          # Run unit tests only
    python main.py test --coverage                         # Run tests with coverage
    python main.py benchmark dorogovtsev_goltsev_mendes 1 3 1  # C++ validation
    python main.py demo degree                             # Demo degree preconditioner
    python main.py validate                                # Generate reference data
"""

import sys
import argparse
from pathlib import Path

# Add the bit_qg package to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging for debugging
from bit_qg.utils.logging_config import setup_logging


def run_tests(args):
    """Run the comprehensive test suite."""
    from bit_qg.utils.test_runner import TestRunner
    
    runner = TestRunner()
    
    print("🧪 BIT_QG Comprehensive Testing Framework")
    print("=" * 70)
    print("🎯 Running complete 5-level validation suite")
    print("📋 Mathematical correctness • Performance • Integration")
    print("=" * 70)
    
    # Run specific level if requested
    if args.level:
        if args.level == 1:
            runner.run_level_1_unit_tests()
        elif args.level == 2:
            runner.run_level_2_integration_tests()
        elif args.level == 3:
            runner.run_level_3_regression_tests()
        elif args.level == 4:
            runner.run_level_4_end_to_end_tests()
        elif args.level == 5:
            runner.run_level_5_stress_tests()
    else:
        # Run all levels
        runner.run_level_1_unit_tests()
        runner.run_level_2_integration_tests()
        runner.run_level_3_regression_tests()
        runner.run_level_4_end_to_end_tests()
        
        if not args.fast:
            runner.run_level_5_stress_tests()
    
    # Run with coverage if requested
    if args.coverage:
        test_paths = ["tests/unit", "tests/integration", "tests/regression", "tests/end_to_end"]
        if Path("tests/stress").exists() and not args.fast:
            test_paths.append("tests/stress")
        runner.run_with_coverage(test_paths)
    
    # Print summary
    if not args.quiet:
        runner.print_summary()


def run_benchmark(args):
    """Run C++ validation benchmark."""
    from bit_qg.utils.cpp_validator import run_cpp_validation
    
    # Parse arguments
    graph = args.graph
    size = args.size
    log_n = args.logN
    runs = args.runs or 1
    
    # Enable logging to see fallback cases
    setup_logging()
    
    # Run validation
    success = run_cpp_validation(graph, size, log_n, runs)
    sys.exit(0 if success else 1)


def run_demo(args):
    """Run demonstration scripts."""
    from bit_qg.utils.demo_runner import run_demo
    
    preconditioner = args.preconditioner or "all"
    run_demo(preconditioner)


def run_validate(args):
    """Generate validation reference data."""
    from bit_qg.utils.reference_generator import generate_reference_results
    
    print("🔄 Generating reference validation data...")
    results = generate_reference_results()
    
    # Print summary
    print("\nSummary:")
    for graph_key, graph_data in results.items():
        print(f"{graph_key}:")
        for solver, solver_data in graph_data.items():
            print(f"  {solver}:")
            for prec_name, prec_data in solver_data.items():
                status = "✅" if prec_data['success'] else "❌"
                print(f"    {prec_name}: {prec_data['iterations']} iter, {prec_data['error']:.2e} error {status}")


def main():
    """Main entry point with command routing."""
    parser = argparse.ArgumentParser(
        description="BIT_QG Python Implementation - Single Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run comprehensive test suite")
    test_parser.add_argument("--level", type=int, choices=[1, 2, 3, 4, 5],
                           help="Run specific test level only (1-5)")
    test_parser.add_argument("--fast", action="store_true",
                           help="Skip slow tests (stress tests, large problems)")
    test_parser.add_argument("--coverage", action="store_true",
                           help="Include coverage analysis")
    test_parser.add_argument("--quiet", action="store_true",
                           help="Minimal output, summary only")
    
    # Benchmark command (C++ validation)
    benchmark_parser = subparsers.add_parser("benchmark", help="Run C++ validation benchmark")
    benchmark_parser.add_argument("graph", help="Graph type (e.g., dorogovtsev_goltsev_mendes)")
    benchmark_parser.add_argument("size", type=int, help="Graph size parameter")
    benchmark_parser.add_argument("logN", type=int, help="Discretization parameter (N = 2^logN - 1 + 2)")
    benchmark_parser.add_argument("runs", type=int, nargs="?", help="Number of benchmark runs (default: 1)")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstrations")
    demo_parser.add_argument("preconditioner", nargs="?", 
                           choices=["degree", "diagonal", "polynomial", "neumann_neumann", "all"],
                           help="Preconditioner to demonstrate (default: all)")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Generate validation reference data")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate handler
    if args.command == "test":
        run_tests(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "demo":
        run_demo(args)
    elif args.command == "validate":
        run_validate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()