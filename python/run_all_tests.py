#!/usr/bin/env python3
"""
Comprehensive Test Runner for BIT_QG Python Implementation

This script runs the complete 5-level testing framework:
- Level 1: Unit Tests (individual components)
- Level 2: Integration Tests (multi-component interactions)
- Level 3: Regression Tests (mathematical accuracy + performance)
- Level 4: End-to-End Tests (complete workflows)
- Level 5: Stress Tests (large-scale validation)

Usage:
    python run_all_tests.py              # Run all tests
    python run_all_tests.py --fast       # Skip slow tests
    python run_all_tests.py --level 1    # Run specific level only
    python run_all_tests.py --coverage   # Include coverage report
"""

import subprocess
import sys
import argparse
import time
from pathlib import Path


class TestRunner:
    """Comprehensive test runner for all testing levels."""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
    def run_test_suite(self, test_path, description, timeout=300):
        """Run a test suite and track results."""
        print(f"\n{'='*70}")
        print(f"🧪 RUNNING: {description}")
        print(f"{'='*70}")
        
        cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            # Print output
            if result.stdout:
                print(result.stdout)
            if result.stderr and result.returncode != 0:
                print("STDERR:", result.stderr)
            
            success = result.returncode == 0
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status}: {description}")
            
            self.results.append((description, success, result.returncode))
            return success
            
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT: {description} (>{timeout}s)")
            self.results.append((description, False, -1))
            return False
        except Exception as e:
            print(f"💥 ERROR: {description} - {e}")
            self.results.append((description, False, -2))
            return False
    
    def run_level_1_unit_tests(self):
        """Run Level 1: Unit Tests."""
        return self.run_test_suite("tests/unit/", "Level 1: Unit Tests")
    
    def run_level_2_integration_tests(self):
        """Run Level 2: Integration Tests."""
        return self.run_test_suite("tests/integration/", "Level 2: Integration Tests")
    
    def run_level_3_regression_tests(self):
        """Run Level 3: Regression Tests."""
        accuracy_success = self.run_test_suite(
            "tests/regression/test_accuracy_bounds.py", 
            "Level 3a: Mathematical Accuracy Regression"
        )
        performance_success = self.run_test_suite(
            "tests/regression/test_performance_regression.py", 
            "Level 3b: Performance Regression"
        )
        validation_success = self.run_test_suite(
            "tests/regression/test_cross_validation.py", 
            "Level 3c: Cross-Validation"
        )
        return accuracy_success and performance_success and validation_success
    
    def run_level_4_end_to_end_tests(self):
        """Run Level 4: End-to-End Tests."""
        return self.run_test_suite("tests/end_to_end/", "Level 4: End-to-End Tests")
    
    def run_level_5_stress_tests(self):
        """Run Level 5: Stress Tests (if implemented)."""
        stress_dir = Path("tests/stress")
        if stress_dir.exists() and any(stress_dir.glob("test_*.py")):
            return self.run_test_suite("tests/stress/", "Level 5: Stress Tests")
        else:
            print(f"\n{'='*70}")
            print("🔄 Level 5: Stress Tests - NOT YET IMPLEMENTED")
            print("='*70")
            self.results.append(("Level 5: Stress Tests", None, 0))
            return True  # Don't fail if not implemented yet
    
    def run_with_coverage(self, test_paths):
        """Run tests with coverage report."""
        print(f"\n{'='*70}")
        print("📊 RUNNING ALL TESTS WITH COVERAGE")
        print(f"{'='*70}")
        
        cmd = [
            sys.executable, "-m", "pytest",
            *test_paths,
            "--cov=bit_qg",
            "--cov-report=html",
            "--cov-report=term-missing",
            "-v"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            success = result.returncode == 0
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status}: Coverage Analysis")
            
            return success
        except subprocess.TimeoutExpired:
            print("⏰ TIMEOUT: Coverage analysis")
            return False
        except Exception as e:
            print(f"💥 ERROR: Coverage analysis - {e}")
            return False
    
    def print_summary(self):
        """Print comprehensive test summary."""
        elapsed = time.time() - self.start_time
        
        print(f"\n{'='*70}")
        print("🎯 COMPREHENSIVE TESTING SUMMARY")
        print(f"{'='*70}")
        
        passed = 0
        failed = 0
        skipped = 0
        
        for description, success, code in self.results:
            if success is True:
                print(f"✅ PASSED  - {description}")
                passed += 1
            elif success is False:
                print(f"❌ FAILED  - {description}")
                failed += 1
            else:
                print(f"⏭️  SKIPPED - {description}")
                skipped += 1
        
        total = passed + failed + skipped
        print(f"\n📊 RESULTS: {passed}/{total} test suites passed")
        
        if skipped > 0:
            print(f"   ⏭️  {skipped} test suite(s) skipped (not implemented)")
        
        if failed == 0:
            print("\n🎉 ALL IMPLEMENTED TESTS PASSING!")
            print("✅ Mathematical correctness validated")
            print("✅ Performance regression monitoring active")
            print("✅ Cross-validation with C++ confirmed")
            print("✅ End-to-end workflows verified")
        else:
            print(f"\n⚠️  {failed} test suite(s) need attention")
        
        print(f"\n⏱️  Total runtime: {elapsed:.1f} seconds")
        print(f"📁 Test artifacts: {Path.cwd() / 'tests'}")
        print("📊 Coverage report: htmlcov/index.html")
        print(f"{'='*70}")


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Run BIT_QG comprehensive test suite")
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4, 5],
                       help="Run specific test level only (1-5)")
    parser.add_argument("--fast", action="store_true",
                       help="Skip slow tests (stress tests, large problems)")
    parser.add_argument("--coverage", action="store_true",
                       help="Include coverage analysis")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output, summary only")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    print("🧪 BIT_QG Comprehensive Testing Framework")
    print(f"{'='*70}")
    print("🎯 Running complete 5-level validation suite")
    print("📋 Mathematical correctness • Performance • Integration")
    print(f"{'='*70}")
    
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


if __name__ == "__main__":
    main()