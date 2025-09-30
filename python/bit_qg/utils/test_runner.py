#!/usr/bin/env python3
"""
Comprehensive Test Runner for BIT_QG Python Implementation

This module implements the complete 5-level testing framework:
- Level 1: Unit Tests (individual components)
- Level 2: Integration Tests (multi-component interactions)
- Level 3: Regression Tests (mathematical accuracy + performance)
- Level 4: End-to-End Tests (complete workflows)
- Level 5: Stress Tests (large-scale validation)
"""

import subprocess
import sys
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