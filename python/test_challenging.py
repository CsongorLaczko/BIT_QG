#!/usr/bin/env python3
"""Test Python on challenging problem to compare with C++."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from true_cpp_python_validation import TrueCppPythonValidator

def main():
    validator = TrueCppPythonValidator()
    results = validator.run_python_standard_test('dorogovtsev_goltsev_mendes', 4, 4)
    
    print("Python Results (dorogovtsev_goltsev_mendes_4, logN=4, N=17):")
    print(f"Condition number: {results.get('condition_number', 0):.1e}")
    print()
    
    for key, value in results.items():
        if isinstance(value, dict) and 'iterations' in value:
            print(f"{key:25}: {value['iterations']:2d} iterations, residual: {value['residual']:.2e}")

if __name__ == "__main__":
    main()