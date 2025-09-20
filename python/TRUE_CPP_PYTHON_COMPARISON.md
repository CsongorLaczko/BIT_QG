# True C++ vs Python Comparison (Identical Problems)
**Generated:** 2025-09-20 23:13:24

## Overview

This report compares C++ and Python implementations solving **EXACTLY THE SAME**
mathematical problems to verify mathematical equivalence. Both use identical
coefficient functions and should produce nearly identical iteration counts.

## Detailed Iteration Comparison

| Graph | LogN | N | Solver | Preconditioner | C++ Iterations | Python Iterations | Iteration Diff | C++ Runtime(s) | Python Runtime(s) | Runtime Ratio | C++ Error | Python Residual | Status |
|-------|------|---|--------|----------------|----------------|-------------------|----------------|----------------|-------------------|---------------|-----------|-----------------|--------|
| dorogovtsev_goltsev_mendes_1 | 3 | 9 | CG | Identity | 0 | 1 | 1 | 0.000041 | 0.000000 | 0.0x | 