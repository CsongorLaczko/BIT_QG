#!/usr/bin/env python3
"""
Quick Enhanced Validation Test
=============================

Simple test to validate different source functions and check if we get
meaningful iteration counts.
"""

import math
import time
import numpy as np
from pathlib import Path

# Add current directory to path to import our modules
import sys
sys.path.append(str(Path(__file__).parent))

def test_different_source_functions():
    """Test different source functions to see iteration behavior."""
    from bit_qg.core import MFQuantumGraph
    from bit_qg.utils import GraphLoader
    from bit_qg.preconditioners import DegreePreconditioner
    import scipy.sparse.linalg as spla
    
    # Define different source functions
    source_functions = {
        "trivial": lambda x: 1.0 * math.exp(-((x - 0.0)**2) * 250 * 4),  # Original sharp
        "moderate": lambda x: 1.0 * math.exp(-((x - 0.5)**2) * 5),        # Broader peak
        "challenging": lambda x: math.sin(10 * math.pi * x) * math.exp(-2 * x),  # Oscillatory
        "constant": lambda x: 1.0,  # Constant source
        "linear": lambda x: x,  # Linear source
    }
    
    # Load a small graph
    loader = GraphLoader()
    graph_file = Path(__file__).parent.parent / "graphs" / "dorogovtsev_goltsev_mendes_1.txt"
    
    print("🧪 Testing Different Source Functions")
    print("=" * 50)
    
    results = {}
    
    for name, f_func in source_functions.items():
        print(f"\n📊 Testing: {name}")
        print(f"   Function: {f_func.__name__ if hasattr(f_func, '__name__') else 'lambda'}")
        
        try:
            # Load graph with custom source function
            edges, vertices = loader.load_from_adjacency_matrix(
                graph_file,
                c_func=lambda x: 1.0 / (1.0 + math.exp(-25 * (x - 0.5))) + 1.0,
                v_func=lambda x: 0.05 / (0.2**2) * (abs(x - 0.5) - 0.2)**2 + 0.05,
                f_func=f_func
            )
            
            N = 9  # Small test case (logN=3)
            
            # Create quantum graph
            mfqg = MFQuantumGraph(N, vertices, edges)
            
            # Check the right-hand side magnitude
            rhs_norm = np.linalg.norm(mfqg.bG)
            rhs_max = np.max(np.abs(mfqg.bG))
            
            print(f"   RHS norm: {rhs_norm:.6e}")
            print(f"   RHS max:  {rhs_max:.6e}")
            
            if rhs_norm < 1e-12:
                print("   ⚠️  RHS is essentially zero - trivial problem")
                continue
            
            # Test with identity (no preconditioner)
            print("   Identity solver:")
            start_time = time.time()
            solution, info = spla.cg(mfqg.AGG, mfqg.bG, rtol=1.49e-08)
            runtime = time.time() - start_time
            residual = np.linalg.norm(mfqg.AGG @ solution - mfqg.bG)
            
            print(f"     Converged: {info == 0}")
            print(f"     Runtime: {runtime:.6f}s")
            print(f"     Residual: {residual:.6e}")
            
            # Test with degree preconditioner
            print("   Degree preconditioner:")
            preconditioner = DegreePreconditioner()
            preconditioner.compute(mfqg)
            M = preconditioner.as_linear_operator()
            
            start_time = time.time()
            solution, info = spla.cg(mfqg.AGG, mfqg.bG, M=M, rtol=1.49e-08)
            runtime = time.time() - start_time
            residual = np.linalg.norm(mfqg.AGG @ solution - mfqg.bG)
            
            print(f"     Converged: {info == 0}")
            print(f"     Runtime: {runtime:.6f}s")
            print(f"     Residual: {residual:.6e}")
            
            results[name] = {
                "rhs_norm": rhs_norm,
                "rhs_max": rhs_max,
                "converged": info == 0,
                "runtime": runtime,
                "residual": residual
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[name] = {"error": str(e)}
    
    print("\n📈 Summary:")
    print("-" * 30)
    for name, data in results.items():
        if "error" in data:
            print(f"{name:12}: ERROR - {data['error']}")
        else:
            print(f"{name:12}: RHS norm {data['rhs_norm']:.2e}, Runtime {data['runtime']:.6f}s")
    
    return results


def test_random_rhs():
    """Test with random right-hand side to see true solver behavior."""
    from bit_qg.core import MFQuantumGraph
    from bit_qg.utils import GraphLoader
    from bit_qg.preconditioners import DegreePreconditioner
    import scipy.sparse.linalg as spla
    
    print("\n🎲 Testing Random Right-Hand Side")
    print("=" * 40)
    
    # Load graph with trivial source (we'll replace RHS)
    loader = GraphLoader()
    graph_file = Path(__file__).parent.parent / "graphs" / "dorogovtsev_goltsev_mendes_1.txt"
    edges, vertices = loader.load_from_adjacency_matrix(graph_file)
    
    N = 9
    mfqg = MFQuantumGraph(N, vertices, edges)
    
    # Replace with random RHS
    np.random.seed(42)  # Reproducible
    random_rhs = np.random.randn(mfqg.AGG.shape[0])
    
    print(f"Matrix size: {mfqg.AGG.shape}")
    print(f"Random RHS norm: {np.linalg.norm(random_rhs):.6e}")
    
    # Test different preconditioners with random RHS
    preconditioners = {
        "Identity": None,
        "Degree": DegreePreconditioner()
    }
    
    for prec_name, preconditioner in preconditioners.items():
        print(f"\n🔧 {prec_name} Preconditioner:")
        
        if preconditioner is not None:
            preconditioner.compute(mfqg)
            M = preconditioner.as_linear_operator()
        else:
            M = None
        
        # Test CG
        start_time = time.time()
        solution, info = spla.cg(mfqg.AGG, random_rhs, M=M, rtol=1.49e-08, maxiter=1000)
        runtime = time.time() - start_time
        residual = np.linalg.norm(mfqg.AGG @ solution - random_rhs)
        
        print(f"   CG - Converged: {info == 0}, Info: {info}")
        print(f"   CG - Runtime: {runtime:.6f}s")
        print(f"   CG - Residual: {residual:.6e}")
        
        # Test BiCGSTAB
        start_time = time.time()
        solution, info = spla.bicgstab(mfqg.AGG, random_rhs, M=M, rtol=1.49e-08, maxiter=1000)
        runtime = time.time() - start_time
        residual = np.linalg.norm(mfqg.AGG @ solution - random_rhs)
        
        print(f"   BiCGSTAB - Converged: {info == 0}, Info: {info}")
        print(f"   BiCGSTAB - Runtime: {runtime:.6f}s")
        print(f"   BiCGSTAB - Residual: {residual:.6e}")


if __name__ == "__main__":
    # Test different source functions
    test_different_source_functions()
    
    # Test random RHS
    test_random_rhs()
    
    print("\n✅ Enhanced validation test complete!")