#!/usr/bin/env python3
"""
Large Problem Validation Test
=============================

Test with larger graphs and higher discretization to see if we can get
meaningful iteration counts.
"""

import math
import time
import numpy as np
from pathlib import Path

# Add current directory to path to import our modules
import sys
sys.path.append(str(Path(__file__).parent))

def test_large_problems():
    """Test with larger graphs and higher discretization."""
    from bit_qg.core import MFQuantumGraph
    from bit_qg.utils import GraphLoader
    from bit_qg.preconditioners import DegreePreconditioner, NeumannNeumannPreconditioner
    import scipy.sparse.linalg as spla
    
    print("🧪 Testing Large Problems for Meaningful Iteration Counts")
    print("=" * 60)
    
    # Test different graph sizes and discretizations
    test_configs = [
        ("dorogovtsev_goltsev_mendes_1.txt", 3, "Small graph, small discretization"),
        ("dorogovtsev_goltsev_mendes_1.txt", 5, "Small graph, large discretization"),  
        ("dorogovtsev_goltsev_mendes_4.txt", 3, "Large graph, small discretization"),
        ("dorogovtsev_goltsev_mendes_4.txt", 5, "Large graph, large discretization"),
    ]
    
    # Different source functions
    source_functions = {
        "moderate": lambda x: 1.0 * math.exp(-((x - 0.5)**2) * 5),        # Broader peak
        "constant": lambda x: 1.0,  # Constant source
    }
    
    for graph_file, log_n, description in test_configs:
        print(f"\n📊 {description}")
        print(f"   Graph: {graph_file}, logN: {log_n}")
        
        N = 2**log_n - 1 + 2
        print(f"   Discretization: N = {N} points per edge")
        
        for func_name, f_func in source_functions.items():
            print(f"\n   🔧 Source function: {func_name}")
            
            try:
                # Load graph
                loader = GraphLoader()
                graph_path = Path(__file__).parent.parent / "graphs" / graph_file
                edges, vertices = loader.load_from_adjacency_matrix(
                    graph_path,
                    c_func=lambda x: 1.0 / (1.0 + math.exp(-25 * (x - 0.5))) + 1.0,
                    v_func=lambda x: 0.05 / (0.2**2) * (abs(x - 0.5) - 0.2)**2 + 0.05,
                    f_func=f_func
                )
                
                # Create quantum graph
                mfqg = MFQuantumGraph(N, vertices, edges)
                
                print(f"      Matrix size: {mfqg.AGG.shape}")
                print(f"      RHS norm: {np.linalg.norm(mfqg.bG):.6e}")
                
                # Test different preconditioners
                preconditioners = [
                    ("Identity", None),
                    ("Degree", DegreePreconditioner()),
                    ("Neumann-Neumann", NeumannNeumannPreconditioner())
                ]
                
                for prec_name, preconditioner in preconditioners:
                    print(f"      📈 {prec_name}:")
                    
                    if preconditioner is not None:
                        preconditioner.compute(mfqg)
                        M = preconditioner.as_linear_operator()
                    else:
                        M = None
                    
                    # Test CG with more iterations allowed
                    start_time = time.time()
                    solution, info = spla.cg(
                        mfqg.AGG, mfqg.bG, M=M, 
                        rtol=1.49e-08, maxiter=1000, atol=0.0
                    )
                    runtime = time.time() - start_time
                    residual = np.linalg.norm(mfqg.AGG @ solution - mfqg.bG)
                    
                    print(f"         CG: info={info}, runtime={runtime:.6f}s, residual={residual:.2e}")
                    
                    # Test BiCGSTAB  
                    start_time = time.time()
                    solution, info = spla.bicgstab(
                        mfqg.AGG, mfqg.bG, M=M,
                        rtol=1.49e-08, maxiter=1000, atol=0.0
                    )
                    runtime = time.time() - start_time
                    residual = np.linalg.norm(mfqg.AGG @ solution - mfqg.bG)
                    
                    print(f"         BiCGSTAB: info={info}, runtime={runtime:.6f}s, residual={residual:.2e}")
            
            except Exception as e:
                print(f"      ❌ Error: {e}")


def test_condition_number():
    """Check matrix condition numbers to understand why solvers converge so fast."""
    from bit_qg.core import MFQuantumGraph
    from bit_qg.utils import GraphLoader
    import scipy.sparse.linalg as spla
    
    print("\n🔍 Matrix Condition Number Analysis")
    print("=" * 40)
    
    # Test different problem sizes
    configs = [
        ("dorogovtsev_goltsev_mendes_1.txt", 3),
        ("dorogovtsev_goltsev_mendes_1.txt", 4), 
        ("dorogovtsev_goltsev_mendes_4.txt", 3),
        ("dorogovtsev_goltsev_mendes_4.txt", 4),
    ]
    
    for graph_file, log_n in configs:
        print(f"\n📊 {graph_file}, logN={log_n}")
        
        try:
            # Load graph with constant source
            loader = GraphLoader()
            graph_path = Path(__file__).parent.parent / "graphs" / graph_file
            edges, vertices = loader.load_from_adjacency_matrix(graph_path)
            
            N = 2**log_n - 1 + 2
            mfqg = MFQuantumGraph(N, vertices, edges)
            
            # Convert to dense for condition number calculation
            A_dense = mfqg.AGG.toarray()
            
            # Calculate condition number
            cond_num = np.linalg.cond(A_dense)
            
            # Calculate eigenvalues for more insight
            eigenvals = np.linalg.eigvals(A_dense)
            min_eig = np.min(eigenvals)
            max_eig = np.max(eigenvals)
            
            print(f"   Matrix size: {A_dense.shape}")
            print(f"   Condition number: {cond_num:.2e}")
            print(f"   Min eigenvalue: {min_eig:.6e}")
            print(f"   Max eigenvalue: {max_eig:.6e}")
            print(f"   Eigenvalue ratio: {max_eig/max(min_eig, 1e-16):.2e}")
            
            if cond_num < 100:
                print("   ✅ Well-conditioned matrix - explains fast convergence")
            elif cond_num < 10000:
                print("   ⚠️  Moderately conditioned matrix")
            else:
                print("   ❌ Poorly conditioned matrix")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    test_large_problems()
    test_condition_number()