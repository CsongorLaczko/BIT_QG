# Copilot Instructions for BIT_QG

## Project Overview
BIT_QG is a scientific computing project focused on quantum graphs and numerical methods. Currently being ported from C++ (Eigen-based) to Python (SciPy/NumPy) while maintaining mathematical correctness and adding PyTorch neural network capabilities.

## Current State: C++ to Python Port
**Active Branch**: `python-port` - Converting Eigen-based C++ implementation to Python

### C++ Codebase (Legacy - Being Ported)
- **Core Algorithm**: `MFQuantumGraph` class in `include/mf_quantum_graph.h` implements finite element methods on quantum graphs
- **Solvers**: Uses Eigen's BiCGSTAB and Conjugate Gradient iterative solvers with custom preconditioners
- **Preconditioners**: `degree_preconditioner.h`, `polynomial_preconditioner.h`, `neumann_neumann_preconditioner.h`
- **Main Executables**: `measure_nn.cpp` (performance benchmarking), `mf_quantum_graph.cpp` (core implementation)
- **Dependencies**: Pure Eigen (no libtorch in C++ code despite `libtorch/` directory presence)

### Python Port Architecture (Target)
- **Core Data Structures**: QGEdge (quantum graph edges with function pointers → callable functions)
- **Linear Algebra**: SciPy sparse matrices, BiCGSTAB/GMRES solvers, NumPy arrays
- **Preconditioners**: Custom preconditioner classes compatible with SciPy linear solvers
- **Future Integration**: PyTorch tensors for neural network extensions (GPU-ready)

## Critical Porting Patterns & Mappings

### C++ → Python Equivalents
- **`Eigen::SparseMatrix<double>`** → `scipy.sparse.csr_matrix` or `csc_matrix`
- **`Eigen::VectorXd`** → `numpy.ndarray` (1D)
- **`Eigen::BiCGSTAB`** → `scipy.sparse.linalg.bicgstab`
- **`Eigen::ConjugateGradient`** → `scipy.sparse.linalg.cg`
- **Function Pointers** (QGEdge c,v,f) → Python callable functions
- **Template Preconditioners** → Python classes with `solve()` method

### Key Mathematical Concepts
- **Finite Element Assembly**: Edge-based discretization with interior/boundary separation (AII, AIG, AGG matrices)
- **Schur Complement**: `AGG*rhs - AIG.T @ solver.solve(AIG @ rhs)` pattern in solve() method
- **Custom Preconditioners**: Degree-based, polynomial, and domain decomposition methods

## Development Workflows

### C++ Build (Legacy)
```cmd
cmake -S . -B build
cmake --build build
.\build\measure_nn.exe
```

### Python Port (Target)
```bash
pip install -r requirements.txt
python -m pytest tests/
python -m bit_qg.benchmarks.measure_nn
```

## Testing & Validation Strategy
- **Numerical Accuracy**: Compare Python results with C++ reference implementations using `numpy.allclose()`
- **Performance Benchmarks**: Port `measure_nn.cpp` timing functionality to Python
- **Test Organization**: 
  - `tests/unit/` - Individual component tests
  - `tests/integration/` - Full algorithm validation against C++ 
  - `tests/benchmarks/` - Performance comparison suite

## Port Progress & Implementation Order
1. **Core Data Structures**: QGEdge class with callable functions
2. **Matrix Assembly**: Finite element discretization logic
3. **Preconditioners**: Custom solver preconditioners
4. **Benchmarking**: Performance measurement utilities
5. **Graph Generation**: Enhanced Python graph utilities
6. **Neural Network Integration**: PyTorch tensor compatibility

## Critical Implementation Notes
- **Eigen EigenBase Pattern**: C++ uses custom matrix-free operators; port to scipy.sparse.linalg.LinearOperator
- **Template Specialization**: C++ preconditioner templates → Python ABC with concrete implementations
- **Memory Layout**: Eigen column-major → ensure NumPy C/F order compatibility
- **Solver Tolerance**: Maintain same convergence criteria across C++/Python versions

## Conventions & Patterns
- **Graph Files**: Input/output graph data is stored in `graphs/*.txt`.
- **Model Export**: Exported models are saved in `pinn/exported_models/`.
- **Headers**: All C++ headers are in `include/`.
- **Custom Preconditioners**: Extend preconditioner classes in `include/` for new numerical methods.
- **Scripts**: Python scripts are used for automation and data generation.
- **Port Package Structure**: 
  ```
  bit_qg/
    core/          # QGEdge, MFQuantumGraph classes
    preconditioners/ # Custom solver preconditioners  
    benchmarks/    # Performance measurement utilities
    utils/         # Graph generation, I/O utilities
  ```

## Integration Points
- **libtorch**: C++ code links against libraries in `libtorch/`.
- **Python-C++ Interop**: Data exchange via file I/O (e.g., graph files, exported models).

## Examples
- To generate a graph: `python graphs/generate.py`
- To build C++ code: `cmake -S . -B build && cmake --build build`
- To run a PINN example: Check `pinn/examples/` for scripts

## Key Files & Directories
- `src/`, `include/`: C++ source and headers
- `graphs/`: Graph data and generation scripts
- `pinn/`: PINN models, scripts, and exported models
- `libtorch/`: External library dependencies
- `build/`: C++ build artifacts

---
_If any section is unclear or missing important details, please provide feedback to improve these instructions._
