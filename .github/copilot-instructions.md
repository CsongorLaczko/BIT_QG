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

### Python Port Structure (Current)
```
python/                    # ISOLATED Python port directory
├── bit_qg/                # Main Python package
│   ├── core/             # ✅ COMPLETE: QGEdge, MFQuantumGraph classes (100% tested)
│   ├── preconditioners/ # ✅ COMPLETE: All 3 preconditioners implemented (97% tested)
│   ├── benchmarks/       # ⏳ TODO: Performance measurement utilities
│   └── utils/            # ⏳ TODO: Graph generation, I/O utilities
├── tests/                # ✅ COMPLETE: Comprehensive unit & integration tests (36/36 passing, 97% coverage)
├── pyproject.toml        # ✅ COMPLETE: Modern uv + ruff configuration
└── README.md             # ✅ COMPLETE: Development setup guide
```

### Port Progress Status (Updated: September 13, 2025)
- ✅ **Core Data Structures**: QGEdge (with callable functions), MFQuantumGraph (full finite element implementation) - **100% test coverage**
- ✅ **Development Environment**: Modern Python tooling (uv, ruff), clean pyproject.toml, no linting issues
- ✅ **Code Quality**: All tests passing (36/36), 97% code coverage, SparseEfficiencyWarnings resolved
- ✅ **All Preconditioners**: DegreePreconditioner, PolynomialPreconditioner & **NeumannNeumannPreconditioner** with SciPy LinearOperator compatibility - **97% combined test coverage**
- ✅ **Integration Tests**: Basic validation tests for mathematical properties and correctness
- ❌ **Graph Utilities**: Need to enhance existing graph generation scripts
- ❌ **Benchmarking**: Need to port `measure_nn.cpp` timing functionality
- ❌ **C++ Integration Tests**: Need direct validation against C++ reference implementations

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
- **Preconditioner Interface**: All preconditioners inherit from PreconditionerBase and support SciPy LinearOperator
- **Degree Preconditioner**: Simple diagonal preconditioner using inverse vertex weights
- **Polynomial Preconditioner**: Advanced preconditioner using Schur complement diagonal entries
- **Neumann-Neumann Preconditioner**: Domain decomposition method with edge-based local solvers and vertex weight averaging

## Development Workflows

### C++ Build (Legacy)
```cmd
cmake -S . -B build
cmake --build build
.\build\measure_nn.exe
```

### Python Port Development (Current)
```bash
cd python/                    # Work in isolated Python directory
uv sync --dev                 # Install dependencies with uv
uv run pytest                 # Run tests (36/36 passing, 97% coverage)
uv run ruff check .           # Lint with ruff (all checks pass)
uv run ruff format .          # Format with ruff
```

**CRITICAL**: Always work in `python/` directory, never mix with C++ root directory structure!

## Testing & Validation Strategy
- **Numerical Accuracy**: Compare Python results with C++ reference implementations using `numpy.allclose()`
- **Performance Benchmarks**: Port `measure_nn.cpp` timing functionality to Python
- **Test Coverage**: Currently achieving 100% test coverage on implemented components
- **Test Organization**: 
  - `tests/unit/` - Individual component tests (✅ Complete for core components)
  - `tests/integration/` - Full algorithm validation against C++ (✅ Basic validation tests, ⏳ C++ comparison pending)
  - `tests/benchmarks/` - Performance comparison suite (⏳ Pending)
- **Quality Assurance**: All code passes ruff linting, pytest runs clean without warnings

## Port Progress & Implementation Order
1. **Core Data Structures**: QGEdge class with callable functions ✅
2. **Matrix Assembly**: Finite element discretization logic ✅
3. **Preconditioners**: Custom solver preconditioners ✅ **ALL THREE COMPLETE**
   - DegreePreconditioner: Simple diagonal inverse vertex weight scaling ✅
   - PolynomialPreconditioner: Advanced Schur complement diagonal preconditioning ✅  
   - NeumannNeumannPreconditioner: Domain decomposition with local edge solvers ✅
4. **Benchmarking**: Performance measurement utilities ⏳
5. **Graph Generation**: Enhanced Python graph utilities ⏳
6. **Neural Network Integration**: PyTorch tensor compatibility ⏳

## Critical Implementation Notes
- **Eigen EigenBase Pattern**: C++ uses custom matrix-free operators; port to scipy.sparse.linalg.LinearOperator
- **Template Specialization**: C++ preconditioner templates → Python ABC with concrete implementations
- **Memory Layout**: Eigen column-major → ensure NumPy C/F order compatibility
- **Solver Tolerance**: Maintain same convergence criteria across C++/Python versions
- **Singular Systems**: Neumann-Neumann preconditioner handles inherently singular local problems with robust fallbacks

## Current Status & Next Priorities (September 13, 2025)

### ✅ COMPLETED: Core Finite Element Framework
All essential mathematical components have been successfully ported from C++ to Python:

- **Complete Preconditioner Suite**: All three preconditioners (Degree, Polynomial, Neumann-Neumann) fully implemented
- **Robust Error Handling**: Singular system detection with LSQR fallbacks for domain decomposition
- **SciPy Integration**: Full LinearOperator compatibility for iterative solvers (BiCGSTAB, CG)
- **Mathematical Correctness**: Exact C++ algorithm replication with 97% test coverage
- **Quality Assurance**: Zero linting errors, 36/36 tests passing, comprehensive validation

### ⏳ NEXT PRIORITIES: Performance & Integration
The core mathematical framework is complete. Focus should now shift to:

1. **Performance Benchmarking**: Port `measure_nn.cpp` to compare Python vs C++ solver performance
2. **Graph Utilities Enhancement**: Expand graph generation capabilities beyond basic examples  
3. **C++ Validation Suite**: Direct numerical comparison tests against C++ reference implementation
4. **PyTorch Integration**: Tensor compatibility for neural network-quantum graph hybrid methods

## Conventions & Patterns
- **Graph Files**: Input/output graph data is stored in `graphs/*.txt`.
- **Model Export**: Exported models are saved in `pinn/exported_models/`.
- **Headers**: All C++ headers are in `include/`.
- **Custom Preconditioners**: Extend preconditioner classes in `include/` for new numerical methods.
- **Scripts**: Python scripts are used for automation and data generation.
- **Port Package Structure**: 
  ```
  python/bit_qg/          # CLEAN structure - no src/ subdirectory!
    core/                 # QGEdge, MFQuantumGraph classes
    preconditioners/      # Custom solver preconditioners  
    benchmarks/           # Performance measurement utilities
    utils/                # Graph generation, I/O utilities
  ```
- **NO DUPLICATE STRUCTURES**: Keep Python port completely isolated in `python/` directory

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
