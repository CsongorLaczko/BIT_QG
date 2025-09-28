# Python Port Development Setup

This directory contains the Python port of the BIT_QG C++ codebase.

## Prerequisites

Install [uv](https://github.com/astral-sh/uv) for Python package management:

```bash
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Development Setup

```bash
# Navigate to Python port directory
cd python/

# Create virtual environment and install dependencies
uv sync --dev

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install package in development mode
uv pip install -e .
```

## Development Commands

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=bit_qg

# Run benchmarks only
uv run pytest tests/benchmarks/ --benchmark-only

# Lint and format code
uv run ruff check .
uv run ruff format .

# Type checking with ty (Astral's fast type checker)
uv run ty check bit_qg/ --ignore unknown-argument

# Type checking
uv run mypy bit_qg/

# Add new dependency
uv add numpy scipy torch

# Add development dependency
uv add --dev pytest ruff mypy
```

## Running the Python Code with C++ Interface

The Python port includes a `cpp_validation.py` script that provides **identical command-line interface** to the C++ `measure_nn.exe`:

### Usage
```bash
uv run python cpp_validation.py <graph> <size> <logN> [runs]
```

### Examples
```bash
# Run with Dorogovtsev-Goltsev-Mendes graph, size 1, discretization level 3, 1 run
uv run python cpp_validation.py dorogovtsev_goltsev_mendes 1 3 1

# Run with Barabási-Albert graph, size 4, discretization level 4 (default 1 run)
uv run python cpp_validation.py barabasi_albert 4 4

# Multiple runs for statistical averaging
uv run python cpp_validation.py dorogovtsev_goltsev_mendes 2 5 10
```

### Parameters
- **graph**: Graph name prefix (e.g., "dorogovtsev_goltsev_mendes", "barabasi_albert")
- **size**: Graph size identifier (1, 2, 3, 4)
- **logN**: Discretization parameter where N = 2^logN - 1 + 2 points per edge
- **runs**: Number of benchmark runs for statistical averaging (optional, default: 1)

### Input Files
The script reads graph adjacency matrices from `../graphs/{graph}_{size}.txt` files.

### Output Format
Produces output identical to C++ `measure_nn.exe`:
```
CG
Vanilla
assembly time: 7.419000e-04 runtime: 4.098000e-04 iterations: 1 error: 3.272772e-15

Degree
assembly time: 5.606000e-04 runtime: 3.323000e-04 iterations: 1 error: 3.272772e-15
...

BiCGSTAB
Vanilla
assembly time: 5.130000e-04 runtime: 4.375000e-04 iterations: 1 error: 6.665946e-15
...
```

Both CG and BiCGSTAB solvers are tested with all 5 preconditioners (Identity/Vanilla, Degree, Diagonal, Polynomial, Neumann-Neumann).

## Project Structure

```
python/
├── bit_qg/              # Main package
│   ├── core/           # Core quantum graph classes
│   ├── preconditioners/ # Custom solver preconditioners
│   ├── benchmarks/     # Performance measurement
│   └── utils/          # Graph I/O and utilities
├── tests/              # Test suite
│   ├── unit/          # Unit tests
│   ├── integration/   # Integration tests
│   └── benchmarks/    # Performance tests
└── pyproject.toml     # Project configuration
```