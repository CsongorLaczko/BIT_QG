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