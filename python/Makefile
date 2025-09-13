# Makefile for BIT_QG Python Port
# Uses uv for package management and ruff for linting/formatting

.PHONY: help install install-dev test test-fast lint format type-check clean build

help:
	@echo "Available commands:"
	@echo "  install       Install package"
	@echo "  install-dev   Install package in development mode with all dependencies"
	@echo "  test          Run all tests"
	@echo "  test-fast     Run tests without benchmarks"
	@echo "  lint          Run linting (ruff check)"
	@echo "  format        Format code (ruff format)"
	@echo "  type-check    Run type checking (mypy)"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build package"

install:
	uv sync

install-dev:
	uv sync --dev

test:
	uv run pytest

test-fast:
	uv run pytest -x --ignore=tests/benchmarks/

lint:
	uv run ruff check .

format:
	uv run ruff format .

type-check:
	uv run mypy bit_qg/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete

build:
	uv build