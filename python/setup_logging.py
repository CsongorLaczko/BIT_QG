#!/usr/bin/env python3
"""
Setup logging configuration for BIT_QG debugging.

This script configures logging to show fallback and error messages from the
quantum graph library. Run this before your main script to enable logging.

Usage:
    import setup_logging  # Just import to configure logging

Or run directly:
    python setup_logging.py
"""

import logging
import sys


def setup_logging(level=logging.WARNING):
    """
    Configure logging for BIT_QG library debugging.

    Args:
        level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING, etc.)
    """
    # Create a formatter that shows the module name and message
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    # Configure BIT_QG specific loggers
    for module in [
        "bit_qg.preconditioners.neumann_neumann",
        "bit_qg.utils.graph_io",
        "bit_qg.benchmarks.benchmarking",
    ]:
        logger = logging.getLogger(module)
        logger.setLevel(level)

    print(f"Logging configured at level: {logging.getLevelName(level)}")
    print("FALLBACK and ERROR messages will now be visible.")


if __name__ == "__main__":
    # Default to WARNING level to see fallbacks and errors
    setup_logging(logging.WARNING)
    print("Logging is now configured. Import this module in your scripts.")
    print("Example:")
    print("  import setup_logging")
    print("  # Now run your quantum graph code")
