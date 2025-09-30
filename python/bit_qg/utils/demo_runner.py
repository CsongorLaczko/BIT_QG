#!/usr/bin/env python3
"""
Demo runner for BIT_QG preconditioners.

This module provides demonstration functionality for different preconditioners.
"""

from pathlib import Path


def run_demo(preconditioner_type: str = "all"):
    """
    Run demonstration scripts for preconditioners.
    
    Args:
        preconditioner_type: Type of preconditioner to demo ("degree", "diagonal", 
                           "polynomial", "neumann_neumann", or "all")
    """
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    
    if preconditioner_type == "all":
        print("🎯 Running all preconditioner demonstrations...")
        print("See examples/ directory for individual demo scripts")
        print(f"Examples directory: {examples_dir}")
        
        # List available examples
        demo_files = list(examples_dir.glob("*.py"))
        if demo_files:
            print("\nAvailable demonstrations:")
            for demo_file in demo_files:
                print(f"  - {demo_file.name}")
        else:
            print("No demonstration files found in examples/ directory")
    else:
        print(f"🎯 Running {preconditioner_type} preconditioner demonstration...")
        demo_file = examples_dir / f"{preconditioner_type}_demo.py"
        
        if demo_file.exists():
            print(f"Running: {demo_file}")
            import subprocess
            import sys
            result = subprocess.run([sys.executable, str(demo_file)])
            if result.returncode != 0:
                print(f"Demo failed with return code: {result.returncode}")
        else:
            print(f"Demo file not found: {demo_file}")
            print("Available preconditioner types: degree, diagonal, polynomial, neumann_neumann, all")