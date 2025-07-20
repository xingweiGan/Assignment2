#!/usr/bin/env python3
"""
Batch benchmarking script for different model sizes.
"""

import subprocess
import sys

# Model configurations from the table
models = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32}
}

def run_benchmark(model_name, config):
    """Run benchmark for a specific model configuration."""
    print(f"{model_name}")
    
    cmd = [
        "python", "-m", "cs336_systems.benchmarking",
        "--d_model", str(config["d_model"]),
        "--d_ff", str(config["d_ff"]),
        "--num_layers", str(config["num_layers"]),
        "--num_heads", str(config["num_heads"]),
        "--warmup_steps", "0",
        "--measurement_steps", "10"
    ]
    
    subprocess.run(cmd)

def main():
    for model_name, config in models.items():
        run_benchmark(model_name, config)
        print()  # Empty line between models

if __name__ == "__main__":
    main()