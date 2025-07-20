#!/usr/bin/env python3
"""
Transformer model benchmarking script for forward and backward pass timing.
"""

import argparse
import timeit
import torch
import torch.nn.functional as F
import numpy as np
from cs336_basics.model import BasicsTransformerLM


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def generate_random_batch(batch_size, context_length, vocab_size, device):
    """Generate random input and target tensors."""
    input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return input_ids, targets


def benchmark_model(model, input_ids, targets, warmup_steps, measurement_steps, forward_only=False):
    """
    Benchmark the model with proper timing and synchronization.
    
    Returns:
        forward_times: list of forward pass times
        backward_times: list of backward pass times (empty if forward_only=True)
    """
    device = next(model.parameters()).device
    
    # Warmup phase
    print(f"Running {warmup_steps} warmup steps...")
    for _ in range(warmup_steps):
        model.zero_grad()
        output = model(input_ids)
        if not forward_only:
            loss = F.cross_entropy(output.view(-1, output.size(-1)), targets.view(-1))
            loss.backward()
        
        # Synchronize after each warmup step
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
    
    print(f"Running {measurement_steps} measurement steps...")
    
    forward_times = []
    backward_times = []
    
    for step in range(measurement_steps):
        model.zero_grad()
        
        # Time forward pass
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
            
        start_time = timeit.default_timer()
        output = model(input_ids)
        
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
            
        forward_time = timeit.default_timer() - start_time
        forward_times.append(forward_time)
        
        # Time backward pass (if not forward_only)
        if not forward_only:
            loss = F.cross_entropy(output.view(-1, output.size(-1)), targets.view(-1))
            
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
                
            start_time = timeit.default_timer()
            loss.backward()
            
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
                
            backward_time = timeit.default_timer() - start_time
            backward_times.append(backward_time)
        
        print(f"Step {step + 1}/{measurement_steps} - Forward: {forward_time:.6f}s" + 
              (f", Backward: {backward_times[-1]:.6f}s" if not forward_only else ""))
    
    return forward_times, backward_times


def print_statistics(forward_times, backward_times, forward_only):
    """Print timing statistics."""
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    # Forward pass statistics
    forward_mean = np.mean(forward_times)
    forward_std = np.std(forward_times)
    print(f"Forward Pass:")
    print(f"  Mean time: {forward_mean:.6f}s")
    print(f"  Std dev:   {forward_std:.6f}s")
    print(f"  Min time:  {np.min(forward_times):.6f}s")
    print(f"  Max time:  {np.max(forward_times):.6f}s")
    
    if not forward_only and backward_times:
        # Backward pass statistics
        backward_mean = np.mean(backward_times)
        backward_std = np.std(backward_times)
        print(f"\nBackward Pass:")
        print(f"  Mean time: {backward_mean:.6f}s")
        print(f"  Std dev:   {backward_std:.6f}s")
        print(f"  Min time:  {np.min(backward_times):.6f}s")
        print(f"  Max time:  {np.max(backward_times):.6f}s")
        
        # Total time per step
        total_times = np.array(forward_times) + np.array(backward_times)
        total_mean = np.mean(total_times)
        total_std = np.std(total_times)
        print(f"\nTotal (Forward + Backward):")
        print(f"  Mean time: {total_mean:.6f}s")
        print(f"  Std dev:   {total_std:.6f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Transformer model forward and backward passes")
    
    # Model architecture arguments
    parser.add_argument("--d_model", type=int, default=512, help="Model embedding dimension")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward layer dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--context_length", type=int, default=128, help="Sequence length")
    
    # Timing arguments
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--measurement_steps", type=int, default=10, help="Number of measurement steps")
    parser.add_argument("--forward_only", action="store_true", help="Only time forward pass")
    
    # Fixed parameters (as specified)
    vocab_size = 10000
    batch_size = 4
    rope_theta = 10000.0
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.d_model % args.num_heads != 0:
        raise ValueError(f"d_model ({args.d_model}) must be divisible by num_heads ({args.num_heads})")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Setup device
    device=torch.device("cuda")
    """"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Metal) device")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU device")
    """
    # Print configuration
    print(f"\nModel Configuration:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  context_length: {args.context_length}")
    print(f"  d_model: {args.d_model}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  d_ff: {args.d_ff}")
    print(f"  rope_theta: {rope_theta}")
    print(f"  batch_size: {batch_size}")
    print(f"  forward_only: {args.forward_only}")
    
    # Initialize model
    print(f"\nInitializing model...")
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=rope_theta
    )
    
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Generate random data
    print(f"\nGenerating random batch...")
    input_ids, targets = generate_random_batch(batch_size, args.context_length, vocab_size, device)
    
    # Run benchmark
    forward_times, backward_times = benchmark_model(
        model=model,
        input_ids=input_ids,
        targets=targets,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        forward_only=args.forward_only
    )
    
    # Print results
    #print_statistics(forward_times, backward_times, args.forward_only)


if __name__ == "__main__":
    main()