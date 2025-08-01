import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import os
import time
from cs336_basics.model import BasicsTransformerLM

def setup_process_group(rank, world_size, backend="gloo"):
    """Setup distributed process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Return device (CPU for gloo, GPU for nccl)
    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
    else:
        device = torch.device("cpu")
    
    return device

def cleanup_process_group():
    """Clean up distributed process group"""
    dist.destroy_process_group()

def create_xl_model():
    """Create XL model based on specifications from section 1.1.2"""
    return BasicsTransformerLM(
        vocab_size=10000,
        context_length=512,
        d_model=1600,
        num_layers=48, 
        num_heads=25,
        d_ff=6400,
        rope_theta=10000.0
    )

def naive_ddp_individual_params(model):
    """Naive DDP: All-reduce each parameter individually"""
    comm_start_time = time.time()
    
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()
    
    comm_end_time = time.time()
    return comm_end_time - comm_start_time

def flat_ddp_batched_params(model):
    """Optimized DDP: Flatten all gradients and do single all-reduce"""
    comm_start_time = time.time()
    
    # Collect all gradients
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.data.view(-1))
    
    if gradients:
        # Flatten all gradients into single tensor
        flat_gradients = torch.cat(gradients)
        
        # Single all-reduce call
        dist.all_reduce(flat_gradients, op=dist.ReduceOp.SUM)
        flat_gradients /= dist.get_world_size()
        
        # Unflatten and put gradients back
        offset = 0
        for param in model.parameters():
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.data = flat_gradients[offset:offset + numel].view(param.grad.shape)
                offset += numel
    
    comm_end_time = time.time()
    return comm_end_time - comm_start_time

def benchmark_ddp_training(rank, world_size, method="naive", num_steps=10):
    """Benchmark DDP training with specified method"""
    
    # Setup
    device = setup_process_group(rank, world_size, backend="gloo")
    dist.barrier()
    
    # Create XL model
    torch.manual_seed(42)  # Same initialization for all ranks
    model = create_xl_model().to(device)
    
    # Broadcast parameters from rank 0
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    # Setup optimizer and loss
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Dummy data (batch_size=4 as specified)
    batch_size = 4
    seq_length = 128  # Shorter sequence for faster benchmarking
    vocab_size = 10000
    
    total_step_time = 0
    total_comm_time = 0
    
    if rank == 0:
        print(f"Starting {method} DDP benchmarking with {num_steps} steps...")
    
    for step in range(num_steps):
        # Generate different data for each rank
        torch.manual_seed(step * world_size + rank)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
        
        step_start_time = time.time()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids)
        
        # Reshape for loss calculation
        loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient communication (the key difference)
        if method == "naive":
            comm_time = naive_ddp_individual_params(model)
        elif method == "flat":
            comm_time = flat_ddp_batched_params(model)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Optimizer step
        optimizer.step()
        
        step_end_time = time.time()
        
        step_time = step_end_time - step_start_time
        total_step_time += step_time
        total_comm_time += comm_time
        
        if rank == 0 and step % 5 == 0:
            print(f"Step {step}: Total time: {step_time:.3f}s, Comm time: {comm_time:.3f}s, "
                  f"Comm %: {comm_time/step_time*100:.1f}%")
    
    # Calculate averages
    avg_step_time = total_step_time / num_steps
    avg_comm_time = total_comm_time / num_steps
    comm_percentage = (avg_comm_time / avg_step_time) * 100
    
    if rank == 0:
        print(f"\n=== {method.upper()} DDP Results ===")
        print(f"Average time per training step: {avg_step_time:.3f}s")
        print(f"Average time communicating gradients: {avg_comm_time:.3f}s")
        print(f"Communication overhead: {comm_percentage:.1f}%")
        print("-" * 50)
    
    cleanup_process_group()
    
    return avg_step_time, avg_comm_time, comm_percentage

def run_comparison():
    """Run comparison between naive and flat DDP implementations"""
    world_size = 2
    num_steps = 10
    
    print("=== DDP BENCHMARKING COMPARISON ===")
    print("Model: XL Transformer (1600 d_model, 48 layers, 25 heads)")
    print(f"World size: {world_size}, Steps: {num_steps}")
    print("=" * 60)
    
    # Benchmark naive DDP
    print("\n1. NAIVE DDP (Individual parameter all-reduce)")
    mp.spawn(benchmark_ddp_training, args=(world_size, "naive", num_steps), 
             nprocs=world_size, join=True)
    
    # Benchmark flat DDP  
    print("\n2. FLAT DDP (Batched all-reduce)")
    mp.spawn(benchmark_ddp_training, args=(world_size, "flat", num_steps),
             nprocs=world_size, join=True)

if __name__ == "__main__":
    run_comparison()