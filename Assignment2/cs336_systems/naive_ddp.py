import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import os
import time
from typing import Iterator

# Model definitions
class ToyModel(nn.Module):
    """Simple toy model for testing DDP functionality"""
    def __init__(self):
        super().__init__()
        # Much larger model for GPU utilization
        self.layers = nn.Sequential(
            nn.Linear(2048, 4096),    # Much larger
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1000)     # Match output_size
        )
        # Test expects this specific parameter
        self.no_grad_fixed_param = nn.Parameter(torch.tensor([2.0, 2.0]), requires_grad=False)
    
    def forward(self, x):
        return self.layers(x)
class ToyModelWithTiedWeights(nn.Module):
    """Toy model with tied weights for testing"""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 10)
        self.linear1 = nn.Linear(10, 64)
        self.linear2 = nn.Linear(64, 5)
        
        # Tie weights between embedding and final layer
        self.output_layer = nn.Linear(10, 5)
        self.output_layer.weight = self.embedding.weight[:5, :]  # Share weights
        
        self.no_grad_fixed_param = nn.Parameter(torch.tensor([2.0, 2.0]), requires_grad=False)
    
    def forward(self, x):
        # Assume x is continuous input that we project to embedding space
        emb = self.linear1(x)
        out = self.linear2(emb)
        return out

# DDP Implementation Functions
def get_ddp_individual_parameters(model):
    """
    Wrapper function that prepares model for DDP training.
    This broadcasts rank 0's parameters to all other ranks.
    """
    # Broadcast parameters from rank 0 to all other ranks
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    return model

def ddp_individual_parameters_on_after_backward(model, optimizer):
    """
    Function to call after backward pass but before optimizer step.
    This manually all-reduces gradients for each parameter.
    """
    # All-reduce gradients for each parameter individually
    for param in model.parameters():
        if param.grad is not None:
            # Gloo doesn't support AVG, so use SUM and manually divide
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()  # Manual averaging

# Utility functions for process group management
def setup_process_group(rank, world_size, backend="nccl"):
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

def validate_ddp_net_equivalence(model):
    """Validate that all ranks have the same model parameters"""
    for param in model.parameters():
        # Gather parameters from all ranks
        gathered_params = [torch.zeros_like(param) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_params, param)
        
        # Check that all parameters are the same
        for i in range(1, len(gathered_params)):
            assert torch.allclose(gathered_params[0], gathered_params[i]), \
                f"Parameters not synchronized across ranks"

# Main training function
def naive_ddp_training(rank, world_size, num_epochs=5):
    """Main training function for naive DDP"""
    
    # Setup process group
    device = setup_process_group(rank, world_size, backend="nccl")
    
    # Barrier to ensure all processes are ready
    dist.barrier(device_ids=[rank])

    # Set the device BEFORE barrier (this fixes the warning)
    torch.cuda.set_device(rank)
    
    # Set different seeds for different ranks initially
    torch.manual_seed(rank)
    
    # Create model
    model = ToyModel().to(device)
    
    # Apply DDP setup (broadcast rank 0's parameters)
    model = get_ddp_individual_parameters(model)
    
    # Validate that all ranks have the same model
    validate_ddp_net_equivalence(model)
    
    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    
    # Create some dummy data for training
    # Each rank gets different data (simulating data parallelism)
    batch_size = 64
    input_size = 2048
    output_size = 1000
    
    print(f"Rank {rank}: Starting training")
    
    for epoch in range(num_epochs):
        # Generate different data for each rank
        torch.manual_seed(epoch * world_size + rank)  # Different data per rank
        data = torch.randn(batch_size, input_size).to(device)
        targets = torch.randn(batch_size, output_size).to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # DDP gradient synchronization (this is the key part!)
        ddp_individual_parameters_on_after_backward(model, optimizer)
        
        # Optimizer step
        optimizer.step()
        
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Final validation that all ranks have the same parameters
    validate_ddp_net_equivalence(model)
    
    if rank == 0:
        print("Training completed successfully!")
    
    # Cleanup
    cleanup_process_group()

# Verification function to compare with single-process training
def single_process_training(num_epochs=5):
    """Single process training for comparison"""
    torch.manual_seed(0)  # Same seed as rank 0
    
    model = ToyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    
    batch_size = 64
    input_size = 2048
    output_size = 1000
    
    for epoch in range(num_epochs):
        # Generate data (same pattern as distributed training)
        total_loss = 0
        
        # Simulate processing all ranks' data
        for rank in range(2):  # Assuming world_size = 2
            torch.manual_seed(epoch * 2 + rank)
            data = torch.randn(batch_size, input_size)
            targets = torch.randn(batch_size, output_size)
            
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            total_loss += loss
        
        # Average loss (simulating DDP averaging)
        avg_loss = total_loss / 2
        
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        
        print(f"Single process - Epoch {epoch}, Loss: {avg_loss.item():.4f}")
    
    return model

if __name__ == "__main__":
    print("Testing Naive DDP Implementation")
    
    # Test with distributed training
    world_size = 2
    print(f"\n=== Running Distributed Training (world_size={world_size}) ===")
    mp.spawn(naive_ddp_training, args=(world_size,), nprocs=world_size, join=True)
    
    # Test with single process for comparison
    print(f"\n=== Running Single Process Training for Comparison ===")
    single_model = single_process_training()
    
    print("\nNaive DDP implementation completed!")