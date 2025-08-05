import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import os
import time
from cs336_basics.model import BasicsTransformerLM

# -------------------------------- PG setup / teardown
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost" #Internal IP address.
    os.environ["MASTER_PORT"] = "10018"
    dist.init_process_group("nccl", rank=rank, world_size=world_size) #Q: when using gloo, super fast but run on CPU, when using nccl, sloer but on GPU.
    torch.cuda.set_device(rank)


# -------------------------------- training loop
def train(rank, world_size, steps=50, batch=8, lr=1e-2):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    model = BasicsTransformerLM(
        vocab_size=1000,
        context_length=128,
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        rope_theta=10000.0
    ).cuda(rank)
    
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    for step in range(steps):
        torch.manual_seed(step)
        seq_len = 64
        x = torch.randint(0, 1000, (batch, seq_len), device=device)
        y = torch.randint(0, 1000, (batch, seq_len), device=device)
        
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = lossf(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        # Naive DDP: average gradients
        for p in model.parameters():
            if p.grad is None: continue
            dist.all_reduce(p.grad, async_op=False)
            p.grad /= world_size

        opt.step()
        
        if (step + 1) % 10 == 0:
            print(f"Rank {rank}, step {step+1:3d}, loss {loss.item():.4f}")    

    # PyTorch recommended cleanup sequence:
    print("a")
    torch.cuda.synchronize()        # 1. Wait for GPU operations-not all reduce but maybe sth like cleanup.
    print("b")
    dist.barrier()                  # 2. Let all CPU wait for each other.  
    print("c")
    dist.destroy_process_group()    # 3. Clean NCCL for all CPU processes.




# -------------------------------- entry point
if __name__ == "__main__":
    world_size = torch.cuda.device_count()    # e.g. 4
    mp.spawn(train, args=(world_size,), nprocs=world_size)
