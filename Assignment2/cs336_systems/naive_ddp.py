# naive_ddp_fixed_port9776.py
import os, torch, torch.distributed as dist, torch.multiprocessing as mp
from torch import nn

# -------------------------------- tiny model
class ToyModel(nn.Module):
    def __init__(self, in_dim=8, hidden=16, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

# -------------------------------- PG setup / teardown
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9776"        # <- fixed port
    dist.init_process_group(                  # 2️⃣ then create PG
        backend="nccl",
        rank=rank,
        world_size=world_size)
    torch.cuda.set_device(rank)


# -------------------------------- training loop
def train(rank, world_size, steps=50, batch=32, lr=1e-2):
    setup(rank, world_size)
    print(rank)
    device = torch.device(f"cuda:{rank}")
    model = ToyModel().cuda(rank)
    opt   = torch.optim.SGD(model.parameters(), lr=lr)
    lossf = nn.MSELoss()

    for step in range(steps):
        torch.manual_seed(step)                           # same fake data
        x = torch.randn(batch, 8,  device=f"cuda:{rank}")
        y = torch.randn(batch, 4,  device=f"cuda:{rank}")

        opt.zero_grad(set_to_none=True)
        lossf(model(x), y).backward()

        # naive DDP: average grads
        for p in model.parameters():
            if p.grad is None: continue
            dist.all_reduce(p.grad, async_op=False)
            p.grad /= world_size

        opt.step()
        if (step + 1) % 10 == 0:
            print(f"step {step+1:3d}  loss {lossf(model(x), y).item():.4f}")
    

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
