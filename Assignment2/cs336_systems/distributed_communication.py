import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost" #Internal IP address.
    os.environ["MASTER_PORT"] = "20007"
    dist.init_process_group("nccl", rank=rank, world_size=world_size) #Q: when using gloo, super fast but run on CPU, when using nccl, sloer but on GPU.
    torch.cuda.set_device(rank)

def benchmark_allreduce(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    #tensor = torch.ones(500_000_000, device=device) * rank  # ~2 GB per GPU, create tensor on GPU.
    tensor= torch.randint(0,10,(3,), device=device)
    torch.cuda.synchronize()

    print("0")
    start = time.time()
    dist.all_reduce(tensor, async_op=False)
    end = time.time()
    torch.cuda.synchronize()
    print(f"[Rank {rank}] All-reduce on GPU took {end - start:.3f} seconds")


    # PyTorch recommended cleanup sequence:
    print("a")
    torch.cuda.synchronize()        # 1. Wait for GPU operations-not all reduce but maybe sth like cleanup.
    print("b")
    dist.barrier()                  # 2. Let all CPU wait for each other.  
    print("c")
    dist.destroy_process_group()    # 3. Clean NCCL for all CPU processes.

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(benchmark_allreduce, args=(world_size,), nprocs=world_size,join=True)
