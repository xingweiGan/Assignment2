import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "9776"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def benchmark_allreduce(rank, world_size):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    tensor = torch.ones(500_000_000, device=device) * rank  # ~2 GB per GPU

    dist.barrier()  # synchronize
    start = time.time()
    dist.all_reduce(tensor)
    dist.barrier()
    end = time.time()

    print(f"[Rank {rank}] All-reduce on GPU took {end - start:.3f} seconds")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(benchmark_allreduce, args=(world_size,), nprocs=world_size)
