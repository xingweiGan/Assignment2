import torch
import torch.distributed as dist
from typing import List

class DDPIndividualParameters(torch.nn.Module):
    """Light‑weight DDP wrapper that overlaps gradient communication
    with back‑prop by launching an async `all_reduce` **per parameter** as
    soon as its gradient is fully accumulated on the local GPU.

    Assumes `dist.init_process_group` has been called **before** the class
    is instantiated and that every process is mapped to one GPU.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._comm_handles: List[dist.Work] = []

        # 1. Ensure all ranks start with identical weights -----------------
        if dist.is_initialized() and dist.get_world_size() > 1:
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)

        # 2. Register a *post‑accumulate* hook per leaf parameter ----------
        if dist.is_initialized() and dist.get_world_size() > 1:
            for p in self.module.parameters():
                if p.requires_grad:
                    p.register_post_accumulate_grad_hook(self._make_hook())

    # ---------------------------------------------------------------------
    def _make_hook(self):
        """Return hook that queues async all‑reduce and stores handle."""
        def hook(param: torch.nn.Parameter):
            handle = dist.all_reduce(param.grad, async_op=True)
            self._comm_handles.append(handle)
        return hook

    # ---------------------------------------------------------------------
    def forward(self, *inputs, **kwargs):  # noqa: ANN001
        return self.module(*inputs, **kwargs)

    # ---------------------------------------------------------------------
    def finish_gradient_synchronization(self):
        """Wait for outstanding all‑reduces and average grads."""
        if not self._comm_handles:
            return
        for h in self._comm_handles:
            h.wait()
        self._comm_handles.clear()

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            scale = 1.0 / world_size
            for p in self.module.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
