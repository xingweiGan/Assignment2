import torch
import torch.distributed as dist
from typing import Any, Dict, List

_MB = 1024 * 1024


class DDPBucketed(torch.nn.Module):
    """Manual Distributed Data Parallel wrapper with *gradient bucketing*.

    • Broadcasts initial parameters from rank‑0 so every worker starts equal.
    • Builds buckets (lists of parameters) whose total size ≤ *bucket_size_mb*.
    • Registers a *post‑accumulate* hook on every parameter.  When all params in
      a bucket have their gradients ready, we launch **one async all‑reduce per
      parameter** in that bucket (still fewer calls than per‑parameter and
      fully overlaps with remaining backward work).
    • `finish_gradient_synchronization()` waits for the queued collectives and
      converts the NCCL/Gloo «sum» into an average so optimizers behave as
      expected.
    """

    # ------------------------------------------------------------------
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()

        if not dist.is_initialized():
            raise RuntimeError("init_process_group must be called before DDPBucketed")

        self.module = module
        self.world_size = dist.get_world_size()

        # 1) Ensure identical initial weights ------------------------------------------------
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        # 2) Build buckets -------------------------------------------------------------------
        bucket_bytes_cap = float("inf") if bucket_size_mb is None else bucket_size_mb * _MB
        self._buckets: List[Dict[str, Any]] = []  # each: {params, ready_cnt, handles}

        current: List[torch.nn.Parameter] = []
        current_size = 0
        for p in reversed(list(self.module.parameters())):  # reverse = roughly backward order
            param_bytes = p.numel() * p.element_size()
            if current and current_size + param_bytes > bucket_bytes_cap:
                self._buckets.append({"params": current, "ready_cnt": 0, "handles": []})
                current, current_size = [], 0
            current.append(p)
            current_size += param_bytes
        if current:
            self._buckets.append({"params": current, "ready_cnt": 0, "handles": []})

        # 3) Register hooks ------------------------------------------------------------------
        for bucket_idx, bucket in enumerate(self._buckets):
            for p in bucket["params"]:
                p.register_post_accumulate_grad_hook(self._make_hook(bucket_idx))

    # ------------------------------------------------------------------
    def _make_hook(self, bucket_idx: int):
        """Return a hook that tracks readiness & queues all‑reduces for a bucket."""

        def _hook(param: torch.nn.Parameter):  # noqa: ANN001
            bucket = self._buckets[bucket_idx]
            bucket["ready_cnt"] += 1
            if bucket["ready_cnt"] == len(bucket["params"]):
                # All grads in this bucket are ready – launch async all‑reduce per param
                for p in bucket["params"]:
                    h = dist.all_reduce(p.grad, async_op=True)
                    bucket["handles"].append(h)
        return _hook

    # ------------------------------------------------------------------
    def forward(self, *inputs: Any, **kwargs: Any):  # noqa: ANN401
        return self.module(*inputs, **kwargs)

    # ------------------------------------------------------------------
    def finish_gradient_synchronization(self):
        """Wait on queued collectives and average gradients."""
        for bucket in self._buckets:
            for h in bucket["handles"]:
                h.wait()
            bucket["handles"].clear()
            bucket["ready_cnt"] = 0  # reset for next backward

        if self.world_size > 1:
            scale = 1.0 / self.world_size
            for p in self.module.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
