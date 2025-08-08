import torch
import torch.distributed as dist
from typing import Any, Dict, List

_MB = 1024 * 1024  # bytes in 1 MiB


class DDPBucketed(torch.nn.Module):
    """A *manual* Distributed Data‑Parallel wrapper that **buckets** gradients.

    *   **Broadcasts** rank‑0 parameters so every worker starts identical.
    *   **Buckets** only parameters that *require gradients*; each bucket’s
        byte‑size ≤ ``bucket_size_mb`` (or unlimited when ``None``).
    *   When all grads in a bucket become ready during backward, we **flatten &
        launch one async ``all_reduce``** – reducing comm calls from *N params*
        to *N buckets*.
    *   :py:meth:`finish_gradient_synchronization` waits, scatters the reduced
        tensor back into each ``p.grad``, and averages by ``world_size`` so the
        optimiser sees the same update as on a single GPU.
    """

    # ---------------------------------------------------------------------
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("init_process_group must be called before DDPBucketed")

        self.module = module  # register as sub‑module so state_dict etc. work
        self.world_size = dist.get_world_size()

        # ── 1 · Synchronise initial parameters ───────────────────────────────
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        # ── 2 · Build buckets (trainable params only) ────────────────────────
        cap_bytes = float("inf") if bucket_size_mb is None else bucket_size_mb * _MB

        self._buckets: List[Dict[str, Any]] = []  # runtime state per bucket
        cur_params: List[torch.nn.Parameter] = []
        cur_bytes = 0

        trainable_params = [p for p in self.module.parameters() if p.requires_grad]
        for p in reversed(trainable_params):  # reversed ≈ grad‑ready order
            p_bytes = p.numel() * p.element_size()
            if cur_params and cur_bytes + p_bytes > cap_bytes:
                self._buckets.append({"params": cur_params, "ready": 0, "handles": []})
                cur_params, cur_bytes = [], 0
            cur_params.append(p)
            cur_bytes += p_bytes
        if cur_params:
            self._buckets.append({"params": cur_params, "ready": 0, "handles": []})

        # ── 3 · Register per‑parameter hooks ─────────────────────────────────
        for idx, bucket in enumerate(self._buckets):
            for p in bucket["params"]:
                p.register_post_accumulate_grad_hook(self._make_hook(idx))

    # ------------------------------------------------------------------
    def _make_hook(self, bucket_idx: int):
        """Return a hook that triggers the bucket‑wide ``all_reduce`` once."""

        def _hook(_param: torch.nn.Parameter):  # noqa: ANN001 – _param unused
            bucket = self._buckets[bucket_idx]
            bucket["ready"] += 1
            if bucket["ready"] == len(bucket["params"]):
                # All grads in this bucket are now populated.
                flat = torch.cat([p.grad.view(-1) for p in bucket["params"]])
                flat = flat.to(bucket["params"][0].grad.device)
                work = dist.all_reduce(flat, async_op=True)
                bucket["handles"].append((work, flat))

        return _hook

    # ------------------------------------------------------------------
    def forward(self, *inputs: Any, **kwargs: Any):  # noqa: ANN401
        return self.module(*inputs, **kwargs)

    # ------------------------------------------------------------------
    def finish_gradient_synchronization(self):
        """Wait on outstanding comms, scatter grads, and average."""
        for bucket in self._buckets:
            for work, flat in bucket["handles"]:
                work.wait()
                offset = 0
                for p in bucket["params"]:
                    n = p.numel()
                    p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
                    offset += n
            bucket["handles"].clear()
            bucket["ready"] = 0

        # Average gradients (∑ / world_size)
        if self.world_size > 1:
            scale = 1.0 / self.world_size
            for p in self.module.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
