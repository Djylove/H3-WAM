#!/usr/bin/env python3
"""Small CUDA/BF16/NCCL preflight for an H3 multi-GPU training node."""

from __future__ import annotations

import json
import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    left = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16)
    right = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16)
    product = left @ right
    if not bool(torch.isfinite(product).all()):
        raise FloatingPointError(f"rank {rank} produced non-finite BF16 matmul")

    reduced = torch.tensor(float(rank + 1), device=device)
    if world_size > 1:
        dist.all_reduce(reduced)
    expected = world_size * (world_size + 1) / 2
    if float(reduced.item()) != expected:
        raise RuntimeError(
            f"rank {rank} all-reduce mismatch: {reduced.item()} != {expected}"
        )

    torch.cuda.synchronize(device)
    report = {
        "host": socket.gethostname(),
        "rank": rank,
        "world_size": world_size,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "all_reduce": float(reduced.item()),
        "bf16_matmul_mean_abs": float(product.float().abs().mean().item()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    print(json.dumps(report, sort_keys=True), flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
