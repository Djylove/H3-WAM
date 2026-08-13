#!/usr/bin/env python3
"""Two-rank contract test for shared-H3 replicated trainable parameters."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from scripts.h3dreamwam.verify_h3_lingbot_four_stream_fsdp import (
    replicated_parameter_max_difference,
    synchronize_replicated_gradients,
)


def main() -> None:
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        if dist.get_world_size() != 2:
            raise RuntimeError("replicated-gradient parity requires exactly two ranks")
        parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        # Deliberately use different data on each rank. The local gradients are
        # 1 and 3; data-parallel averaging must produce 2 on both ranks.
        multiplier = torch.tensor([1.0 + 2.0 * rank])
        (parameter * multiplier).sum().backward()
        synchronize_replicated_gradients([parameter])
        torch.testing.assert_close(parameter.grad, torch.tensor([2.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.25)
        optimizer.step()
        torch.testing.assert_close(parameter, torch.tensor([0.5]))
        difference = replicated_parameter_max_difference(
            [parameter], torch.device("cpu")
        )
        if difference != 0.0:
            raise RuntimeError(f"replicated parameter divergence: {difference}")
        if rank == 0:
            print(
                '{"event":"replicated_gradient_parity","world_size":2,'
                '"expected_gradient":2.0,"parameter_after_step":0.5,"status":"PASS"}',
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
