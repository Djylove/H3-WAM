#!/usr/bin/env python3
"""Full-50-layer FSDP smoke for the H3 LingBot four-stream model."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--last-trainable-layers", type=int, default=2)
    parser.add_argument("--video-frames", type=int, default=2)
    parser.add_argument("--tokens-per-frame", type=int, default=98)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--actions-per-chunk", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def global_grad_norm(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    marker: str,
    device: torch.device,
) -> float:
    total = torch.zeros((), device=device, dtype=torch.float32)
    for name, parameter in named_parameters:
        if marker in name and parameter.grad is not None:
            total += parameter.grad.detach().float().square().sum()
    dist.all_reduce(total)
    return float(total.sqrt())


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2 or not torch.cuda.is_available():
        raise RuntimeError("this smoke requires multi-GPU CUDA torchrun")
    if not 1 <= args.last_trainable_layers <= 50:
        raise ValueError("last-trainable-layers must be in [1,50]")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    from diffusers import MiniMaxH3Transformer3DModel
    from fastwam.models.h3dreamwam import (
        H3DreamActionExpert,
        H3LingBotPairedLayer,
        H3LingBotWAM,
        align_h3_action_chunk_ids,
        initialize_action_expert_from_h3,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    started = time.perf_counter()
    h3 = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model.resolve(),
        subfolder="transformer",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        action_expert = H3DreamActionExpert(
            action_dim=7,
            state_dim=8,
            text_dim=5120,
            hidden_dim=1024,
            ffn_dim=4096,
            num_heads=56,
            head_dim=128,
            num_layers=50,
            frequency_dim=256,
            full_width_rmsnorm=True,
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    initialization = initialize_action_expert_from_h3(
        action_expert, h3, alpha_scaling=True
    )
    h3.requires_grad_(False)
    action_expert.requires_grad_(False)
    for block in h3.transformer_blocks[-args.last_trainable_layers :]:
        block.requires_grad_(True)
    for block in action_expert.blocks[-args.last_trainable_layers :]:
        block.requires_grad_(True)
    h3.proj_out.requires_grad_(True)
    action_expert.output.requires_grad_(True)

    model = H3LingBotWAM(
        h3,
        action_expert,
        use_gradient_checkpointing=True,
        compute_dtype=torch.bfloat16,
    )
    ignored_modules = [*model.h3.children(), *model.action_expert.children()]
    ignored_modules = [
        module for module in ignored_modules if len(list(module.parameters())) > 0
    ]
    for module in ignored_modules:
        module.to(device)
    model.train()
    model = FSDP(
        model,
        # The paired unit must be the only nested FSDP boundary. Wrapping its
        # H3 block again leaves AdaLN parameters as 1-D shards during the outer
        # forward (observed as ``mat2 must be a matrix``).
        auto_wrap_policy=ModuleWrapPolicy({H3LingBotPairedLayer}),
        ignored_modules=ignored_modules,
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=False,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        ),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        foreach=False,
    )
    load_seconds = time.perf_counter() - started

    video_tokens = args.video_frames * args.tokens_per_frame
    input_width = model.module.h3.proj_in.in_features
    frame_ids = torch.arange(args.video_frames, device=device).repeat_interleave(
        args.tokens_per_frame
    )
    video_chunks, action_chunks = align_h3_action_chunk_ids(
        video_frame_ids=frame_ids,
        action_horizon=args.action_horizon,
        actions_per_chunk=args.actions_per_chunk,
    )
    side = max(1, int(math.sqrt(args.tokens_per_frame)))
    spatial = torch.arange(args.tokens_per_frame, device=device)
    positions = []
    for frame in range(args.video_frames):
        positions.append(
            torch.stack(
                (
                    torch.full_like(spatial, frame),
                    torch.div(spatial, side, rounding_mode="floor"),
                    spatial % side,
                ),
                dim=-1,
            ).float()
        )
    video_position_ids = torch.cat(positions, dim=0)
    context_position_ids = torch.zeros(3, 3, device=device)
    clean_video = torch.randn(
        1, video_tokens, input_width, generator=generator, device=device
    )
    video_noise = torch.randn(
        clean_video.shape, generator=generator, device=device
    )
    sigma = torch.tensor([0.5], device=device)
    noisy_video = (1.0 - sigma[:, None, None]) * clean_video
    noisy_video += sigma[:, None, None] * video_noise
    clean_action = torch.randn(
        1, args.action_horizon, 7, generator=generator, device=device
    )
    action_noise = torch.randn(
        clean_action.shape, generator=generator, device=device
    )
    noisy_action = (1.0 - sigma[:, None, None]) * clean_action
    noisy_action += sigma[:, None, None] * action_noise
    context = torch.randn(
        1, 2, 5120, generator=generator, device=device, dtype=torch.bfloat16
    )
    state = torch.randn(1, 8, generator=generator, device=device)
    context_mask = torch.ones(1, 2, device=device, dtype=torch.bool)
    video_target = clean_video - video_noise
    action_target = action_noise - clean_action

    torch.cuda.reset_peak_memory_stats(device)
    history = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            noisy_video_rows=noisy_video,
            clean_video_rows=clean_video,
            video_position_ids=video_position_ids,
            video_chunk_ids=video_chunks,
            noisy_video_timestep=sigma,
            clean_video_timestep=torch.zeros_like(sigma),
            noisy_actions=noisy_action,
            clean_actions=clean_action,
            action_chunk_ids=action_chunks,
            noisy_action_timestep=sigma * 1000.0,
            context=context,
            context_position_ids=context_position_ids,
            state=state,
            context_mask=context_mask,
        )
        future = video_chunks > video_chunks.min()
        video_loss = F.mse_loss(
            output.video_velocity_rows[:, future].float(),
            video_target[:, future].float(),
        )
        action_loss = F.mse_loss(
            output.action_velocity.float(), action_target.float()
        )
        loss = video_loss + action_loss
        loss.backward()
        named = list(model.named_parameters())
        h3_gradient = global_grad_norm(named, ".h3_block.", device)
        action_gradient = global_grad_norm(named, ".action_block.", device)
        if not all(
            math.isfinite(value) and value > 0
            for value in (
                float(loss.detach()),
                h3_gradient,
                action_gradient,
            )
        ):
            raise RuntimeError("full FSDP smoke produced non-finite/zero signal")
        # Clip the two experts independently. A large freshly interpolated
        # ActionDiT norm must not suppress the pretrained H3 update (or the
        # reverse), and FSDP owns the global norm calculation for its shards.
        all_with_grad = [
            parameter for parameter in model.parameters() if parameter.grad is not None
        ]
        expert_clip_norms = {}
        expert_markers = {
            "h3": (".h3_block.", ".h3.proj_out."),
            "action": (".action_block.", ".action_expert.output."),
        }
        for expert, markers in expert_markers.items():
            active_ids = {
                id(parameter)
                for name, parameter in named
                if any(marker in name for marker in markers)
                and parameter.grad is not None
            }
            hidden = [
                (parameter, parameter.grad)
                for parameter in all_with_grad
                if id(parameter) not in active_ids
            ]
            for parameter, _ in hidden:
                parameter.grad = None
            expert_clip_norms[expert] = float(model.clip_grad_norm_(1.0))
            for parameter, gradient in hidden:
                parameter.grad = gradient
        optimizer.step()
        item = {
            "step": step,
            "loss": float(loss.detach()),
            "video_loss": float(video_loss.detach()),
            "action_loss": float(action_loss.detach()),
            "h3_gradient_norm": h3_gradient,
            "action_gradient_norm": action_gradient,
            "expert_clip_norms": expert_clip_norms,
        }
        history.append(item)
        if rank == 0:
            print(json.dumps(item), flush=True)

    if rank == 0:
        report = {
            "event": "h3_lingbot_four_stream_full_fsdp_smoke",
            "world_size": world_size,
            "steps": args.steps,
            "layers": 50,
            "last_trainable_layers": args.last_trainable_layers,
            "video_tokens": video_tokens,
            "action_horizon": args.action_horizon,
            "history": history,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "load_seconds": load_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "initialization": initialization.__dict__,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
