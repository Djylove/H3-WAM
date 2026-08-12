#!/usr/bin/env python3
"""Real H3 one-layer gradient smoke for the shared LingBot backbone port."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--video-frames", type=int, default=2)
    parser.add_argument("--tokens-per-frame", type=int, default=98)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--actions-per-chunk", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.video_frames < 2:
        raise ValueError("steps must be positive and video-frames must be >=2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from diffusers import MiniMaxH3Transformer3DModel
    from fastwam.models.h3dreamwam import (
        H3LingBotSharedWAM,
        align_h3_action_chunk_ids,
    )

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    started = time.perf_counter()
    h3 = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model.resolve(),
        subfolder="transformer",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    h3.requires_grad_(False)
    # Isolate the last pretrained block while exercising the exact shared
    # four-stream model wrapper and the new action adapters.
    h3.transformer_blocks = torch.nn.ModuleList([h3.transformer_blocks[-1]])
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = H3LingBotSharedWAM(
            h3,
            action_dim=7,
            state_dim=8,
            text_dim=5120,
            use_gradient_checkpointing=False,
            compute_dtype=torch.bfloat16,
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    model.shared_layers[0].requires_grad_(True)
    model.action_adapters.requires_grad_(True)
    model.h3.proj_out.requires_grad_(True)
    model.to(device).train()

    video_tokens = args.video_frames * args.tokens_per_frame
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
    video_positions = torch.cat(positions, dim=0)
    # Match upstream get_mesh_id(action=True): action rows occupy temporal
    # subpositions and use -1 on both spatial axes.
    action_positions = torch.stack(
        (
            torch.arange(args.action_horizon, device=device).float()
            / args.actions_per_chunk,
            torch.full((args.action_horizon,), -1.0, device=device),
            torch.full((args.action_horizon,), -1.0, device=device),
        ),
        dim=-1,
    )
    input_width = model.h3.proj_in.in_features
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
    arguments = {
        "noisy_video_rows": noisy_video,
        "clean_video_rows": clean_video,
        "video_position_ids": video_positions,
        "video_chunk_ids": video_chunks,
        "noisy_video_timestep": sigma,
        "clean_video_timestep": torch.ones_like(sigma),
        "noisy_actions": noisy_action,
        "clean_actions": clean_action,
        "action_position_ids": action_positions,
        "action_chunk_ids": action_chunks,
        "noisy_action_timestep": sigma,
        "clean_action_timestep": torch.ones_like(sigma),
        "context": context,
        "context_position_ids": torch.zeros(3, 3, device=device),
        "state": state,
        "context_mask": torch.ones(1, 2, device=device, dtype=torch.bool),
    }
    video_target = clean_video - video_noise
    action_target = action_noise - clean_action
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    history = []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(**arguments)
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
        block_gradient = grad_norm(list(model.shared_layers[0].parameters()))
        adapter_gradient = grad_norm(list(model.action_adapters.parameters()))
        if not all(
            math.isfinite(value) and value > 0
            for value in (float(loss.detach()), block_gradient, adapter_gradient)
        ):
            raise RuntimeError("shared four-stream smoke produced invalid signal")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        item = {
            "step": step,
            "loss": float(loss.detach()),
            "video_loss": float(video_loss.detach()),
            "action_loss": float(action_loss.detach()),
            "shared_block_gradient_norm": block_gradient,
            "action_adapter_gradient_norm": adapter_gradient,
        }
        history.append(item)
        print(json.dumps(item), flush=True)

    report = {
        "event": "h3_lingbot_shared_four_stream_real_layer_smoke",
        "model": str(args.model.resolve()),
        "layers": 1,
        "steps": args.steps,
        "video_tokens": video_tokens,
        "action_horizon": args.action_horizon,
        "action_modality_id": model.action_modality_id,
        "history": history,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
