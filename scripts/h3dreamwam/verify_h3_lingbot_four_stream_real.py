#!/usr/bin/env python3
"""Real MiniMax-H3 layer smoke for the LingBot-style four-stream port."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch


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
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt())


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.video_frames < 2:
        raise ValueError("steps must be positive and video-frames must be >=2")
    if args.tokens_per_frame <= 0 or args.action_horizon <= 0:
        raise ValueError("token/action sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from diffusers import MiniMaxH3Transformer3DModel
    from fastwam.models.h3dreamwam import (
        H3DreamActionExpert,
        align_h3_action_chunk_ids,
        four_stream_h3_action_layer,
        initialize_action_expert_from_h3,
    )
    from fastwam.models.h3dreamwam.model import apply_h3_rotary

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
    h3_block = h3.transformer_blocks[-1].to(device)
    action_block = action_expert.blocks[-1].to(device)
    h3_block.requires_grad_(True)
    action_block.requires_grad_(True)
    for module in (
        h3.proj_in,
        h3.time_proj,
        h3.time_embedder,
        h3.rope,
        action_expert.action_embedding,
        action_expert.state_embedding,
        action_expert.context_embedding,
        action_expert.time_embedding,
        action_expert.time_projection,
    ):
        module.to(device)

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
    one_frame_positions = torch.stack(
        (
            torch.zeros_like(spatial),
            torch.div(spatial, side, rounding_mode="floor"),
            spatial % side,
        ),
        dim=-1,
    ).float()
    position_ids = torch.cat(
        [
            one_frame_positions
            + torch.tensor([frame, 0, 0], device=device)
            for frame in range(args.video_frames)
        ],
        dim=0,
    )
    rotary_emb = h3.rope(position_ids)
    input_width = h3.proj_in.in_features
    clean_rows = torch.randn(
        1, video_tokens, input_width, generator=generator, device=device
    )
    noisy_rows = torch.randn(
        1, video_tokens, input_width, generator=generator, device=device
    )
    clean_video = h3.proj_in(clean_rows.to(h3.proj_in.weight.dtype))
    noisy_video = h3.proj_in(noisy_rows.to(h3.proj_in.weight.dtype))
    # H3 intentionally keeps some I/O modules in FP32 while its transformer
    # blocks are BF16. The production paired model casts at every FSDP block
    # boundary; mirror that contract in this isolated layer smoke.
    block_dtype = h3_block.attn.to_q.weight.dtype
    clean_video = clean_video.to(block_dtype)
    noisy_video = noisy_video.to(block_dtype)

    context = torch.randn(
        1, 2, 5120, generator=generator, device=device, dtype=torch.bfloat16
    )
    context_mask = torch.ones(1, 2, dtype=torch.bool, device=device)
    state = torch.randn(1, 8, generator=generator, device=device)
    clean_actions = torch.randn(
        1,
        args.action_horizon,
        7,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    noisy_actions = torch.randn(
        1,
        args.action_horizon,
        7,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    noisy_action_state = action_expert.prepare(
        noisy_actions=noisy_actions,
        timestep=torch.full((1,), 500.0, device=device),
        context=context,
        context_mask=context_mask,
        state=state,
    )
    clean_action_state = action_expert.prepare(
        noisy_actions=clean_actions,
        timestep=torch.zeros(1, device=device),
        context=context,
        context_mask=context_mask,
        state=state,
    )
    noisy_time = h3.time_embedder(
        h3.time_proj(torch.tensor([0.5], device=device)).to(
            h3.time_embedder.linear_1.weight.dtype
        )
    )
    clean_time = h3.time_embedder(
        h3.time_proj(torch.tensor([0.0], device=device)).to(
            h3.time_embedder.linear_1.weight.dtype
        )
    )
    adaln_indices = torch.zeros(video_tokens, dtype=torch.long, device=device)
    parameters = [*h3_block.parameters(), *action_block.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)

    history = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = four_stream_h3_action_layer(
            h3_block=h3_block,
            action_block=action_block,
            noisy_video_hidden=noisy_video,
            clean_video_hidden=clean_video,
            noisy_action_hidden=noisy_action_state["tokens"],
            clean_action_hidden=clean_action_state["tokens"],
            noisy_h3_temb=noisy_time,
            clean_h3_temb=clean_time,
            noisy_h3_adaln_indices=adaln_indices,
            clean_h3_adaln_indices=adaln_indices,
            h3_rotary_emb=rotary_emb,
            h3_apply_rotary=apply_h3_rotary,
            noisy_action_time_modulation=noisy_action_state["time_modulation"],
            clean_action_time_modulation=clean_action_state["time_modulation"],
            action_context=noisy_action_state["context"],
            action_context_mask=noisy_action_state["context_mask"],
            video_chunk_ids=video_chunks,
            action_chunk_ids=action_chunks,
        )
        future_rows = video_chunks > video_chunks.min()
        video_loss = outputs[0][:, future_rows].float().square().mean()
        action_loss = outputs[2].float().square().mean()
        loss = video_loss + action_loss
        loss.backward()
        h3_gradient = grad_norm(list(h3_block.parameters()))
        action_gradient = grad_norm(list(action_block.parameters()))
        if not all(
            math.isfinite(value) and value > 0
            for value in (float(loss), h3_gradient, action_gradient)
        ):
            raise RuntimeError("four-stream smoke produced non-finite/zero signal")
        optimizer.step()
        history.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "video_loss": float(video_loss.detach()),
                "action_loss": float(action_loss.detach()),
                "h3_gradient_norm": h3_gradient,
                "action_gradient_norm": action_gradient,
            }
        )
        print(json.dumps(history[-1]), flush=True)

    report = {
        "event": "h3_lingbot_four_stream_real_smoke",
        "model": str(args.model.resolve()),
        "steps": args.steps,
        "video_frames": args.video_frames,
        "tokens_per_frame": args.tokens_per_frame,
        "video_tokens": video_tokens,
        "action_horizon": args.action_horizon,
        "actions_per_chunk": args.actions_per_chunk,
        "video_chunks": int(video_chunks.max()) + 1,
        "action_chunks": int(action_chunks.max()) + 1,
        "history": history,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "initialization": initialization.__dict__,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
