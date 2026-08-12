#!/usr/bin/env python3
"""Overfit the H3 action adapter on one cached LIBERO window."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("window", type=Path)
    parser.add_argument("context", type=Path)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-last-blocks", type=int, default=50)
    parser.add_argument("--video-loss-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.comfy_root.resolve()))

    import comfy.model_management as model_management
    import comfy.sd

    from fastwam.models.h3wam import (
        H3ActionAdapter,
        H3ActionBridge,
        H3ActionFlowScheduler,
        enable_comfy_h3_autograd,
        h3_lora_parameters,
        h3_lora_state_dict,
        inject_h3_attention_lora,
        make_first_frame_payload,
        prepare_h3wam_flow_batch,
    )

    if args.steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(args.seed)
    model_management.in_training = True
    enable_comfy_h3_autograd(checkpoint_blocks=True)
    device = model_management.get_torch_device()
    if device.type != "cuda":
        raise RuntimeError(f"CUDA is required, got {device}")

    window = torch.load(args.window, map_location="cpu", weights_only=False)
    cached_context = torch.load(args.context, map_location="cpu", weights_only=False)
    load_started = time.perf_counter()
    patcher = comfy.sd.load_diffusion_model(str(args.checkpoint.resolve()))
    model_management.load_models_gpu([patcher])
    torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - load_started

    h3_model = patcher.model.diffusion_model
    actions = window["actions"].unsqueeze(0).to(device=device, dtype=torch.float32)
    state = window["state"].unsqueeze(0).to(device=device, dtype=torch.float32)
    video_latents = window["video_latents"].to(device=device, dtype=torch.bfloat16)
    first_frame_latents = window["first_frame_latents"].to(
        device=device,
        dtype=torch.bfloat16,
    )
    context = cached_context["context"].to(device=device, dtype=torch.bfloat16)
    token_tags = cached_context["token_tags"]

    adapter = H3ActionAdapter(
        action_dim=actions.shape[-1],
        state_dim=state.shape[-1],
    ).to(device=device, dtype=torch.float32)
    bridge = H3ActionBridge(h3_model, adapter, freeze_h3=True)
    lora_report = None
    if args.lora_rank > 0:
        lora_report = inject_h3_attention_lora(
            h3_model,
            rank=args.lora_rank,
            last_n_blocks=args.lora_last_blocks,
        )
    bridge.train()
    scheduler = H3ActionFlowScheduler(
        video_shift=float(h3_model.sigma_shift_video),
        action_shift=float(h3_model.sigma_shift_audio),
    )
    trainable_parameters = list(adapter.parameters()) + h3_lora_parameters(h3_model)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)

    video_sigma = torch.tensor([args.sigma], device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    video_noise = torch.randn(
        video_latents.shape,
        generator=generator,
        device=device,
        dtype=video_latents.dtype,
    )
    action_noise = torch.randn(
        actions.shape,
        generator=generator,
        device=device,
        dtype=actions.dtype,
    )
    flow_batch = prepare_h3wam_flow_batch(
        video_latents=video_latents,
        actions=actions,
        scheduler=scheduler,
        video_sigma=video_sigma,
        video_noise=video_noise,
        action_noise=action_noise,
    )
    payload = make_first_frame_payload(
        first_frame_latents,
        frame_count=int(window["h3_frame_count"]),
        seed=args.seed,
    )
    payload["text_token_tags"] = token_tags

    losses = []
    torch.cuda.reset_peak_memory_stats(device)
    train_started = time.perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = bridge(
            video_latents=flow_batch.noisy_video_latents,
            noisy_actions=flow_batch.noisy_actions,
            timestep=flow_batch.timestep,
            context=context,
            state=state,
            minimax_payload=payload,
        )
        action_loss = (
            output.action_velocity.float() - flow_batch.action_target.float()
        ).square().mean()
        video_loss = (
            output.video_velocity.float() - flow_batch.video_target.float()
        ).square().mean()
        loss = action_loss + args.video_loss_weight * video_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - train_started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "h3_lora": h3_lora_state_dict(h3_model),
            "action_dim": adapter.action_dim,
            "state_dim": adapter.state_dim,
            "task": window["task"],
            "losses": losses,
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "steps": args.steps,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "loss_ratio": losses[-1] / losses[0],
                "model_load_seconds": model_load_seconds,
                "train_seconds": train_seconds,
                "seconds_per_step": train_seconds / args.steps,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
                "lora_parameters": 0 if lora_report is None else lora_report.parameters,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
