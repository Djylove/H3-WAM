#!/usr/bin/env python3
"""Benchmark the project action adapter against the local ComfyUI H3 model.

Run this with the Python environment used by ComfyUI.  Text and video inputs
are synthetic on purpose: this script isolates model loading, the action-slot
gradient path, latency, and peak VRAM before the LIBERO data work starts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--latent-frames", type=int, default=2)
    parser.add_argument("--latent-height", type=int, default=4)
    parser.add_argument("--latent-width", type=int, default=4)
    parser.add_argument("--text-tokens", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--first-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include synthetic FL2VA first-frame conditioning (default: true)",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    comfy_root = args.comfy_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not (comfy_root / "comfy").is_dir():
        raise FileNotFoundError(f"ComfyUI package not found under {comfy_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    sys.path.insert(0, str(comfy_root))

    import comfy.model_management as model_management
    import comfy.sd

    from fastwam.models.h3wam import (
        H3ActionAdapter,
        H3ActionBridge,
        H3ActionFlowScheduler,
        enable_comfy_h3_autograd,
        h3wam_action_training_step,
        make_first_frame_payload,
    )

    if args.backward:
        enable_comfy_h3_autograd()

    device = model_management.get_torch_device()
    if device.type != "cuda":
        raise RuntimeError(f"this benchmark requires CUDA, got {device}")

    model_management.in_training = bool(args.backward)
    torch.cuda.reset_peak_memory_stats(device)

    load_started = time.perf_counter()
    patcher = comfy.sd.load_diffusion_model(str(checkpoint))
    model_management.load_models_gpu([patcher])
    synchronize(device)
    load_seconds = time.perf_counter() - load_started

    h3_model = patcher.model.diffusion_model
    adapter = H3ActionAdapter(action_dim=args.action_dim).to(device=device, dtype=torch.float32)
    bridge = H3ActionBridge(h3_model, adapter, freeze_h3=True)
    scheduler = H3ActionFlowScheduler(
        video_shift=float(h3_model.sigma_shift_video),
        action_shift=float(h3_model.sigma_shift_audio),
    )
    bridge.train(args.backward)

    video_latents = torch.randn(
        1,
        24,
        args.latent_frames,
        args.latent_height,
        args.latent_width,
        device=device,
        dtype=torch.bfloat16,
    )
    actions = torch.randn(
        1,
        args.horizon,
        args.action_dim,
        device=device,
        dtype=torch.float32,
    )
    # Supplying already-refined 5376-D text states skips Qwen and H3's token
    # refiner, which is exactly the cached-text deployment path.
    context = torch.randn(
        1,
        args.text_tokens,
        h3_model.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    video_sigma = torch.tensor([args.sigma], device=device, dtype=torch.float32)
    minimax_payload = None
    if args.first_frame:
        if args.latent_frames < 2 or (args.latent_frames - 2) % 5 != 0:
            raise ValueError(
                "FL2VA latent frames must follow H3's 5n+2 grid, "
                f"got {args.latent_frames}"
            )
        frame_count = ((args.latent_frames - 2) // 5) * 17 + 5
        first_frame_latents = torch.randn(
            1,
            24,
            1,
            args.latent_height,
            args.latent_width,
            device=device,
            dtype=torch.bfloat16,
        )
        minimax_payload = make_first_frame_payload(
            first_frame_latents,
            frame_count=frame_count,
        )

    last_loss = float("nan")

    def one_step(do_backward: bool) -> float:
        nonlocal last_loss
        adapter.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with torch.set_grad_enabled(do_backward):
            loss, _, _ = h3wam_action_training_step(
                bridge,
                video_latents=video_latents,
                actions=actions,
                context=context,
                scheduler=scheduler,
                minimax_payload=minimax_payload,
                video_sigma=video_sigma,
            )
            last_loss = float(loss.detach().item())
            if do_backward:
                loss.backward()
        synchronize(device)
        if do_backward and not any(p.grad is not None for p in adapter.parameters()):
            raise RuntimeError("backward completed without gradients on the action adapter")
        return time.perf_counter() - started

    for _ in range(args.warmup):
        one_step(args.backward)

    torch.cuda.reset_peak_memory_stats(device)
    durations = [one_step(args.backward) for _ in range(args.iterations)]
    result = {
        "checkpoint": str(checkpoint),
        "device": torch.cuda.get_device_name(device),
        "backward": args.backward,
        "action_dim": args.action_dim,
        "horizon": args.horizon,
        "first_frame": args.first_frame,
        "video_latent_shape": list(video_latents.shape),
        "text_tokens": args.text_tokens,
        "load_seconds": load_seconds,
        "step_seconds": durations,
        "mean_step_seconds": sum(durations) / len(durations),
        "last_loss": last_loss,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "adapter_gradients_present": any(p.grad is not None for p in adapter.parameters()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
