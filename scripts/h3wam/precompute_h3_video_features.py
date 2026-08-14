#!/usr/bin/env python3
"""Cache layerwise MiniMax H3 first-frame tokens for an independent ActionDiT."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--h3-tail-delta",
        type=Path,
        help="Feature-only BF16 tail delta applied over the quantized Comfy H3 base.",
    )
    parser.add_argument(
        "--h3-lora-checkpoint",
        type=Path,
        help="Optional H3 video/action LoRA checkpoint applied before feature caching.",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[9, 19, 29, 39, 49])
    parser.add_argument("--output-subdir", default="h3_video_features")
    parser.add_argument(
        "--fixed-context-id",
        help=(
            "Use one cached refined context for every window. This matches "
            "deployment with context-mode=cached while retaining each window's visual latent."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Deterministically split the manifest across independent GPU workers.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based worker index in [0, num-shards).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--action-horizon",
        type=int,
        help=(
            "Override the cached action length used for H3's zero audio tokens. "
            "This must match deployment for short-horizon policies."
        ),
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1000.0,
        help="H3 video-expert timestep; FastWAM action inference prefills video at 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.comfy_root.resolve()))
    import comfy.model_management as model_management
    import comfy.sd

    from fastwam.models.h3wam import (
        H3BlockFeatureCapture,
        inject_h3_attention_lora,
        load_h3_comfy_feature_delta,
        load_h3_lora_state_dict,
        make_first_frame_payload,
    )

    with args.manifest.open(encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    if args.limit is not None:
        items = items[: args.limit]
    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    manifest_items = len(items)
    items = items[args.shard_index :: args.num_shards]
    if not items:
        raise ValueError(
            f"manifest shard {args.shard_index}/{args.num_shards} contains no feature windows"
        )
    if args.h3_tail_delta is not None and args.h3_lora_checkpoint is not None:
        raise ValueError("use either an H3 tail delta or H3 LoRA, not both")

    patcher = comfy.sd.load_diffusion_model(str(args.h3_checkpoint.resolve()))
    model_management.load_models_gpu([patcher])
    model = patcher.model.diffusion_model
    tail_delta_report = None
    if args.h3_tail_delta is not None:
        tail_delta_report = load_h3_comfy_feature_delta(
            model, args.h3_tail_delta
        )
        # Comfy's inference-only fused SwiGLU path bypasses wrapped fc modules.
        model_management.in_training = True
    lora_metadata = None
    if args.h3_lora_checkpoint is not None:
        lora_metadata = torch.load(
            args.h3_lora_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        h3_lora = lora_metadata.get("h3_lora")
        if not h3_lora:
            raise ValueError("H3 LoRA checkpoint does not contain h3_lora weights")
        rank = int(lora_metadata["h3_lora_rank"])
        include_mlp = bool(lora_metadata.get("h3_lora_include_mlp", False))
        if include_mlp:
            model_management.in_training = True
        inject_h3_attention_lora(
            model,
            rank=rank,
            alpha=float(lora_metadata.get("h3_lora_alpha", rank)),
            last_n_blocks=int(lora_metadata["h3_lora_last_blocks"]),
            include_mlp=include_mlp,
        )
        load_h3_lora_state_dict(model, h3_lora)
    device = model_management.get_torch_device()
    layers = tuple(sorted(set(args.layers)))
    if layers[0] < 0 or layers[-1] >= len(model.blocks):
        raise ValueError(f"layers {layers} exceed H3 depth {len(model.blocks)}")

    output_root = args.cache_root / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    fixed_conditioning = None
    if args.fixed_context_id is not None:
        fixed_conditioning = torch.load(
            args.cache_root
            / "refined_contexts"
            / f"{args.fixed_context_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
    started = time.perf_counter()
    completed = 0
    for item in items:
        output = output_root / f"{item['id']}.pt"
        if output.exists() and not args.overwrite:
            completed += 1
            continue
        window = torch.load(
            args.cache_root / "windows" / f"{item['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        conditioning = fixed_conditioning
        if conditioning is None:
            conditioning = torch.load(
                args.cache_root / "refined_contexts" / f"{item['id']}.pt",
                map_location="cpu",
                weights_only=False,
            )
        context = conditioning["context"].to(device=device, dtype=torch.bfloat16)
        first_frame = window["first_frame_latents"].to(
            device=device, dtype=torch.bfloat16
        )
        if "video_latents" in window:
            video = torch.zeros_like(
                window["video_latents"], device=device, dtype=torch.bfloat16
            )
        else:
            video = torch.zeros(
                (1, 24, 12, first_frame.shape[-2], first_frame.shape[-1]),
                device=device,
                dtype=torch.bfloat16,
            )
        action_horizon = (
            int(args.action_horizon)
            if args.action_horizon is not None
            else int(window["actions"].shape[0])
        )
        if action_horizon <= 0:
            raise ValueError("action-horizon must be positive")
        audio = torch.zeros(
            (1, 32, 2, action_horizon), device=device, dtype=torch.float32
        )
        text_len = int(context.shape[1])
        frame_rows = int(
            first_frame.shape[2]
            * (first_frame.shape[3] // 2)
            * (first_frame.shape[4] // 2)
        )
        capture = H3BlockFeatureCapture(
            layers, token_start=text_len, token_stop=text_len + frame_rows
        )
        payload = make_first_frame_payload(
            first_frame, frame_count=int(window["h3_frame_count"])
        )
        payload["text_token_tags"] = conditioning["token_tags"].to(device)
        with torch.inference_mode():
            model(
                [video, audio],
                torch.tensor([args.timestep], device=device),
                context,
                transformer_options=capture.transformer_options(),
                minimax_payload=payload,
            )
        features = capture.stacked().to(device="cpu", dtype=torch.bfloat16)
        torch.save(
            {
                "features": features,
                "layers": layers,
                "token_start": text_len,
                "token_stop": text_len + frame_rows,
                "episode": int(item["episode"]),
                "start": int(item["start"]),
                "context_id": args.fixed_context_id or item["id"],
                "timestep": float(args.timestep),
                "action_horizon": action_horizon,
                "h3_lora_checkpoint": (
                    None
                    if args.h3_lora_checkpoint is None
                    else str(args.h3_lora_checkpoint.resolve())
                ),
                "h3_tail_delta": (
                    None
                    if args.h3_tail_delta is None
                    else str(args.h3_tail_delta.resolve())
                ),
                "h3_tail_delta_source_step": (
                    None if tail_delta_report is None else tail_delta_report.source_step
                ),
                "manifest_items": manifest_items,
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
            },
            output,
        )
        completed += 1
        if completed % args.progress_every == 0 or completed == len(items):
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(items),
                        "manifest_items": manifest_items,
                        "num_shards": args.num_shards,
                        "shard_index": args.shard_index,
                        "elapsed_seconds": round(elapsed, 2),
                        "seconds_per_window": round(elapsed / completed, 3),
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                        / 2**30,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
