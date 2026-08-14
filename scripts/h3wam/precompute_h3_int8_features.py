#!/usr/bin/env python3
"""Cache MiniMax-H3 observation features with the standalone INT8 runtime.

The script deliberately uses only the official Diffusers packing helpers, the
project-native H3 INT8 backbone and ``comfy-kitchen``'s CUDA kernels.  It does
not import ComfyUI, its model manager, nodes, server, or workflow runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F


AUDIO_CHANNELS = 2
AUDIO_LATENT_CHANNELS = 32
PATCH_SIZE = (1, 2, 2)
DREAMWAM_KV_SCHEMA = "h3_dreamwam_kv_v1"
DREAMWAM_KV_LAYERS = (9, 19, 29, 39, 49)
DREAMWAM_COMMIT = "6e989facc0c452fd3488d75f60bc36411005558c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--output-subdir", default="h3_int8_last32_features")
    parser.add_argument(
        "--dreamwam-kv-output-subdir",
        default="h3_int8_dreamwam_kv_5x32",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=(49,))
    parser.add_argument(
        "--dreamwam-kv-carrier",
        action="store_true",
        help=(
            "Write the independent h3_dreamwam_kv_v1 schema instead of the "
            "historical pooled hidden-feature schema. Disabled by default."
        ),
    )
    parser.add_argument(
        "--also-starwam-feature-cache",
        action="store_true",
        help=(
            "While writing DreamWAM K/V, also write the standard pooled hidden "
            "feature cache selected by --layers/--output-subdir from the same H3 forward."
        ),
    )
    parser.add_argument(
        "--dreamwam-kv-layers",
        type=int,
        nargs="+",
        default=DREAMWAM_KV_LAYERS,
    )
    parser.add_argument("--capture-token-count", type=int, default=32)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--target-latent-frames", type=int, default=12)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--condition-video-timestep", type=float, default=1.0)
    parser.add_argument(
        "--capture-compatibility",
        choices=("none", "comfy_alias_v1"),
        default="none",
        help=(
            "Reproduce an explicitly identified historical capture contract. "
            "comfy_alias_v1 repeats the final selected layer because the old "
            "Comfy hook retained aliases to an in-place-mutated hidden tensor."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def pool_feature_tokens(features: torch.Tensor, token_count: int) -> torch.Tensor:
    """Apply StarWAM's exact feature-token pooling convention."""

    if features.ndim != 3:
        raise ValueError("features must be [layers,tokens,hidden]")
    if token_count <= 0 or features.shape[1] <= token_count:
        return features
    return F.adaptive_avg_pool1d(features.transpose(1, 2), token_count).transpose(1, 2)


def apply_capture_compatibility(features: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply a named historical feature-capture contract without changing default caches."""

    if features.ndim != 3:
        raise ValueError("features must be [layers,tokens,hidden]")
    if mode == "none":
        return features
    if mode == "comfy_alias_v1":
        return features[-1:].expand_as(features).clone()
    raise ValueError(f"unsupported capture compatibility mode {mode!r}")


def pool_kv_tokens(tensor: torch.Tensor, token_count: int) -> torch.Tensor:
    """Pool one H3 layer's projected K or V over sequence only."""

    if tensor.ndim != 3:
        raise ValueError("K/V tensor must be [tokens,heads,head_dim]")
    if token_count <= 0 or tensor.shape[0] <= token_count:
        return tensor
    tokens, heads, head_dim = tensor.shape
    flattened = tensor.reshape(tokens, heads * head_dim)
    pooled = F.adaptive_avg_pool1d(
        flattened.transpose(0, 1).unsqueeze(0), token_count
    ).squeeze(0).transpose(0, 1)
    return pooled.reshape(token_count, heads, head_dim)


def prepare_dreamwam_kv_cache(
    captured_kv: dict[int, dict[str, torch.Tensor]],
    *,
    layers: tuple[int, ...],
    token_count: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """Detach, pool and clone distinct per-layer K/V tensors for disk storage."""

    if tuple(sorted(captured_kv)) != layers:
        raise ValueError(
            f"captured K/V layers differ from requested layers: "
            f"{tuple(sorted(captured_kv))} vs {layers}"
        )
    result: dict[int, dict[str, torch.Tensor]] = {}
    storage_pointers: set[int] = set()
    expected_shape = None
    for layer in layers:
        item = captured_kv[layer]
        if set(item) != {"k", "v"}:
            raise ValueError(f"captured K/V layer {layer} must contain k and v")
        result[layer] = {}
        for name in ("k", "v"):
            source = item[name]
            if source.ndim != 4 or source.shape[0] != 1:
                raise ValueError(
                    f"captured layer {layer} {name} must be [1,tokens,heads,head_dim]"
                )
            tensor = pool_kv_tokens(source[0], token_count).to(
                device="cpu", dtype=torch.bfloat16
            ).clone()
            if expected_shape is None:
                expected_shape = tuple(tensor.shape)
            elif tuple(tensor.shape) != expected_shape:
                raise ValueError("all captured K/V tensors must share one shape")
            pointer = tensor.untyped_storage().data_ptr()
            if pointer in storage_pointers:
                raise RuntimeError("DreamWAM K/V cache tensors unexpectedly alias storage")
            storage_pointers.add(pointer)
            result[layer][name] = tensor
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.timestep <= 1.0:
        raise ValueError("H3 curve timestep must be in [0,1]")
    if not 0.0 <= args.condition_video_timestep <= 1.0:
        raise ValueError("H3 condition-video curve timestep must be in [0,1]")
    if min(args.action_horizon, args.target_latent_frames, args.progress_every) <= 0:
        raise ValueError("positive horizon, latent-frame and progress arguments are required")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0,num-shards)")
    if args.also_starwam_feature_cache and not args.dreamwam_kv_carrier:
        raise ValueError("--also-starwam-feature-cache requires --dreamwam-kv-carrier")

    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )
    from fastwam.models.h3wam import H3Int8FeatureBackbone

    rows = [
        json.loads(line)
        for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    manifest_items = len(rows)
    rows = rows[args.shard_index :: args.num_shards]
    if not rows:
        raise ValueError("selected manifest shard is empty")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cache_root = args.cache_root.resolve()
    output_root = cache_root / (
        args.dreamwam_kv_output_subdir
        if args.dreamwam_kv_carrier
        else args.output_subdir
    )
    output_root.mkdir(parents=True, exist_ok=True)
    starwam_output_root = (
        cache_root / args.output_subdir
        if args.also_starwam_feature_cache
        else None
    )
    if starwam_output_root is not None:
        if starwam_output_root == output_root:
            raise ValueError("DreamWAM and StarWAM output subdirectories must differ")
        starwam_output_root.mkdir(parents=True, exist_ok=True)
    layers = tuple(
        sorted(
            set(
                int(layer)
                for layer in (
                    args.dreamwam_kv_layers
                    if args.dreamwam_kv_carrier
                    else args.layers
                )
            )
        )
    )
    if args.dreamwam_kv_carrier and args.capture_compatibility != "none":
        raise ValueError("DreamWAM K/V caches do not support historical feature alias mode")
    feature_layers = tuple(sorted(set(int(layer) for layer in args.layers)))
    model = H3Int8FeatureBackbone.from_checkpoint(args.h3_checkpoint).to(device).eval()

    layouts: dict[tuple[str, int, int], dict[str, torch.Tensor | int]] = {}
    started = time.perf_counter()
    completed = 0
    written = 0
    written_files = 0
    for row in rows:
        output = output_root / f"{row['id']}.pt"
        starwam_output = (
            starwam_output_root / f"{row['id']}.pt"
            if starwam_output_root is not None
            else None
        )
        required_outputs = [output]
        if starwam_output is not None:
            required_outputs.append(starwam_output)
        if all(path.exists() for path in required_outputs) and not args.overwrite:
            completed += 1
            continue
        window = torch.load(
            cache_root / "windows" / f"{row['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        context_id = str(row["context_id"])
        context_item = torch.load(
            cache_root / "contexts" / f"{context_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        first = window["first_frame_latents"].to(device=device, dtype=torch.float32)
        _, channels, _, latent_height, latent_width = first.shape
        layout_key = (context_id, latent_height, latent_width)
        if layout_key not in layouts:
            packed = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
                text_token_tags=context_item["token_tags"].long(),
                num_latent_frames=args.target_latent_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                num_audio_latents=args.action_horizon,
                patch_size=PATCH_SIZE,
                audio_channels=AUDIO_CHANNELS,
                audio_tag=2,
                video_tag=0,
                keyframe_anchors=("first",),
            )
            (
                position_ids,
                token_tags,
                video_indices,
                audio_indices,
                text_indices,
                num_condition_video_rows,
                num_condition_audio_rows,
            ) = packed
            unique_timesteps, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                video_indices=video_indices,
                audio_indices=audio_indices,
                num_condition_video_rows=num_condition_video_rows,
                num_condition_audio_rows=num_condition_audio_rows,
                num_text_tokens=text_indices.numel(),
                video_timestep=args.timestep,
                audio_timestep=0.0,
                condition_video_timestep=args.condition_video_timestep,
                condition_audio_timestep=1.0,
            )
            layouts[layout_key] = {
                "position_ids": position_ids.to(device),
                "token_tags": token_tags.to(device),
                "video_indices": video_indices.to(device),
                "audio_indices": audio_indices.to(device),
                "text_indices": text_indices.to(device),
                "unique_timesteps": unique_timesteps.to(device),
                "timestep_indices": timestep_indices.to(device),
                "condition_indices": video_indices[:num_condition_video_rows].to(device),
                "num_condition_video_rows": int(num_condition_video_rows),
            }
        layout = layouts[layout_key]

        target = torch.zeros(
            (1, channels, args.target_latent_frames, latent_height, latent_width),
            device=device,
            dtype=torch.float32,
        )
        row_dim = channels * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2]
        first_rows = patchify_video_latents(first, PATCH_SIZE).reshape(1, -1, row_dim)
        target_rows = patchify_video_latents(target, PATCH_SIZE).reshape(1, -1, row_dim)
        video_rows = torch.cat((first_rows, target_rows), dim=1)
        audio_rows = torch.zeros(
            (1, args.action_horizon * AUDIO_CHANNELS, AUDIO_LATENT_CHANNELS),
            device=device,
            dtype=torch.float32,
        )
        context = context_item["context"].to(device=device, dtype=torch.float32)
        context_width = int(context.shape[-1])
        context_mode = "refined_h3" if context_width == 5376 else "raw_qwen"
        with torch.inference_mode():
            result = model(
                hidden_states=video_rows,
                audio_hidden_states=audio_rows,
                encoder_hidden_states=context,
                timestep=layout["unique_timesteps"],
                timestep_indices=layout["timestep_indices"],
                token_tags=layout["token_tags"],
                position_ids=layout["position_ids"],
                video_indices=layout["video_indices"],
                audio_indices=layout["audio_indices"],
                text_indices=layout["text_indices"],
                capture_layers=(
                    feature_layers
                    if (not args.dreamwam_kv_carrier or args.also_starwam_feature_cache)
                    else ()
                ),
                capture_indices=(
                    layout["condition_indices"]
                    if (not args.dreamwam_kv_carrier or args.also_starwam_feature_cache)
                    else None
                ),
                capture_kv_layers=layers if args.dreamwam_kv_carrier else (),
                kv_capture_indices=(
                    layout["condition_indices"]
                    if args.dreamwam_kv_carrier
                    else None
                ),
            )
        cache_outputs: list[tuple[Path, dict[str, object]]] = []
        if args.dreamwam_kv_carrier:
            kv_cache = prepare_dreamwam_kv_cache(
                result.captured_kv,
                layers=layers,
                token_count=args.capture_token_count,
            )
            finite_tensors = [
                tensor
                for item in kv_cache.values()
                for tensor in item.values()
            ]
            if not all(torch.isfinite(tensor.float()).all() for tensor in finite_tensors):
                raise RuntimeError(f"non-finite INT8 K/V for {row['id']}")
            first_k = kv_cache[layers[0]]["k"]
            cache_outputs.append((output, {
                "schema": DREAMWAM_KV_SCHEMA,
                "video_kv_cache": kv_cache,
                "layers": layers,
                "capture_token_count": int(first_k.shape[0]),
                "num_heads": int(first_k.shape[1]),
                "attn_head_dim": int(first_k.shape[2]),
                "capture_token_strategy": "adaptive_avg_pool1d_sequence_v1",
                "dreamwam_commit": DREAMWAM_COMMIT,
            }))
        if not args.dreamwam_kv_carrier or args.also_starwam_feature_cache:
            features = torch.stack(
                [result.captured_features[layer][0] for layer in feature_layers], dim=0
            )
            features = apply_capture_compatibility(features, args.capture_compatibility)
            features = pool_feature_tokens(features, args.capture_token_count).to(
                device="cpu", dtype=torch.bfloat16
            )
            if not torch.isfinite(features).all():
                raise RuntimeError(f"non-finite INT8 features for {row['id']}")
            feature_body = {
                "features": features,
                "layers": feature_layers,
                "capture_token_count": int(features.shape[1]),
                "capture_token_strategy": "starwam_adaptive_avg_pool1d_v1",
                "capture_compatibility": args.capture_compatibility,
            }
            cache_outputs.append((
                output if starwam_output is None else starwam_output,
                feature_body,
            ))
        common_body = {
                "episode": int(row["episode"]),
                "start": int(row["start"]),
                "suite": str(row["suite"]),
                "context_id": context_id,
                "context_width": context_width,
                "context_mode": context_mode,
                "timestep": float(args.timestep),
                "condition_video_timestep": float(args.condition_video_timestep),
                "action_horizon": int(args.action_horizon),
                "backbone": "H3Int8FeatureBackbone",
                "quantization": "int8_tensorwise_convrot",
                "checkpoint": str(args.h3_checkpoint.resolve()),
                "manifest_items": manifest_items,
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
        }
        wrote_window = False
        for destination, cache_body in cache_outputs:
            if destination.exists() and not args.overwrite:
                continue
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.tmp"
            )
            torch.save({**cache_body, **common_body}, temporary)
            os.replace(temporary, destination)
            wrote_window = True
            written_files += 1
        completed += 1
        written += int(wrote_window)
        if completed % args.progress_every == 0 or completed == len(rows):
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "written": written,
                        "written_files": written_files,
                        "total": len(rows),
                        "shard_index": args.shard_index,
                        "elapsed_seconds": round(elapsed, 2),
                        "seconds_per_window": round(elapsed / max(written, 1), 3),
                        "peak_allocated_gib": round(
                            torch.cuda.max_memory_allocated(device) / 2**30, 3
                        ),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
