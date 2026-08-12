#!/usr/bin/env python3
"""Cache local-recipe H3 observation tokens from the official BF16 backbone."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_h3_bf16_fsdp import (  # noqa: E402
    AUDIO_CHANNELS,
    AUDIO_LATENT_CHANNELS,
    replicated_non_block_modules,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--context-id", required=True)
    parser.add_argument("--output-subdir", default="h3_official_features_fixedctx")
    parser.add_argument("--layers", type=int, nargs="+", default=(9, 19, 29, 39, 49))
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--target-latent-frames", type=int, default=12)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--replicated-data-parallel",
        action="store_true",
        help=(
            "Load one complete BF16 H3 per rank and shard windows across GPUs. "
            "This is faster than FSDP inference when each GPU has enough memory."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.action_horizon, args.target_latent_frames, args.batch_size, args.progress_every) <= 0:
        raise ValueError("positive feature-cache arguments are required")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )
    from fastwam.models.h3wam import H3OfficialFeatureCapture
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    rows = [
        json.loads(line)
        for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("manifest is empty")
    cache_root = args.cache_root.resolve()
    output_root = cache_root / args.output_subdir
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    h3 = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model.resolve(),
        subfolder="transformer",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    h3.requires_grad_(False)
    if hasattr(h3, "disable_gradient_checkpointing"):
        h3.disable_gradient_checkpointing()
    if args.replicated_data_parallel:
        h3 = h3.to(device).eval()
        transformer_blocks = h3.transformer_blocks
        h3_config = h3.config
        work_rows = rows[rank::dist.get_world_size()]
    else:
        replicated_modules = replicated_non_block_modules(h3)
        for module in replicated_modules:
            module.to(device)
        h3 = FSDP(
            h3,
            auto_wrap_policy=ModuleWrapPolicy({MiniMaxH3TransformerBlock}),
            device_id=device,
            use_orig_params=True,
            limit_all_gathers=True,
            sync_module_states=False,
            ignored_modules=replicated_modules,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            ),
        ).eval()
        transformer_blocks = h3.module.transformer_blocks
        h3_config = h3.module.config
        work_rows = rows
    layers = tuple(sorted(set(args.layers)))
    patch_size = tuple(h3_config.patch_size)
    context_item = torch.load(args.context.resolve(), map_location="cpu", weights_only=False)
    context = context_item["context"].to(device=device, dtype=torch.float32)
    text_tags = context_item["token_tags"].long()

    first_example = torch.load(
        cache_root / "windows" / f"{work_rows[0]['id']}.pt",
        map_location="cpu",
        weights_only=False,
    )["first_frame_latents"]
    _, channels, _, latent_height, latent_width = first_example.shape
    layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=text_tags,
        num_latent_frames=args.target_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=args.action_horizon,
        patch_size=patch_size,
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
    ) = layout
    position_ids = position_ids.to(device)
    token_tags = token_tags.to(device)
    video_indices_device = video_indices.to(device)
    audio_indices_device = audio_indices.to(device)
    text_indices_device = text_indices.to(device)
    unique_timesteps, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices=video_indices,
        audio_indices=audio_indices,
        num_condition_video_rows=num_condition_video_rows,
        num_condition_audio_rows=num_condition_audio_rows,
        num_text_tokens=text_indices.numel(),
        video_timestep=args.timestep,
        audio_timestep=0.0,
        condition_video_timestep=1.0,
        condition_audio_timestep=1.0,
    )
    capture = H3OfficialFeatureCapture(
        transformer_blocks,
        layers,
        video_indices_device[:num_condition_video_rows],
    )
    started = time.perf_counter()
    completed = 0
    try:
        for offset in range(0, len(work_rows), args.batch_size):
            source_batch_rows = work_rows[offset : offset + args.batch_size]
            batch_rows = [
                row
                for row in source_batch_rows
                if args.overwrite
                or not (output_root / f"{row['id']}.pt").exists()
            ]
            if not batch_rows:
                completed += len(source_batch_rows)
                continue
            first = torch.cat(
                [
                    torch.load(
                        cache_root / "windows" / f"{row['id']}.pt",
                        map_location="cpu",
                        weights_only=False,
                    )["first_frame_latents"]
                    for row in batch_rows
                ]
            ).to(device=device, dtype=torch.float32)
            batch_size = first.shape[0]
            target = torch.zeros(
                (batch_size, channels, args.target_latent_frames, latent_height, latent_width),
                device=device,
                dtype=torch.float32,
            )
            row_dim = channels * patch_size[0] * patch_size[1] * patch_size[2]
            first_rows = patchify_video_latents(first, patch_size).reshape(
                batch_size, -1, row_dim
            )
            target_rows = patchify_video_latents(target, patch_size).reshape(
                batch_size, -1, row_dim
            )
            video_rows = torch.cat((first_rows, target_rows), dim=1)
            audio_rows = torch.zeros(
                (batch_size, args.action_horizon * AUDIO_CHANNELS, AUDIO_LATENT_CHANNELS),
                device=device,
                dtype=torch.float32,
            )
            capture.clear()
            with torch.inference_mode():
                h3(
                    hidden_states=video_rows,
                    audio_hidden_states=audio_rows,
                    encoder_hidden_states=context.expand(batch_size, -1, -1),
                    timestep=unique_timesteps.to(device),
                    timestep_indices=timestep_indices.to(device),
                    token_tags=token_tags,
                    position_ids=position_ids,
                    video_indices=video_indices_device,
                    audio_indices=audio_indices_device,
                    text_indices=text_indices_device,
                    return_dict=True,
                )
                features = capture.stacked().to(device="cpu", dtype=torch.bfloat16)
            if args.replicated_data_parallel or rank == 0:
                if features.shape[:3] != (batch_size, len(layers), num_condition_video_rows):
                    raise RuntimeError(f"unexpected feature shape {tuple(features.shape)}")
                for index, row in enumerate(batch_rows):
                    path = output_root / f"{row['id']}.pt"
                    if path.exists() and not args.overwrite:
                        continue
                    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
                    torch.save(
                        {
                            # Detach this sample from the batch storage. Without
                            # clone(), torch.save serializes the full backing
                            # batch once for every per-window artifact.
                            "features": features[index].clone(),
                            "layers": layers,
                            "episode": int(row["episode"]),
                            "start": int(row["start"]),
                            "context_id": args.context_id,
                            "timestep": float(args.timestep),
                            "action_horizon": args.action_horizon,
                            "backbone": "diffusers.MiniMaxH3Transformer3DModel",
                        },
                        temporary,
                    )
                    os.replace(temporary, path)
            completed += len(source_batch_rows)
            if rank == 0 and (
                completed % args.progress_every < len(source_batch_rows)
                or completed == len(work_rows)
            ):
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "complete": completed,
                            "total": len(work_rows),
                            "rank": rank,
                            "elapsed_seconds": round(elapsed, 2),
                            "seconds_per_window": round(elapsed / completed, 4),
                            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                        }
                    ),
                    flush=True,
                )
    finally:
        capture.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
