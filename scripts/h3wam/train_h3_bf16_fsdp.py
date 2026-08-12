#!/usr/bin/env python3
"""Run the first native-BF16 MiniMax-H3 WAM training smoke with FSDP.

This intentionally uses the official Diffusers H3 implementation rather than
ComfyUI.  The first milestone is deliberately narrow: load the released BF16
``transformer`` partition on every rank, shard it with FSDP, unfreeze only the
last N transformer blocks, and prove a finite video-flow backward/update on a
real cached LIBERO window.

Launch example (one node, eight GPUs)::

    torchrun --standalone --nproc-per-node=8 \
      scripts/h3wam/train_h3_bf16_fsdp.py \
      --model /home/h3wam_finetune/models/MiniMax-H3 \
      --data-root /home/h3wam_finetune/data/v0 \
      --output-dir /home/h3wam_finetune/outputs/v0 \
      --steps 1 --last-blocks 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


KEYFRAME_TIMESTEP = 0.999
VIDEO_FLOW_SHIFT = 12.0
AUDIO_CHANNELS = 2
AUDIO_LATENT_CHANNELS = 32
AUDIO_LATENTS_PER_SECOND = 40
H3_FPS = 24.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--last-blocks", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--fp32-master-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep trainable tail shards/Adam states FP32 and cast them to BF16 only for forward.",
    )
    parser.add_argument(
        "--verify-parameter-update",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSONL rows with cached sample IDs. Enables deterministic multi-window training.",
    )
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument("--validation-batches-per-rank", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save same-world-size rank-sharded resume checkpoints; 0 disables.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint directory containing rankXXXXX.pt files.",
    )
    parser.add_argument(
        "--sample-id",
        help="Cached sample basename without .pt; default uses the first common file.",
    )
    parser.add_argument(
        "--limit-all-gathers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def shifted_video_timestep(
    *, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    """Sample H3's shifted flow time, where 0 is noise and 1 is clean."""
    base_sigma = torch.rand((), generator=generator, device=device)
    sigma = VIDEO_FLOW_SHIFT * base_sigma / (
        1.0 + (VIDEO_FLOW_SHIFT - 1.0) * base_sigma
    )
    return 1.0 - sigma


def audio_latent_count(pixel_frames: int) -> int:
    return max(1, round(pixel_frames / H3_FPS * AUDIO_LATENTS_PER_SECOND))


def resolve_sample(
    data_root: Path,
    sample_id: str | None,
    context_id: str | None = None,
) -> tuple[Path, Path]:
    windows = data_root / "windows"
    contexts = data_root / "contexts"
    if sample_id:
        window = windows / f"{sample_id}.pt"
        context = contexts / f"{context_id or sample_id}.pt"
        if not window.is_file() or not context.is_file():
            raise FileNotFoundError(
                f"missing cached pair for sample={sample_id!r}, "
                f"context={context_id or sample_id!r}"
            )
        return window, context

    common = sorted({path.name for path in windows.glob("*.pt")} & {path.name for path in contexts.glob("*.pt")})
    if not common:
        raise FileNotFoundError(
            f"no matching .pt files under {windows} and {contexts}"
        )
    return windows / common[0], contexts / common[0]


def load_training_rows(
    data_root: Path, manifest: Path | None, sample_id: str | None
) -> list[dict]:
    if manifest is None:
        window, _ = resolve_sample(data_root, sample_id)
        return [{"id": window.stem}]
    rows = [
        json.loads(line)
        for line in manifest.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any("id" not in row for row in rows):
        raise ValueError("training manifest must contain at least one row with an id")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("training manifest contains duplicate sample IDs")
    for row in rows:
        resolve_sample(
            data_root,
            str(row["id"]),
            str(row.get("context_id", row["id"])),
        )
    return rows


def save_rank_checkpoint(
    *,
    output_dir: Path,
    step: int,
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints" / f"step{step:06d}"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    trainable_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    artifact = {
        "format": "h3wam_fsdp_same_world_size_v2",
        "step": step,
        "rank": rank,
        "world_size": world_size,
        "trainable_storage_dtypes": sorted(
            {str(parameter.dtype) for parameter in model.parameters() if parameter.requires_grad}
        ),
        "trainable_state": trainable_state,
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_generator_state": generator.get_state().cpu(),
    }
    path = checkpoint_dir / f"rank{rank:05d}.pt"
    partial = path.with_suffix(".pt.partial")
    torch.save(artifact, partial)
    os.replace(partial, path)
    dist.barrier()
    if rank == 0:
        manifest = {
            "format": artifact["format"],
            "step": step,
            "world_size": world_size,
            "rank_files": [f"rank{index:05d}.pt" for index in range(world_size)],
        }
        (checkpoint_dir / "checkpoint.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    dist.barrier()
    return checkpoint_dir


def load_rank_checkpoint(
    *,
    checkpoint_dir: Path,
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    device: torch.device,
) -> int:
    artifact = torch.load(
        checkpoint_dir.resolve() / f"rank{rank:05d}.pt",
        map_location=device,
        weights_only=False,
    )
    if artifact.get("format") != "h3wam_fsdp_same_world_size_v2":
        raise ValueError("unsupported H3-WAM checkpoint format")
    if int(artifact["world_size"]) != world_size or int(artifact["rank"]) != rank:
        raise ValueError("checkpoint rank/world size does not match this launch")
    named = dict(model.named_parameters())
    expected = {name for name, parameter in named.items() if parameter.requires_grad}
    expected_dtypes = sorted({str(named[name].dtype) for name in expected})
    if artifact.get("trainable_storage_dtypes") != expected_dtypes:
        raise ValueError("checkpoint trainable storage dtype does not match model")
    state = artifact["trainable_state"]
    if set(state) != expected:
        raise ValueError("checkpoint trainable parameter names do not match model")
    with torch.no_grad():
        for name, value in state.items():
            if named[name].shape != value.shape:
                raise ValueError(f"checkpoint shape mismatch for {name}")
            named[name].copy_(value.to(device=device, dtype=named[name].dtype))
    optimizer.load_state_dict(artifact["optimizer"])
    torch.set_rng_state(artifact["torch_rng_state"].cpu())
    generator.set_state(artifact["cuda_generator_state"].cpu())
    return int(artifact["step"])


def validate_cached_sample(window: dict, conditioning: dict) -> None:
    video = window["video_latents"]
    first = window["first_frame_latents"]
    context = conditioning["context"]
    tags = conditioning["token_tags"]
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError(f"expected video [1,24,T,H,W], got {tuple(video.shape)}")
    expected_first = (video.shape[0], video.shape[1], 1, video.shape[3], video.shape[4])
    if tuple(first.shape) != expected_first:
        raise ValueError(
            f"first-frame latent shape {tuple(first.shape)} != {expected_first}"
        )
    if context.ndim != 3 or tuple(context.shape[:1]) != (1,) or context.shape[-1] != 5120:
        raise ValueError(f"official H3 requires raw [1,L,5120] context, got {tuple(context.shape)}")
    if tags.ndim != 1 or tags.numel() != context.shape[1]:
        raise ValueError(
            f"token tags {tuple(tags.shape)} do not match context length {context.shape[1]}"
        )
    if not set(tags.unique().tolist()).issubset({0, 1}):
        raise ValueError("cached prompt token tags must contain only H3 video/text tags 0/1")


def set_trainable_tail(model: torch.nn.Module, last_blocks: int) -> list[str]:
    model.requires_grad_(False)
    blocks = model.transformer_blocks
    if not 1 <= last_blocks <= len(blocks):
        raise ValueError(f"last-blocks must be in [1,{len(blocks)}], got {last_blocks}")
    first = len(blocks) - last_blocks
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if name.startswith("transformer_blocks."):
            parts = name.split(".")
            if len(parts) > 1 and parts[1].isdigit() and int(parts[1]) >= first:
                parameter.requires_grad_(True)
                names.append(name)
    if not names:
        raise RuntimeError("no trainable H3 tail parameters were selected")
    return names


def replicated_non_block_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return the small mixed-dtype H3 modules kept replicated on every rank.

    Classic FSDP requires every flat parameter handle to have one dtype.  The
    released H3 checkpoint intentionally mixes FP32 input/output/time modules
    with a BF16 context/refiner/block stack, so flattening all non-block root
    parameters together is invalid.  These frozen modules are small enough to
    replicate; the 50 large, uniform-BF16 transformer blocks remain sharded.
    """
    modules = [
        child
        for name, child in model.named_children()
        if name != "transformer_blocks"
    ]
    if not modules:
        raise RuntimeError("H3 has no non-block modules to replicate")
    return modules


def distributed_mean(value: torch.Tensor) -> float:
    reduced = value.detach().float().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return float((reduced / dist.get_world_size()).item())


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError("learning rate and gradient clip must be positive")
    if (
        args.log_every <= 0
        or args.checkpoint_every < 0
        or args.validation_every < 0
        or args.validation_batches_per_rank <= 0
    ):
        raise ValueError("log/validation/checkpoint intervals are invalid")
    if (args.validation_manifest is None) != (args.validation_every == 0):
        raise ValueError(
            "validation-manifest and a positive validation-every must be set together"
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 2:
        raise RuntimeError("this smoke is designed for torchrun with at least two GPUs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    # Imports stay here so --help and static validation remain useful before the
    # H3 Diffusers branch is installed.
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3TransformerBlock,
    )
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    output_dir = args.output_dir.resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    data_root = args.data_root.resolve()
    training_rows = load_training_rows(data_root, args.manifest, args.sample_id)
    random.Random(args.seed).shuffle(training_rows)
    validation_rows = (
        load_training_rows(data_root, args.validation_manifest, None)
        if args.validation_manifest is not None
        else []
    )

    load_started = time.perf_counter()
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model.resolve(),
        subfolder="transformer",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    trainable_names = set_trainable_tail(model, args.last_blocks)
    if args.fp32_master_weights:
        for block in model.transformer_blocks[-args.last_blocks :]:
            block.to(torch.float32)
    full_trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    model.enable_gradient_checkpointing()
    model.train()
    load_seconds = time.perf_counter() - load_started

    replicated_modules = replicated_non_block_modules(model)
    for module in replicated_modules:
        module.to(device)
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({MiniMaxH3TransformerBlock}),
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=args.limit_all_gathers,
        sync_module_states=False,
        ignored_modules=replicated_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    patch_size = tuple(model.module.config.patch_size)

    def prepare_row(row: dict) -> dict:
        window_path, context_path = resolve_sample(data_root, str(row["id"]))
        window = torch.load(window_path, map_location="cpu", weights_only=False)
        conditioning = torch.load(context_path, map_location="cpu", weights_only=False)
        validate_cached_sample(window, conditioning)
        clean = window["video_latents"].to(device=device, dtype=torch.float32)
        first = window["first_frame_latents"].to(device=device, dtype=torch.float32)
        context = conditioning["context"].to(device=device, dtype=torch.float32)
        text_tags = conditioning["token_tags"].to(dtype=torch.long)
        _, _, num_latent_frames, latent_height, latent_width = clean.shape
        pixel_frames = int(window.get("h3_frame_count", num_latent_frames))
        num_audio_latents = audio_latent_count(pixel_frames)
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=text_tags,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
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
        return {
            "id": str(row["id"]),
            "window_path": window_path,
            "context_path": context_path,
            "clean": clean,
            "first": first,
            "context": context,
            "num_audio_latents": num_audio_latents,
            "position_ids": position_ids.to(device),
            "token_tags": token_tags.to(device),
            "video_indices": video_indices.to(device),
            "audio_indices": audio_indices.to(device),
            "text_indices": text_indices.to(device),
            "num_condition_video_rows": num_condition_video_rows,
            "num_condition_audio_rows": num_condition_audio_rows,
        }

    def flow_loss(batch: dict, rng: torch.Generator) -> torch.Tensor:
        clean = batch["clean"]
        first = batch["first"]
        timestep = shifted_video_timestep(generator=rng, device=device)
        noise = torch.randn(
            clean.shape, generator=rng, device=device, dtype=torch.float32
        )
        noisy = timestep * clean + (1.0 - timestep) * noise
        target_rows = patchify_video_latents(clean - noise, patch_size)[None]
        noisy_rows = patchify_video_latents(noisy, patch_size)
        condition_noise = torch.randn(
            first.shape, generator=rng, device=device, dtype=torch.float32
        )
        noised_first = (
            KEYFRAME_TIMESTEP * first
            + (1.0 - KEYFRAME_TIMESTEP) * condition_noise
        )
        condition_rows = patchify_video_latents(noised_first, patch_size)
        video_rows = torch.cat([condition_rows, noisy_rows], dim=0)[None]
        audio_rows = torch.randn(
            (
                1,
                batch["num_audio_latents"] * AUDIO_CHANNELS,
                AUDIO_LATENT_CHANNELS,
            ),
            generator=rng,
            device=device,
            dtype=torch.float32,
        )
        unique_timesteps, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices=batch["video_indices"].cpu(),
            audio_indices=batch["audio_indices"].cpu(),
            num_condition_video_rows=batch["num_condition_video_rows"],
            num_condition_audio_rows=batch["num_condition_audio_rows"],
            num_text_tokens=batch["text_indices"].numel(),
            video_timestep=float(timestep),
            audio_timestep=0.0,
            condition_video_timestep=max(float(timestep), KEYFRAME_TIMESTEP),
            condition_audio_timestep=1.0,
        )
        output = model(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=batch["context"],
            timestep=unique_timesteps.to(device),
            timestep_indices=timestep_indices.to(device),
            token_tags=batch["token_tags"],
            position_ids=batch["position_ids"],
            video_indices=batch["video_indices"],
            audio_indices=batch["audio_indices"],
            text_indices=batch["text_indices"],
            return_dict=True,
        )
        predicted = output.sample[:, batch["num_condition_video_rows"] :]
        if predicted.shape != target_rows.shape:
            raise RuntimeError(
                f"prediction {tuple(predicted.shape)} != target {tuple(target_rows.shape)}"
            )
        loss = F.mse_loss(predicted.float(), target_rows.float())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"rank {rank} produced non-finite loss {loss.item()}")
        return loss

    @torch.no_grad()
    def validation_loss() -> float:
        if not validation_rows:
            raise RuntimeError("validation requested without validation rows")
        model.eval()
        rng = torch.Generator(device=device).manual_seed(args.seed + 100_000 + rank)
        total = torch.zeros((), device=device, dtype=torch.float32)
        for index in range(args.validation_batches_per_rank):
            row_index = (index * world_size + rank) % len(validation_rows)
            total += flow_loss(prepare_row(validation_rows[row_index]), rng).float()
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        model.train()
        return float(
            (total / (args.validation_batches_per_rank * world_size)).item()
        )

    first_batch = prepare_row(training_rows[rank % len(training_rows)])
    start_step = 0
    if args.resume is not None:
        start_step = load_rank_checkpoint(
            checkpoint_dir=args.resume,
            rank=rank,
            world_size=world_size,
            model=model,
            optimizer=optimizer,
            generator=generator,
            device=device,
        )
        if start_step >= args.steps:
            raise ValueError(f"resume step {start_step} is not below requested steps {args.steps}")

    metadata = {
        "host": socket.gethostname(),
        "world_size": world_size,
        "model": str(args.model.resolve()),
        "data_root": str(data_root),
        "manifest": str(args.manifest.resolve()) if args.manifest else None,
        "training_windows": len(training_rows),
        "validation_manifest": (
            str(args.validation_manifest.resolve()) if args.validation_manifest else None
        ),
        "validation_windows": len(validation_rows),
        "first_window": str(first_batch["window_path"]),
        "first_context": str(first_batch["context_path"]),
        "last_blocks": args.last_blocks,
        "full_trainable_parameter_count": full_trainable_parameter_count,
        "fp32_master_weights": args.fp32_master_weights,
        "trainable_storage_dtypes": sorted({str(p.dtype) for p in trainable}),
        "rank0_trainable_shard_parameter_count": sum(p.numel() for p in trainable),
        "replicated_frozen_parameter_count": sum(
            parameter.numel()
            for module in replicated_modules
            for parameter in module.parameters()
        ),
        "trainable_tensor_count": len(trainable_names),
        "video_shape": list(first_batch["clean"].shape),
        "context_shape": list(first_batch["context"].shape),
        "num_audio_latents": first_batch["num_audio_latents"],
        "sequence_length": int(first_batch["position_ids"].shape[0]),
        "num_condition_video_rows": first_batch["num_condition_video_rows"],
        "start_step": start_step,
        "load_seconds": load_seconds,
        "torch": torch.__version__,
    }
    if rank == 0:
        print(json.dumps({"event": "ready", **metadata}, sort_keys=True), flush=True)

    del first_batch
    last_validation: float | None = None
    if validation_rows:
        last_validation = validation_loss()
        if rank == 0:
            print(
                json.dumps(
                    {"event": "validation", "step": start_step, "loss": last_validation},
                    sort_keys=True,
                ),
                flush=True,
            )
    initial_validation = last_validation
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    last_checkpoint: str | None = None
    update_probe_parameter = next(
        parameter
        for parameter in trainable
        if parameter.requires_grad and parameter.numel() > 0
    )
    for step in range(start_step + 1, args.steps + 1):
        row_index = ((step - 1) * world_size + rank) % len(training_rows)
        batch = prepare_row(training_rows[row_index])
        optimizer.zero_grad(set_to_none=True)
        update_probe_before = None
        if args.verify_parameter_update and step == start_step + 1:
            update_probe_before = update_probe_parameter.detach().flatten()[:4096].float().clone()
        loss = flow_loss(batch, generator)
        loss.backward()
        grad_norm = model.clip_grad_norm_(args.gradient_clip)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(
                f"rank {rank} produced non-finite gradient norm {grad_norm.item()}"
            )
        optimizer.step()
        parameter_update_max_abs: float | None = None
        if update_probe_before is not None:
            update_delta = (
                update_probe_parameter.detach().flatten()[:4096].float()
                - update_probe_before
            ).abs().max()
            dist.all_reduce(update_delta, op=dist.ReduceOp.MAX)
            parameter_update_max_abs = float(update_delta.item())
            if parameter_update_max_abs == 0.0:
                raise RuntimeError("optimizer step produced a zero parameter-update probe")
        torch.cuda.synchronize(device)

        mean_loss = distributed_mean(loss)
        mean_grad_norm = distributed_mean(grad_norm)
        if validation_rows and (
            step % args.validation_every == 0 or step == args.steps
        ):
            last_validation = validation_loss()
            if rank == 0:
                print(
                    json.dumps(
                        {"event": "validation", "step": step, "loss": last_validation},
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if rank == 0 and (step % args.log_every == 0 or step == args.steps):
            print(
                json.dumps(
                    {
                        "event": "step",
                        "step": step,
                        "loss": mean_loss,
                        "gradient_norm": mean_grad_norm,
                        "validation_loss": last_validation,
                        "parameter_update_max_abs": parameter_update_max_abs,
                        "rank0_sample_id": batch["id"],
                        "elapsed_seconds": time.perf_counter() - started,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.checkpoint_every and (
            step % args.checkpoint_every == 0 or step == args.steps
        ):
            checkpoint_path = save_rank_checkpoint(
                output_dir=output_dir,
                step=step,
                rank=rank,
                world_size=world_size,
                model=model,
                optimizer=optimizer,
                generator=generator,
            )
            last_checkpoint = str(checkpoint_path)
            if rank == 0:
                print(
                    json.dumps(
                        {"event": "checkpoint", "step": step, "path": last_checkpoint},
                        sort_keys=True,
                    ),
                    flush=True,
                )

    dist.barrier()
    if rank == 0:
        report = {
            "event": "complete",
            "steps": args.steps,
            "last_checkpoint": last_checkpoint,
            "initial_validation_loss": initial_validation,
            "last_validation_loss": last_validation,
            **metadata,
        }
        report_path = output_dir / "smoke_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"event": "report", "path": str(report_path)}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
