#!/usr/bin/env python3
"""Checkpointed C71 Light-WAM state-fusion training over frozen online INT8 H3.

This is a backbone port, not an official Light-WAM reproduction.  It preserves
the audited C58 dense LIBERO/action-normalization boundary while replacing the
flow ActionDiT with the byte-pinned Light-WAM direct-regression state-fusion
expert.  H3 is executed online and remains frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam.c58_online_training import (  # noqa: E402
    C58OnlineFrozenH3Dataset,
    C58OnlineFrozenH3Provider,
    attach_online_h3_kv,
    collate_c58_online,
    move_c58_online_batch,
)
from fastwam.models.h3wam.lightwam_state_fusion import (  # noqa: E402
    LIGHTWAM_COMMIT,
    LIGHTWAM_H3_CARRIER_LAYERS,
    LIGHTWAM_STATE_FUSION_SHA256,
    H3LightWAMStateFusionPolicy,
)


FORMAT = "h3wam-c71-lightwam-online-train-v1"
CHECKPOINT_SCHEMA = 1
EXPECTED_H3_SHA256 = (
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
)
CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "completed_steps",
        "model",
        "optimizer",
        "lr_scheduler",
        "contract",
        "probe_prediction",
        "probe_sample_ids",
        "rng_states",
        "data_state",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--expected-source-freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--restore-check-only", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--probe-sample-offset", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--scheduler-horizon", type=int, default=10000)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=32)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: dict[str, Any], output: Path) -> int:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    return output.stat().st_size


def atomic_json(payload: dict[str, Any], output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def masked_direct_action_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_is_pad: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share [B,T,A] shape")
    if action_is_pad.shape != prediction.shape[:2]:
        raise ValueError("action_is_pad must be [B,T]")
    valid = (~action_is_pad.bool()).unsqueeze(-1).expand_as(prediction)
    if int(valid.sum()) == 0:
        raise ValueError("direct action loss requires a non-padding target")
    return prediction.float().sub(target.float()).square().masked_select(valid).mean()


def module_grad_norm(module: nn.Module) -> float:
    squared = sum(
        float(parameter.grad.detach().float().square().sum().cpu())
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(squared)


def distributed_setup() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("C71 online training requires CUDA")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def infinite_batches(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def scheduler_factor(
    step: int,
    *,
    warmup_steps: int,
    horizon: int,
    minimum_ratio: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(1, step + 1) / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, horizon - warmup_steps)))
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def validate_checkpoint(payload: dict[str, Any], contract: dict[str, Any]) -> None:
    if set(payload) != CHECKPOINT_KEYS:
        raise ValueError("C71 checkpoint key contract mismatch")
    if payload["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("C71 checkpoint schema mismatch")
    if payload["contract"] != contract:
        raise ValueError("C71 checkpoint training contract mismatch")
    if int(payload["completed_steps"]) < 0:
        raise ValueError("C71 checkpoint completed_steps is invalid")


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def main() -> None:
    args = parse_args()
    if args.steps < 0 or (args.steps == 0 and not args.restore_check_only):
        raise ValueError("steps must be positive except for restore-check-only")
    if args.restore_check_only and args.load_checkpoint is None:
        raise ValueError("restore-check-only requires --load-checkpoint")
    if args.limit <= 0:
        raise ValueError("an explicit positive data slice --limit is required")
    if args.scheduler_horizon <= 0 or args.warmup_steps < 0:
        raise ValueError("scheduler horizon/warmup are invalid")
    if not 0.0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError("min learning rate must be within [0, learning_rate]")

    resolved = {
        name: path.resolve()
        for name, path in {
            "manifest": args.manifest,
            "source_manifest": args.source_manifest,
            "cache_root": args.cache_root,
            "h3_checkpoint": args.h3_checkpoint,
            "source_freeze": args.source_freeze,
        }.items()
    }
    for name in ("manifest", "source_manifest", "h3_checkpoint", "source_freeze"):
        if not resolved[name].is_file():
            raise FileNotFoundError(resolved[name])
    if not resolved["cache_root"].is_dir():
        raise FileNotFoundError(resolved["cache_root"])
    h3_sha = sha256_file(resolved["h3_checkpoint"])
    freeze_sha = sha256_file(resolved["source_freeze"])
    if h3_sha != EXPECTED_H3_SHA256:
        raise ValueError(f"C71 H3 checkpoint SHA mismatch: {h3_sha}")
    if freeze_sha != args.expected_source_freeze_sha256:
        raise ValueError("C71 immutable source manifest SHA mismatch")

    rank, world_size, device = distributed_setup()
    dtype = torch.bfloat16
    seed = args.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    dataset = C58OnlineFrozenH3Dataset(
        resolved["manifest"],
        resolved["source_manifest"],
        resolved["cache_root"],
        resolved["h3_checkpoint"],
        action_horizon=args.action_horizon,
        sample_offset=args.sample_offset,
        limit=args.limit,
    )
    expected_rank_samples = args.steps if not args.restore_check_only else 0
    if not args.restore_check_only and len(dataset) < world_size * args.steps:
        raise ValueError("C71 explicit slice is too short for unique per-rank samples")
    sampler = (
        DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=False)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_c58_online,
    )
    probe_dataset = C58OnlineFrozenH3Dataset(
        resolved["manifest"],
        resolved["source_manifest"],
        resolved["cache_root"],
        resolved["h3_checkpoint"],
        action_horizon=args.action_horizon,
        sample_offset=args.probe_sample_offset,
        limit=1,
    )
    probe_cpu = collate_c58_online([probe_dataset[0]])

    provider = C58OnlineFrozenH3Provider(
        resolved["h3_checkpoint"], layers=LIGHTWAM_H3_CARRIER_LAYERS
    ).to(device=device).eval()
    provider.requires_grad_(False)
    model = H3LightWAMStateFusionPolicy(enabled=True).to(device=device, dtype=dtype)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    minimum_ratio = args.min_learning_rate / args.learning_rate
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: scheduler_factor(
            step,
            warmup_steps=args.warmup_steps,
            horizon=args.scheduler_horizon,
            minimum_ratio=minimum_ratio,
        ),
    )
    contract = {
        "classification": "backbone_port",
        "parent": "C58 frozen online INT8 H3 dense action boundary",
        "unique_variable": "byte-pinned Light-WAM three-state direct-regression expert",
        "lightwam_commit": LIGHTWAM_COMMIT,
        "lightwam_state_fusion_sha256": LIGHTWAM_STATE_FUSION_SHA256,
        "h3_checkpoint_sha256": h3_sha,
        "source_freeze_sha256": freeze_sha,
        "h3_trainable": False,
        "h3_layers": list(LIGHTWAM_H3_CARRIER_LAYERS),
        "feature": "value_state",
        "capture_tokens": 32,
        "action_horizon": args.action_horizon,
        "action_dim": 7,
        "proprio_dim": 8,
        "objective": "uniform_full_horizon_masked_normalized_action_mse",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW_beta0.9_0.95",
        "warmup_steps": args.warmup_steps,
        "scheduler_horizon": args.scheduler_horizon,
        "min_learning_rate": args.min_learning_rate,
        "seed": args.seed,
        "manifest_sha256": dataset.manifest_sha256,
        "source_manifest_sha256": dataset.source_manifest_sha256,
        "stats_sha256": dataset.stats_sha256,
        "trainable_parameters": trainable_parameters,
    }

    probe = move_c58_online_batch(probe_cpu, device, dtype)
    probe = attach_online_h3_kv(probe, provider)
    completed_steps = 0
    loaded_payload: dict[str, Any] | None = None
    restore_max_abs: float | None = None
    if args.load_checkpoint is not None:
        loaded_payload = torch.load(args.load_checkpoint.resolve(), map_location="cpu", weights_only=False)
        validate_checkpoint(loaded_payload, contract)
        model.load_state_dict(loaded_payload["model"], strict=True)
        optimizer.load_state_dict(loaded_payload["optimizer"])
        lr_scheduler.load_state_dict(loaded_payload["lr_scheduler"])
        completed_steps = int(loaded_payload["completed_steps"])
        if not args.restore_check_only:
            current_ids = {str(row["id"]) for row in dataset.rows}
            overlap = current_ids & set(loaded_payload["data_state"]["sample_ids"])
            if overlap:
                raise ValueError(f"C71 stage reuses {len(overlap)} previously consumed rows")
        model.eval()
        with torch.no_grad():
            restored = model(
                torch.zeros_like(probe["actions"]),
                torch.zeros((1,), device=device, dtype=dtype),
                text_context=probe["text_context"],
                text_mask=probe["text_mask"],
                proprio=probe["proprio"],
                video_kv_cache=probe["video_kv_cache"],
            ).float()
        if loaded_payload["probe_sample_ids"] != probe_cpu["sample_ids"]:
            raise ValueError("C71 restore probe identity mismatch")
        expected = loaded_payload["probe_prediction"].to(restored)
        restore_max_abs = float((restored - expected).abs().max())
        if restore_max_abs != 0.0:
            raise RuntimeError(f"C71 strict restore mismatch: {restore_max_abs}")

    unwrapped = model
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
        unwrapped = model.module

    history: list[dict[str, Any]] = []
    consumed_ids: list[str] = []
    if not args.restore_check_only:
        model.train()
        iterator = iter(infinite_batches(loader))
        tracked = unwrapped.state_fusion_action_expert.output[-1].weight
        tracked_before = tracked.detach().float().clone()
        for local_step in range(1, args.steps + 1):
            batch_cpu = next(iterator)
            batch = move_c58_online_batch(batch_cpu, device, dtype)
            batch = attach_online_h3_kv(batch, provider)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                torch.zeros_like(batch["actions"]),
                torch.zeros((1,), device=device, dtype=dtype),
                text_context=batch["text_context"],
                text_mask=batch["text_mask"],
                proprio=batch["proprio"],
                video_kv_cache=batch["video_kv_cache"],
            )
            loss = masked_direct_action_mse(prediction, batch["actions"], batch["action_is_pad"])
            if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
                raise RuntimeError("C71 produced non-finite training output")
            loss.backward()
            expert = unwrapped.state_fusion_action_expert
            gradients = {
                "query_poolers": module_grad_norm(expert.layer_poolers),
                "layer_compressors": module_grad_norm(expert.layer_compressors),
                "fusion_trunk": module_grad_norm(expert.fused_proj) + module_grad_norm(expert.trunk),
                "step_position": module_grad_norm(expert.step_pos_proj),
                "proprio_encoder": module_grad_norm(unwrapped.proprio_encoder),
                "output_head": module_grad_norm(expert.output),
            }
            if not all(math.isfinite(value) and value > 0.0 for value in gradients.values()):
                raise RuntimeError(f"C71 invalid declared gradient path: {gradients}")
            clipped = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True))
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
            update = float((tracked.detach().float() - tracked_before).abs().max())
            if not math.isfinite(update) or update <= 0.0:
                raise RuntimeError("C71 optimizer did not update output head")
            tracked_before = tracked.detach().float().clone()
            consumed_ids.extend(batch["sample_ids"])
            record = {
                "step": completed_steps + local_step,
                "loss": float(loss.detach()),
                "prediction_std": float(prediction.detach().float().std()),
                "gradient_norms": gradients,
                "clipped_gradient_norm": clipped,
                "output_update_max_abs": update,
                "learning_rate": learning_rate,
                "sample_ids": list(batch["sample_ids"]),
            }
            history.append(record)
            if rank == 0:
                print(json.dumps(record, sort_keys=True), flush=True)
        completed_steps += args.steps

    unwrapped.eval()
    with torch.no_grad():
        probe_prediction = unwrapped(
            torch.zeros_like(probe["actions"]),
            torch.zeros((1,), device=device, dtype=dtype),
            text_context=probe["text_context"],
            text_mask=probe["text_mask"],
            proprio=probe["proprio"],
            video_kv_cache=probe["video_kv_cache"],
        ).float()
    if not torch.isfinite(probe_prediction).all() or float(probe_prediction.std()) <= 0.0:
        raise RuntimeError("C71 checkpoint probe is non-finite or constant")

    local_state = {"rng": capture_rng_state(device), "ids": consumed_ids}
    if dist.is_initialized():
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_state)
        complete = [item for item in gathered if item is not None]
        if len(complete) != world_size:
            raise RuntimeError("C71 failed to gather rank state")
    else:
        complete = [local_state]
    if not args.restore_check_only and any(len(item["ids"]) != expected_rank_samples for item in complete):
        raise RuntimeError("C71 rank sample accounting mismatch")

    checkpoint_bytes: int | None = None
    if rank == 0 and args.save_checkpoint is not None and not args.restore_check_only:
        historical = [] if loaded_payload is None else list(loaded_payload["data_state"]["sample_ids"])
        stage_ids = [sample_id for item in complete for sample_id in item["ids"]]
        if len(set(stage_ids)) != len(stage_ids) or set(stage_ids) & set(historical):
            raise RuntimeError("C71 sample ledger contains duplicates")
        data_state = {
            "resume_mode": "explicit_disjoint_stage_slice_v1",
            "sample_offset": args.sample_offset,
            "limit": args.limit,
            "selected_windows": len(dataset),
            "steps_in_invocation": args.steps,
            "sample_ids": historical + stage_ids,
        }
        checkpoint_bytes = atomic_torch_save(
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "completed_steps": completed_steps,
                "model": unwrapped.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "contract": contract,
                "probe_prediction": probe_prediction.cpu(),
                "probe_sample_ids": list(probe_cpu["sample_ids"]),
                "rng_states": [item["rng"] for item in complete],
                "data_state": data_state,
            },
            args.save_checkpoint,
        )
    if dist.is_initialized():
        dist.barrier()

    rank_audit = {
        "rank": rank,
        "sample_ids": consumed_ids,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    if dist.is_initialized():
        rank_audits: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(rank_audits, rank_audit)
    else:
        rank_audits = [rank_audit]

    if rank == 0:
        elapsed = time.perf_counter() - started
        training_samples = 0 if args.restore_check_only else world_size * args.steps
        report = {
            "format": FORMAT,
            "status": "PASS_C71_STRICT_RESTORE" if args.restore_check_only else "PASS_C71_CHECKPOINTED_TRAIN_STAGE",
            "permission": "GO_CANARY" if completed_steps <= 10 else "GO_LONG",
            "effect_status": "NOT_EVIDENCE_READY",
            "resolved_argv": sys.argv,
            "world_size": world_size,
            "global_batch_size": world_size,
            "completed_steps": completed_steps,
            "steps_this_invocation": 0 if args.restore_check_only else args.steps,
            "training_samples": training_samples,
            "cumulative_training_samples": completed_steps * world_size,
            "unique_train_windows": dataset.manifest_items,
            "effective_epochs_this_invocation": training_samples / dataset.manifest_items,
            "cumulative_effective_epochs": completed_steps * world_size / dataset.manifest_items,
            "contract": contract,
            "restore_probe_max_abs": restore_max_abs,
            "probe_prediction_mean": float(probe_prediction.mean()),
            "probe_prediction_std": float(probe_prediction.std()),
            "saved_checkpoint": None if args.save_checkpoint is None else str(args.save_checkpoint.resolve()),
            "checkpoint_file_size_bytes": checkpoint_bytes,
            "history": history,
            "per_rank_runtime": rank_audits,
            "elapsed_seconds": elapsed,
            "seconds_per_step": None if not history else elapsed / len(history),
            "claim_boundary": "Mechanical/diagnostic training only; loss and restore do not establish LIBERO improvement.",
        }
        atomic_json(report, args.output)
        print(json.dumps({key: report[key] for key in ("status", "completed_steps", "training_samples", "effective_epochs_this_invocation", "elapsed_seconds")}, sort_keys=True), flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
