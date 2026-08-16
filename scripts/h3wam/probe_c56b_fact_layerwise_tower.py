#!/usr/bin/env python3
"""Real 8-rank mechanical probe for C56b's shared 30-layer FACT tower."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam.fact_backbone_port import fact_backbone_port_losses  # noqa: E402
from fastwam.models.h3wam.fact_layerwise_tower import (  # noqa: E402
    H3FACTLayerwiseTowerPolicy,
)
from fastwam.models.h3wam.fastwam_full_tower import (  # noqa: E402
    LAYERWISE_H3_50_TO_ACTION_30,
    initialize_full_tower_from_d0,
)


FORMAT = "h3wam-c56b-fact-layerwise-mechanical-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--kv-subdir", required=True)
    parser.add_argument("--sample-offset", type=int, default=112000)
    parser.add_argument("--d0-parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    return parser.parse_args()


def corrupt_at_timestep(
    clean: torch.Tensor, timestep: torch.Tensor, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    noise = torch.randn(
        clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
    )
    sigma = (timestep.float() / 1000.0).to(clean.dtype)
    sigma = sigma.view(-1, *([1] * (clean.ndim - 1)))
    return clean * (1.0 - sigma) + noise * sigma, noise - clean


def prediction(model: H3FACTLayerwiseTowerPolicy, inputs: dict[str, Any]):
    model.eval()
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if inputs["noisy_actions"].device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), context:
        output = model(**inputs)
    return {
        key: output[key].detach().float().cpu()
        for key in ("action", "future_state", "value", "future_representation")
    }


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.learning_rate <= 0:
        raise ValueError("C56b mechanical budget must be positive")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 8 or not torch.cuda.is_available():
        raise RuntimeError("C56b real probe requires exactly eight CUDA ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    base_output = args.output_root.resolve()
    output = base_output / "ranks" / f"rank{rank:02d}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    started = time.perf_counter()
    c58 = load_script(
        "_c56b_c58_trainer",
        REPO_ROOT / "scripts/h3wam/train_h3_fastwam_full_tower.py",
    )
    dataset = c58.PARENT.CachedDreamWAMKVDataset(
        args.manifest,
        args.cache_root,
        args.kv_subdir,
        source_manifest=args.source_manifest,
        carrier_layers=LAYERWISE_H3_50_TO_ACTION_30,
        capture_token_count=32,
        kv_pool_strategy=c58.PARENT.DREAMWAM_KV_STRATEGY,
        num_heads=56,
        attn_head_dim=128,
        action_horizon=32,
        limit=8,
        sample_offset=args.sample_offset,
    )
    batch = c58.PARENT.move_batch(
        c58.PARENT.collate_cached_batch([dataset[rank]]),
        device,
        torch.bfloat16,
    )
    parent_path = args.d0_parent_checkpoint.resolve()
    parent_sha = sha256_file(parent_path)
    if parent_sha != args.expected_parent_sha256:
        raise ValueError("C56b D0 parent SHA256 mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent.get("contract", {}).get("candidate") != "D0":
        raise ValueError("C56b parent is not D0")

    spec = c58.ModelSpec(
        carrier_layers=LAYERWISE_H3_50_TO_ACTION_30,
        carrier_source_mode=c58.LAYERWISE_H3_50_TO_ACTION_30_MODE,
    )
    tower = c58.build_model(spec, device=device, dtype=torch.bfloat16)
    expansion = initialize_full_tower_from_d0(tower, parent["model"])
    model = H3FACTLayerwiseTowerPolicy(
        tower, future_state_dim=8, future_representation_dim=256
    ).to(device=device, dtype=torch.bfloat16)
    initialization_seconds = time.perf_counter() - started

    scheduler = c58.PARENT.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    noisy_actions, action_target, timestep = (
        c58.PARENT.PARENT.deterministic_flow_batch(
            batch["actions"], scheduler, seed=args.seed + rank * 1009
        )
    )
    future_state_clean = batch["proprio"].unsqueeze(1)
    layer49_value = batch["video_kv_cache"][49]["v"].flatten(2).mean(dim=1)
    future_representation_clean = layer49_value[:, None, :256]
    value_clean = torch.zeros((1, 1, 1), device=device, dtype=torch.bfloat16)
    noisy_future_state, future_state_target = corrupt_at_timestep(
        future_state_clean, timestep, seed=args.seed + rank * 1009 + 1
    )
    noisy_value, value_target = corrupt_at_timestep(
        value_clean, timestep, seed=args.seed + rank * 1009 + 2
    )
    noisy_future_representation, future_representation_target = corrupt_at_timestep(
        future_representation_clean, timestep, seed=args.seed + rank * 1009 + 3
    )
    inputs = {
        "noisy_actions": noisy_actions,
        "timestep": timestep,
        "clean_actions": batch["actions"],
        "noisy_future_state": noisy_future_state,
        "noisy_value": noisy_value,
        "noisy_future_representation": noisy_future_representation,
        "text_context": batch["text_context"],
        "proprio": batch["proprio"],
        "video_kv_cache": batch["video_kv_cache"],
        "text_mask": batch["text_mask"],
    }
    targets = {
        "action_target": action_target,
        "future_state_target": future_state_target,
        "value_target": value_target,
        "future_representation_target": future_representation_target,
        "action_is_pad": batch["action_is_pad"],
        "action_loss_mask": torch.ones(1, device=device),
        "future_loss_mask": torch.ones(1, device=device),
        "value_loss_mask": torch.ones(1, device=device),
    }
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        parent_stage1 = model.tower(
            noisy_actions,
            timestep,
            text_context=batch["text_context"],
            proprio=batch["proprio"],
            video_kv_cache=batch["video_kv_cache"],
            text_mask=batch["text_mask"],
        )
        c56b_stage1 = model.forward_action(
            noisy_actions,
            timestep,
            text_context=batch["text_context"],
            proprio=batch["proprio"],
            video_kv_cache=batch["video_kv_cache"],
            text_mask=batch["text_mask"],
        )
    stage1_max_abs = float((parent_stage1.float() - c56b_stage1.float()).abs().max())
    if stage1_max_abs != 0.0:
        raise RuntimeError(f"C56b Stage1 changed C58b: {stage1_max_abs}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=1e-4
    )
    step_started = time.perf_counter()
    last_losses = None
    gradient_norms = None
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**inputs)
            losses = fact_backbone_port_losses(outputs, **targets)
        if not all(torch.isfinite(value) for value in losses.values()):
            raise RuntimeError("non-finite C56b mechanical loss")
        losses["loss"].backward()
        gradient_norms = [
            float(block.self_attn.o.weight.grad.float().norm().cpu())
            for block in model.shared_blocks
        ]
        if not all(math.isfinite(value) and value > 0 for value in gradient_norms):
            raise RuntimeError("C56b future/action gradient misses a shared block")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        last_losses = losses
    step_seconds = time.perf_counter() - step_started
    assert last_losses is not None and gradient_norms is not None

    before_restore = prediction(model, inputs)
    saved = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).view(-1)[0].add_(1)
    model.load_state_dict(saved, strict=True)
    after_restore = prediction(model, inputs)
    restore_max_abs = max(
        float((before_restore[key] - after_restore[key]).abs().max())
        for key in before_restore
    )
    if restore_max_abs != 0.0:
        raise RuntimeError(f"C56b strict restore failed: {restore_max_abs}")

    output.mkdir(parents=True)
    report = {
        "format": FORMAT,
        "status": "PASS_C56B_MECHANICAL_PROBE",
        "effect_status": "NOT_EVIDENCE_READY",
        "rank": rank,
        "world_size": world_size,
        "sample_id": str(dataset.rows[rank]["id"]),
        "d0_parent_sha256": parent_sha,
        "steps": args.steps,
        "global_batch": world_size,
        "stage1_max_abs": stage1_max_abs,
        "restore_max_abs": restore_max_abs,
        "shared_blocks": len(model.shared_blocks),
        "h3_layers": list(LAYERWISE_H3_50_TO_ACTION_30),
        "initialization": expansion.to_dict(),
        "losses": {
            key: float(value.detach().float().cpu()) for key, value in last_losses.items()
        },
        "min_shared_block_gradient_norm": min(gradient_norms),
        "max_shared_block_gradient_norm": max(gradient_norms),
        "initialization_seconds": initialization_seconds,
        "optimizer_step_seconds_total": step_seconds,
        "optimizer_step_seconds_mean": step_seconds / args.steps,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    dist.barrier()
    if rank == 0:
        reports = [
            json.loads(
                (base_output / "ranks" / f"rank{index:02d}" / "report.json").read_text()
            )
            for index in range(world_size)
        ]
        aggregate = {
            "format": FORMAT,
            "status": "PASS_C56B_EIGHT_GPU_MECHANICAL_PROBE",
            "effect_status": "NOT_EVIDENCE_READY",
            "world_size": world_size,
            "steps": args.steps,
            "global_batch": world_size,
            "max_stage1_abs": max(item["stage1_max_abs"] for item in reports),
            "max_restore_abs": max(item["restore_max_abs"] for item in reports),
            "max_step_seconds_mean": max(
                item["optimizer_step_seconds_mean"] for item in reports
            ),
            "max_initialization_seconds": max(
                item["initialization_seconds"] for item in reports
            ),
            "max_peak_cuda_reserved_bytes": max(
                item["peak_cuda_reserved_bytes"] for item in reports
            ),
        }
        base_output.mkdir(parents=True, exist_ok=True)
        (base_output / "AGGREGATE.json").write_text(
            json.dumps(aggregate, indent=2) + "\n"
        )
        print(json.dumps(aggregate, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
