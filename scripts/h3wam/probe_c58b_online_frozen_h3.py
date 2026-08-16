#!/usr/bin/env python3
"""Gate online frozen INT8 H3 -> official FastWAM full30 ActionDiT training.

This is deliberately a real training step, not a shape-only smoke test.  It
proves that live H3 K/V is identical to the audited disk-cache path on one
registered sample, then performs one full backward/AdamW step while reporting
the single-A800 memory and timing envelope.  Disk K/V is used only as a parity
oracle; the measured training step recomputes all 30 H3 layers online.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam.fastwam_full_tower import (  # noqa: E402
    LAYERWISE_H3_50_TO_ACTION_30,
    initialize_full_tower_from_d0,
)
from fastwam.models.h3wam.int8_backbone import H3Int8FeatureBackbone  # noqa: E402
from fastwam.models.h3wam.int8_online import (  # noqa: E402
    H3Int8OnlineKVContract,
    H3Int8OnlineKVProvider,
    SEQUENCE_KV_POOL,
)


def _load_c58():
    path = Path(__file__).with_name("train_h3_fastwam_full_tower.py")
    spec = importlib.util.spec_from_file_location("_c58b_online_parent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load C58 trainer {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C58 = _load_c58()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--kv-subdir", required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--d0-parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=112000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def selected_manifest_row(manifest: Path, sample_offset: int) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in manifest.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if sample_offset < 0 or sample_offset >= len(rows):
        raise ValueError("sample-offset must select an existing manifest row")
    return rows[sample_offset]


def compare_kv_exact(
    online: dict[int, dict[str, torch.Tensor]],
    cached: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    if tuple(online) != tuple(cached):
        raise RuntimeError("online/cached H3 K/V layer order differs")
    results: dict[str, Any] = {}
    for layer in online:
        for name in ("k", "v"):
            actual = online[layer][name].detach().cpu()
            expected = cached[layer][name].detach().cpu()
            max_abs = float((actual.float() - expected.float()).abs().max())
            exact = torch.equal(actual, expected)
            results[f"{layer}.{name}"] = {
                "exact": exact,
                "max_abs": max_abs,
                "shape": list(actual.shape),
            }
            if not exact:
                raise RuntimeError(
                    f"online/cache parity failed at H3 layer {layer} {name}: {max_abs}"
                )
    return results


def cuda_memory() -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cpu" or not torch.cuda.is_available():
        raise ValueError("C58b online training gate requires CUDA")
    if min(args.learning_rate, args.max_grad_norm) <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimizer arguments")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    started = time.perf_counter()

    row = selected_manifest_row(args.manifest, args.sample_offset)
    cached_dataset = C58.PARENT.CachedDreamWAMKVDataset(
        args.manifest,
        args.cache_root,
        args.kv_subdir,
        source_manifest=args.source_manifest,
        carrier_layers=LAYERWISE_H3_50_TO_ACTION_30,
        capture_token_count=32,
        kv_pool_strategy=C58.PARENT.DREAMWAM_KV_STRATEGY,
        num_heads=56,
        attn_head_dim=128,
        action_horizon=32,
        limit=1,
        sample_offset=args.sample_offset,
    )
    if str(cached_dataset.rows[0]["id"]) != str(row["id"]):
        raise RuntimeError("parity cache escaped the selected online row")
    cached_cpu = C58.PARENT.collate_cached_batch([cached_dataset[0]])
    if cached_cpu["sample_ids"] != [str(row["id"])]:
        raise RuntimeError("cached parity sample identity mismatch")

    cache_root = args.cache_root.resolve()
    window = torch.load(
        cache_root / "windows" / f"{row['id']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    context = torch.load(
        cache_root / "contexts" / f"{row['context_id']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if context.get("text_only") is not True:
        raise ValueError("online probe requires an audited text-only H3 context")
    first_frame = window["first_frame_latents"].to(device=device)
    h3_context = context["context"].to(device=device)
    token_tags = context["token_tags"].to(device=device)

    h3_load_started = time.perf_counter()
    backbone = H3Int8FeatureBackbone.from_checkpoint(args.h3_checkpoint.resolve())
    backbone = backbone.to(device=device).eval()
    backbone.requires_grad_(False)
    provider = H3Int8OnlineKVProvider(
        backbone,
        H3Int8OnlineKVContract(
            layers=LAYERWISE_H3_50_TO_ACTION_30,
            action_horizon=32,
            target_latent_frames=12,
            video_timestep=1.0,
            condition_video_timestep=1.0,
            capture_token_count=32,
            pool_strategy=SEQUENCE_KV_POOL,
        ),
    ).eval()
    torch.cuda.synchronize()
    h3_load_seconds = time.perf_counter() - h3_load_started
    memory_after_h3 = cuda_memory()

    d0_payload = torch.load(
        args.d0_parent_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    C58.require_d0_parent(d0_payload, dataset=cached_dataset, args=argparse.Namespace(
        expected_h3_checkpoint_sha256=C58.PARENT.H3_INT8_CHECKPOINT_SHA256,
        action_horizon=32,
        action_shift=5.0,
    ))
    spec = C58.ModelSpec(
        carrier_layers=LAYERWISE_H3_50_TO_ACTION_30,
        carrier_source_mode=C58.LAYERWISE_H3_50_TO_ACTION_30_MODE,
        action_layers=30,
    )
    action_model = C58.build_model(
        spec,
        device=device,
        dtype=dtype,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    initialization = initialize_full_tower_from_d0(
        action_model, d0_payload["model"]
    ).to_dict()
    del d0_payload
    optimizer = torch.optim.AdamW(
        action_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    memory_after_models = cuda_memory()

    parity_started = time.perf_counter()
    online_kv = provider(first_frame, h3_context, token_tags)
    torch.cuda.synchronize()
    online_extract_parity_seconds = time.perf_counter() - parity_started
    cached_device = C58.PARENT.move_batch(cached_cpu, device, dtype)
    kv_parity = compare_kv_exact(online_kv, cached_device["video_kv_cache"])

    flow_scheduler = C58.PARENT.FlowMatchScheduler(
        num_train_timesteps=1000, shift=5.0
    )
    noisy, _, timesteps = C58.PARENT.PARENT.deterministic_flow_batch(
        cached_device["actions"], flow_scheduler, seed=args.seed + 8_000_001
    )
    action_model.eval()
    online_batch = dict(cached_device)
    online_batch["video_kv_cache"] = online_kv
    with torch.no_grad():
        cached_prediction = C58.forward_policy(
            action_model, cached_device, noisy, timesteps
        )
        online_prediction = C58.forward_policy(
            action_model, online_batch, noisy, timesteps
        )
    action_max_abs = float(
        (cached_prediction.float() - online_prediction.float()).abs().max()
    )
    if not torch.equal(cached_prediction, online_prediction):
        raise RuntimeError(f"online/cache action parity failed: {action_max_abs}")
    del cached_prediction, online_prediction, online_kv
    torch.cuda.empty_cache()

    # The measured step is genuinely online: recompute all thirty H3 K/V
    # tensors after resetting the memory counters, then backward through every
    # official FastWAM ActionDiT block and materialize AdamW optimizer state.
    torch.cuda.reset_peak_memory_stats()
    step_started = time.perf_counter()
    extract_started = time.perf_counter()
    live_kv = provider(first_frame, h3_context, token_tags)
    torch.cuda.synchronize()
    online_extract_train_seconds = time.perf_counter() - extract_started
    live_batch = dict(cached_device)
    live_batch["video_kv_cache"] = live_kv
    action_model.train()
    optimizer.zero_grad(set_to_none=True)
    noisy, target, timesteps = C58.PARENT.PARENT.deterministic_flow_batch(
        live_batch["actions"], flow_scheduler, seed=args.seed + 1_000_003
    )
    action_started = time.perf_counter()
    prediction = C58.forward_policy(action_model, live_batch, noisy, timesteps)
    loss = C58.PARENT.flow_matching_loss(
        prediction,
        target,
        timesteps,
        flow_scheduler,
        is_pad_mask=live_batch["action_is_pad"],
    )
    loss.backward()
    block_gradients = [
        C58.PARENT.PARENT.module_grad_norm(block)
        for block in action_model.action_expert.blocks
    ]
    if len(block_gradients) != 30 or min(block_gradients) <= 0:
        raise RuntimeError("the online step did not train all 30 ActionDiT blocks")
    total_grad_norm = float(
        torch.nn.utils.clip_grad_norm_(action_model.parameters(), args.max_grad_norm)
    )
    head_before = action_model.action_expert.head.weight.detach().float().clone()
    optimizer.step()
    head_delta = float(
        (action_model.action_expert.head.weight.detach().float() - head_before)
        .abs()
        .max()
    )
    torch.cuda.synchronize()
    action_train_seconds = time.perf_counter() - action_started
    total_step_seconds = time.perf_counter() - step_started
    if head_delta <= 0:
        raise RuntimeError("optimizer step did not update the ActionDiT head")

    report = {
        "format": "h3wam-c58b-online-frozen-h3-one-step-gate-v1",
        "status": "PASS_ONLINE_FROZEN_H3_ONE_STEP",
        "sample_id": str(row["id"]),
        "sample_offset": args.sample_offset,
        "online_contract": {
            "h3_trainable": False,
            "h3_quantization": "int8_tensorwise_convrot",
            "h3_layers": list(LAYERWISE_H3_50_TO_ACTION_30),
            "kv_pool": SEQUENCE_KV_POOL,
            "kv_tokens": 32,
            "action_tower": "official FastWAM full30 ActionDiT",
            "action_tower_trainable": True,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "disk_kv_role": "parity_oracle_only_not_training_input",
        },
        "initialization": initialization,
        "parity": {
            "kv_all_exact": True,
            "kv": kv_parity,
            "action_exact": True,
            "action_max_abs": action_max_abs,
        },
        "training": {
            "loss": float(loss.detach()),
            "timestep": float(timesteps.float().mean()),
            "block_gradient_norms": block_gradients,
            "all_30_blocks_nonzero_gradient": True,
            "total_gradient_norm_before_clip": total_grad_norm,
            "action_head_max_abs_update": head_delta,
        },
        "timing_seconds": {
            "h3_checkpoint_load": h3_load_seconds,
            "online_extract_parity": online_extract_parity_seconds,
            "online_extract_training": online_extract_train_seconds,
            "action_forward_backward_optimizer": action_train_seconds,
            "online_training_step_total": total_step_seconds,
            "script_total": time.perf_counter() - started,
        },
        "memory": {
            "after_h3_load": memory_after_h3,
            "after_h3_and_action_models": memory_after_models,
            "one_step_peak": cuda_memory(),
        },
        "claim_boundary": (
            "Proves exact online/cache feature parity and one real single-GPU "
            "training step only; it does not prove optimization quality or rollout success."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
