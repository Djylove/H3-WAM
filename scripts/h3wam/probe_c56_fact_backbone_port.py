#!/usr/bin/env python3
"""One-step mechanical probe for the full causal-token H3/FACT backbone port.

The real mode consumes one immutable C48/C59 rollout row and the D0 parent.  It
does not create a research checkpoint suitable for comparison or long training;
the saved payload exists only to prove strict save/restore after one optimizer
step.  ``--synthetic`` is a CPU unit smoke with the same causal graph.
"""

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

from fastwam.models.h3wam.dreamwam_kv_carrier import (  # noqa: E402
    DEFAULT_H3_CARRIER_LAYERS,
    H3DreamWAMKVCarrierPolicy,
    REPEAT_LAYER49_CARRIER_SOURCE,
)
from fastwam.models.h3wam.fact_backbone_port import (  # noqa: E402
    C59FailureOverlay,
    C60CausalFailureLabels,
    FACT_CURRENT_AUDITED_COMMIT,
    FACT_PINNED_COMMIT,
    H3FACTBackbonePort,
    fact_backbone_port_losses,
)


FORMAT = "h3wam-c56-fact-backbone-port-mechanical-v1"
EXPECTED_H3_SHA256 = (
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
)
EXPECTED_C60_SHA256 = "1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"


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
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--expected-parent-sha256")
    parser.add_argument("--demo-cache-root", type=Path)
    parser.add_argument("--rollout-dataset", type=Path)
    parser.add_argument("--rollout-projected-features", type=Path)
    parser.add_argument("--rollout-kv-root", type=Path)
    parser.add_argument("--c59-overlay-root", type=Path)
    parser.add_argument("--c60-failure-dataset", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--causal-layers", type=int, default=5)
    parser.add_argument("--causal-heads", type=int, default=16)
    return parser.parse_args()


def flow_corrupt(
    clean: torch.Tensor, timestep: torch.Tensor, *, seed: int, max_timestep: float = 1000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    noise = torch.randn(
        clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
    )
    sigma = (timestep.float() / max_timestep).to(clean.dtype)
    sigma = sigma.view(-1, *([1] * (clean.ndim - 1)))
    return (1.0 - sigma) * clean + sigma * noise, noise - clean


def small_synthetic(
    device: torch.device, args: argparse.Namespace
) -> tuple[H3FACTBackbonePort, dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    carrier = H3DreamWAMKVCarrierPolicy(
        enabled=True,
        carrier_layers=(1, 3),
        action_dim=2,
        proprio_dim=3,
        context_dim=6,
        hidden_dim=16,
        ffn_dim=32,
        num_heads=2,
        attn_head_dim=4,
        freq_dim=8,
    ).to(device)
    model = H3FACTBackbonePort(
        carrier,
        hidden_dim=16,
        future_representation_dim=5,
        future_state_dim=3,
        causal_layers=2,
        causal_heads=4,
        causal_ffn_dim=32,
        h3_source_layer=3,
    ).to(device)
    batch = 2
    cache = {
        layer: {
            name: torch.randn(batch, 4, 2, 4, device=device).clone()
            for name in ("k", "v")
        }
        for layer in (1, 3)
    }
    clean_actions = torch.randn(batch, 4, 2, device=device)
    future_state = torch.randn(batch, 3, device=device)
    value = torch.randn(batch, device=device)
    future_representation = torch.randn(batch, 5, device=device)
    timestep = torch.tensor([800.0, 250.0], device=device)
    noisy_actions, action_target = flow_corrupt(
        clean_actions, timestep, seed=args.seed + 1
    )
    noisy_state, state_target = flow_corrupt(
        future_state, timestep, seed=args.seed + 2
    )
    noisy_value, value_target = flow_corrupt(value, timestep, seed=args.seed + 3)
    noisy_representation, representation_target = flow_corrupt(
        future_representation, timestep, seed=args.seed + 4
    )
    inputs = {
        "noisy_actions": noisy_actions,
        "timestep": timestep,
        "clean_actions": clean_actions,
        "noisy_future_state": noisy_state,
        "noisy_value": noisy_value,
        "noisy_future_representation": noisy_representation,
        "text_context": torch.randn(batch, 3, 6, device=device),
        "text_mask": torch.tensor(
            [[True, True, False], [True, True, True]], device=device
        ),
        "proprio": torch.randn(batch, 3, device=device),
        "video_kv_cache": cache,
    }
    targets = {
        "action_target": action_target,
        "future_state_target": state_target,
        "value_target": value_target,
        "future_representation_target": representation_target,
        "action_is_pad": torch.zeros(batch, 4, dtype=torch.bool, device=device),
        "action_loss_mask": torch.tensor([1.0, 0.0], device=device),
        "future_loss_mask": torch.ones(batch, device=device),
        "value_loss_mask": torch.ones(batch, device=device),
    }
    identity = {"mode": "synthetic", "sample_id": [0, 1]}
    return model, inputs, targets, identity


def real_sample(
    device: torch.device, args: argparse.Namespace
) -> tuple[H3FACTBackbonePort, dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    required = (
        "parent_checkpoint",
        "expected_parent_sha256",
        "demo_cache_root",
        "rollout_dataset",
        "rollout_projected_features",
        "rollout_kv_root",
        "c59_overlay_root",
        "c60_failure_dataset",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"real probe missing arguments: {missing}")
    parent_path = args.parent_checkpoint.resolve()
    parent_sha = sha256_file(parent_path)
    if parent_sha != args.expected_parent_sha256:
        raise ValueError("D0 parent SHA256 mismatch")

    c55 = load_script(
        "_c56_c55_loader", REPO_ROOT / "scripts/h3wam/train_c55_fact_joint_action.py"
    )
    d0 = c55.D0
    rollout = c55.C55RolloutDataset(
        args.rollout_dataset.resolve(),
        args.rollout_projected_features.resolve(),
        args.rollout_kv_root.resolve(),
        args.demo_cache_root.resolve(),
        split="train",
    )
    overlay = C59FailureOverlay(
        args.c59_overlay_root.resolve(),
        source_dataset=args.rollout_dataset.resolve(),
        value_contract="fact_code_remaining_plus_penalty",
    )
    c60 = C60CausalFailureLabels(
        args.c60_failure_dataset.resolve(),
        expected_sha256=EXPECTED_C60_SHA256,
        value_contract="fact_code_remaining_plus_penalty",
    )
    # Use one deterministic success row.  Failure masking/onset semantics are
    # separately exhaustively audited by C59; this probe is architecture-only.
    probe_rank = int(getattr(args, "probe_rank", 0))
    row_index = rollout.success_indices[probe_rank % len(rollout.success_indices)]
    item = rollout[row_index]
    sample_id = int(rollout.rows[row_index]["sample_id"])
    overlay_row = overlay.for_sample(sample_id)
    if overlay_row["action_loss_mask"] != 1:
        raise RuntimeError("selected mechanical row is not a supervised action row")
    batch = c55.move_targets(c55.collate_rollout(item), device)

    spec = d0.ModelSpec(
        carrier_layers=DEFAULT_H3_CARRIER_LAYERS,
        carrier_source_mode=REPEAT_LAYER49_CARRIER_SOURCE,
    )
    carrier = d0.build_model(spec, device=device, dtype=torch.bfloat16)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    contract = parent.get("contract", {})
    expected_parent = {
        "candidate": "D0",
        "carrier_source_mode": REPEAT_LAYER49_CARRIER_SOURCE,
        "h3_checkpoint_sha256": EXPECTED_H3_SHA256,
        "action_horizon": 32,
        "kv_strategy": d0.DREAMWAM_KV_STRATEGY,
    }
    mismatch = [
        key for key, value in expected_parent.items() if contract.get(key) != value
    ]
    if mismatch:
        raise ValueError(f"D0 parent contract mismatch: {mismatch}")
    carrier.load_state_dict(parent["model"], strict=True)
    model = H3FACTBackbonePort(
        carrier,
        hidden_dim=1024,
        future_representation_dim=256,
        future_state_dim=8,
        causal_layers=args.causal_layers,
        causal_heads=args.causal_heads,
        causal_ffn_dim=4096,
        h3_source_layer=49,
    ).to(device=device, dtype=torch.bfloat16)

    scheduler = d0.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    noisy_actions, action_target, timestep = c55.PARENT.deterministic_flow_batch(
        batch["actions"], scheduler, seed=args.seed + 1
    )
    future_state = batch["future_state"].to(torch.bfloat16)
    value_raw = torch.tensor(
        [float(overlay_row["value_target"]) - 1.0],
        device=device,
        dtype=torch.bfloat16,
    )
    future_representation = batch["future_h3"].to(torch.bfloat16)
    noisy_state, state_target = flow_corrupt(
        future_state, timestep, seed=args.seed + 2
    )
    noisy_value, value_target = flow_corrupt(value_raw, timestep, seed=args.seed + 3)
    noisy_representation, representation_target = flow_corrupt(
        future_representation, timestep, seed=args.seed + 4
    )
    inputs = {
        "noisy_actions": noisy_actions,
        "timestep": timestep,
        "clean_actions": batch["actions"],
        "noisy_future_state": noisy_state,
        "noisy_value": noisy_value,
        "noisy_future_representation": noisy_representation,
        "text_context": batch["text_context"],
        "text_mask": batch["text_mask"],
        "proprio": batch["proprio"],
        "video_kv_cache": batch["video_kv_cache"],
    }
    targets = {
        "action_target": action_target,
        "future_state_target": state_target,
        "value_target": value_target,
        "future_representation_target": representation_target,
        "action_is_pad": batch["action_is_pad"],
        "action_loss_mask": torch.tensor([1.0], device=device),
        "future_loss_mask": torch.tensor([1.0], device=device),
        "value_loss_mask": torch.tensor([1.0], device=device),
    }
    identity = {
        "mode": "real_c48_c59",
        "sample_id": sample_id,
        "parent_checkpoint": str(parent_path),
        "parent_sha256": parent_sha,
        "rollout_dataset_sha256": sha256_file(args.rollout_dataset.resolve()),
        "c59_overlay": str(args.c59_overlay_root.resolve()),
        "c59_value_contract": overlay.value_contract,
        "c59_label": overlay_row,
        "c60_failure_dataset": str(args.c60_failure_dataset.resolve()),
        "c60_failure_dataset_sha256": c60.sha256,
        "c60_train_rows": len(c60.split("train")),
        "c60_validation_rows": len(c60.split("validation")),
    }
    return model, inputs, targets, identity


def prediction_probe(model: H3FACTBackbonePort, inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
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
    if args.learning_rate <= 0 or args.causal_layers <= 0 or args.causal_heads <= 0:
        raise ValueError("probe hyperparameters must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.probe_rank = rank
    base_output_root = args.output_root.resolve()
    output_root = (
        base_output_root / "ranks" / f"rank{rank:02d}"
        if world_size > 1
        else base_output_root
    )
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {output_root}")
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(device)
    if world_size > 1:
        if device.type != "cuda":
            raise RuntimeError("multi-rank C56 probe requires CUDA/NCCL")
        dist.init_process_group("nccl")
    torch.manual_seed(args.seed + rank)
    started = time.perf_counter()
    model, inputs, targets, identity = (
        small_synthetic(device, args) if args.synthetic else real_sample(device, args)
    )
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=1e-4
    )

    # Parent equivalence is guaranteed by the zero action residual before step 1.
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), context:
        parent_prediction = model.carrier(
            inputs["noisy_actions"],
            inputs["timestep"],
            text_context=inputs["text_context"],
            text_mask=inputs["text_mask"],
            proprio=inputs["proprio"],
            video_kv_cache=inputs["video_kv_cache"],
        )
        initial_prediction = model.forward_action(
            inputs["noisy_actions"],
            inputs["timestep"],
            text_context=inputs["text_context"],
            text_mask=inputs["text_mask"],
            proprio=inputs["proprio"],
            video_kv_cache=inputs["video_kv_cache"],
        )
        parent_max_abs = float(
            (parent_prediction.float() - initial_prediction.float()).abs().max().cpu()
        )
    if parent_max_abs != 0.0:
        raise RuntimeError(f"step0 D0 preservation failed: {parent_max_abs}")

    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with context:
        output = model(**inputs)
    losses = fact_backbone_port_losses(output, **targets)
    if not all(torch.isfinite(value) for value in losses.values()):
        raise RuntimeError("non-finite C56 loss")
    losses["loss"].backward()
    gradient_groups = {
        "causal_backbone": sum(
            float(parameter.grad.float().square().sum().cpu())
            for parameter in model.causal_backbone.parameters()
            if parameter.grad is not None
        )
        ** 0.5,
        "carrier_clean_action_path": sum(
            float(parameter.grad.float().square().sum().cpu())
            for parameter in model.carrier.action_expert.blocks[0].parameters()
            if parameter.grad is not None
        )
        ** 0.5,
        "future_representation_decoder": sum(
            float(parameter.grad.float().square().sum().cpu())
            for parameter in model.future_representation_decoder.parameters()
            if parameter.grad is not None
        )
        ** 0.5,
    }
    if not all(math.isfinite(value) and value > 0 for value in gradient_groups.values()):
        raise RuntimeError(f"C56 gradient path failed: {gradient_groups}")
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    )
    optimizer.step()
    probe = prediction_probe(model, inputs)

    output_root.mkdir(parents=True)
    checkpoint = output_root / "mechanical-step1.pt"
    payload = {
        "format": FORMAT,
        "completed_steps": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "identity": identity,
        "topology": {
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "device": str(device),
        },
        "probe_prediction": probe,
    }
    temporary = output_root / ".mechanical-step1.pt.partial"
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)

    restored, _, _, _ = (
        small_synthetic(device, args) if args.synthetic else real_sample(device, args)
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored.load_state_dict(saved["model"], strict=True)
    restored_probe = prediction_probe(restored, inputs)
    restore_max_abs = max(
        float((restored_probe[key] - probe[key]).abs().max()) for key in probe
    )
    if restore_max_abs != 0.0:
        raise RuntimeError(f"strict restore probe failed: {restore_max_abs}")

    report = {
        "format": FORMAT,
        "status": "PASS_C56_MECHANICAL_PROBE",
        "effect_status": "NOT_EVIDENCE_READY",
        "training_permission": "PROBE_ONLY",
        "identity": identity,
        "resolved_config": {
            "causal_layers": 2 if args.synthetic else args.causal_layers,
            "causal_heads": 4 if args.synthetic else args.causal_heads,
            "learning_rate": args.learning_rate,
            "optimizer_steps": 1,
            "global_batch": int(inputs["noisy_actions"].shape[0]),
            "fact_pinned_commit": FACT_PINNED_COMMIT,
            "fact_current_audited_commit": FACT_CURRENT_AUDITED_COMMIT,
            "value_contract": "fact_code_remaining_plus_penalty",
            "loss_weights": {
                "action": 10.0,
                "future_representation": 1.0,
                "future_state": 0.4,
                "value": 0.4,
            },
        },
        "losses": {key: float(value.detach().cpu()) for key, value in losses.items()},
        "gradient_norms": gradient_groups,
        "clipped_gradient_norm": grad_norm,
        "step0_parent_max_abs": parent_max_abs,
        "restore_max_abs": restore_max_abs,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if world_size > 1:
        dist.barrier()
        if rank == 0:
            reports = [
                json.loads(
                    (base_output_root / "ranks" / f"rank{index:02d}" / "report.json").read_text(
                        encoding="utf-8"
                    )
                )
                for index in range(world_size)
            ]
            aggregate = {
                "format": FORMAT,
                "status": (
                    "PASS_C56_EIGHT_GPU_MECHANICAL_PROBE"
                    if all(item["status"] == "PASS_C56_MECHANICAL_PROBE" for item in reports)
                    else "FAIL_C56_EIGHT_GPU_MECHANICAL_PROBE"
                ),
                "effect_status": "NOT_EVIDENCE_READY",
                "world_size": world_size,
                "sample_ids": [item["identity"]["sample_id"] for item in reports],
                "max_restore_abs": max(item["restore_max_abs"] for item in reports),
                "max_step0_parent_abs": max(item["step0_parent_max_abs"] for item in reports),
                "max_peak_cuda_allocated_bytes": max(
                    item["peak_cuda_allocated_bytes"] for item in reports
                ),
                "max_peak_cuda_reserved_bytes": max(
                    item["peak_cuda_reserved_bytes"] for item in reports
                ),
                "max_elapsed_seconds": max(item["elapsed_seconds"] for item in reports),
                "reports": [
                    str(base_output_root / "ranks" / f"rank{index:02d}" / "report.json")
                    for index in range(world_size)
                ],
            }
            (base_output_root / "AGGREGATE.json").write_text(
                json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(aggregate, sort_keys=True))
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
