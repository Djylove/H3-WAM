#!/usr/bin/env python3
"""Evaluate one frozen C60 milestone on the immutable balanced-80 split."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_sibling(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PAIRED = load_sibling("_c60_milestone_paired", "evaluate_c56b_fact_online_paired.py")
ONLINE = PAIRED.ONLINE
TRAIN = PAIRED.TRAIN
PROTOCOL = PAIRED.PROTOCOL
LAYERS = tuple(PAIRED.LAYERWISE_H3_50_TO_ACTION_30)
FORMAT = "h3wam-c60-fact-milestone-balanced80-v1"
SELECTED_IDS_SHA256 = PAIRED.SELECTED_IDS_SHA256
C58_PARENT_SHA256 = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
C60_DATASET_SHA256 = "1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"
C60_OBSERVATIONS_SHA256 = "b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55"
MILESTONES = tuple(range(1000, 10001, 1000))


@dataclass(frozen=True)
class Config:
    checkpoint: Path
    restore_audit: Path
    milestone: int
    h3_checkpoint: Path
    source_manifest: Path
    train_manifest: Path
    val_manifest: Path
    cache_root: Path
    output: Path
    device: str = "cuda:0"
    seed: int = 42
    inference_steps: int = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(config: Config) -> tuple[dict[str, Any], str]:
    checkpoint = config.checkpoint.resolve()
    checkpoint_sha = sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if (
        set(payload) != {
            "schema_version", "completed_steps", "model", "optimizer",
            "lr_scheduler", "contract", "probe_step", "probe_predictions",
        }
        or payload.get("schema_version") != 1
        or payload.get("completed_steps") != config.milestone
        or payload.get("probe_step") != config.milestone
        or not isinstance(payload.get("probe_predictions"), list)
        or len(payload["probe_predictions"]) != 8
    ):
        raise ValueError("C60 milestone checkpoint schema/step mismatch")
    contract = payload.get("contract", {})
    fixed = {
        "format": TRAIN.FORMAT,
        "classification": "FACT_full_backbone_port_online_frozen_int8_h3",
        "rank_categories": list(TRAIN.RANK_CATEGORIES),
        "loss_weights": [10.0, 1.0, 0.4, 0.4],
        "target_norm_sha256": "95df1f65eba1b1c3bfb9cebea90983ca54dffa69f60e6135354eb67e8551d000",
        "h3_sha256": TRAIN.EXPECTED_H3_SHA256,
        "d0_sha256": TRAIN.EXPECTED_D0_SHA256,
        "c58_parent_sha256": C58_PARENT_SHA256,
        "causal_failure_dataset_sha256": C60_DATASET_SHA256,
        "causal_failure_observations_sha256": C60_OBSERVATIONS_SHA256,
        "base_lr": 2e-5, "action_lr": 2e-4, "warmup_steps": 500,
        "scheduler_horizon": 10_000, "weight_decay": 1e-4,
        "max_grad_norm": 1.0, "seed": 20260816,
        "gradient_checkpointing": True, "action_horizon": 32,
        "action_shift": 5.0, "h3_carrier_layers": list(LAYERS),
        "h3_execution": "online_frozen_int8_per_rank_v1", "no_kv_cache": True,
    }
    mismatch = {
        key: {"actual": contract.get(key), "expected": value}
        for key, value in fixed.items() if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"C60 fixed training contract mismatch: {mismatch}")
    initialization = contract.get("initialization", {})
    if initialization != {
        "initialization_contract": "strict_online_c58b_parent_v1",
        "c58_completed_steps": 10_000,
    }:
        raise ValueError("C60 initialization contract mismatch")
    audit = json.loads(config.restore_audit.resolve().read_text(encoding="utf-8"))
    if (
        audit.get("format") != "h3wam-c56b-milestone-restore-audit-v1"
        or audit.get("status") != "PASS_C56B_MILESTONE_STRICT_RESTORE"
        or audit.get("milestone") != config.milestone
        or Path(audit.get("checkpoint", "")).resolve() != checkpoint
        or audit.get("checkpoint_size_bytes") != checkpoint.stat().st_size
        or audit.get("restore_max_abs") != 0.0
        or not all(audit.get("gate", {}).values())
    ):
        raise ValueError("C60 milestone strict-restore audit mismatch")
    return payload, checkpoint_sha


def run(config: Config) -> dict[str, Any]:
    if config.milestone not in MILESTONES:
        raise ValueError("milestone must be one of s1000..s10000")
    if config.seed != 42 or config.inference_steps != 10:
        raise ValueError("balanced80 seed/solver is fixed")
    payload, checkpoint_sha = load_checkpoint(config)
    source_rows = PROTOCOL.read_jsonl(config.source_manifest.resolve())
    train_rows = PROTOCOL.read_jsonl(config.train_manifest.resolve())
    val_rows = PROTOCOL.read_jsonl(config.val_manifest.resolve())
    split = PROTOCOL.validate_episode_disjoint_manifests(source_rows, train_rows, val_rows)
    selected, selection = PROTOCOL.select_validation_rows(val_rows, samples_per_task=2)
    if (
        len(selected) != 80
        or selection["selected_ids_sha256"] != SELECTED_IDS_SHA256
        or selection["selected_task_count"] != 40
        or any(count != 2 for count in selection["task_counts"].values())
    ):
        raise ValueError("balanced80 selection drifted")
    contract = payload["contract"]
    actual_data = {
        "source_manifest_sha256": sha256_file(config.source_manifest.resolve()),
        "demo_manifest_sha256": sha256_file(config.train_manifest.resolve()),
        "demo_stats_sha256": sha256_file(config.cache_root.resolve() / "stats.pt"),
    }
    if any(contract.get(key) != value for key, value in actual_data.items()):
        raise ValueError("heldout data differs from C60 training contract")
    visual_mapping, visual_contract = PROTOCOL.build_visual_feature_shuffle(selected)
    dataset = ONLINE.OnlineC58bValidationDataset(
        selected, cache_root=config.cache_root.resolve(), action_horizon=32,
        visual_mapping=visual_mapping,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=ONLINE.collate_online,
    )
    device, dtype = PROTOCOL._resolve_device_dtype(config.device)
    if device.type != "cuda":
        raise ValueError("online INT8 H3 milestone evaluation requires CUDA")
    if sha256_file(config.h3_checkpoint.resolve()) != TRAIN.EXPECTED_H3_SHA256:
        raise ValueError("H3 checkpoint identity mismatch")
    provider = ONLINE.C58OnlineFrozenH3Provider(
        config.h3_checkpoint.resolve(), layers=LAYERS
    ).to(device=device).eval()
    scheduler = PROTOCOL.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    started = time.perf_counter()
    arm = PAIRED.evaluate_arm(
        name=f"c60_s{config.milestone}", payload=payload, loader=loader,
        provider=provider, dataset=dataset, scheduler=scheduler,
        device=device, dtype=dtype, seed=config.seed,
    )
    gates = PAIRED._conditioning_gate(arm)
    gc.collect()
    torch.cuda.empty_cache()
    report = {
        "format": FORMAT,
        "status": "PASS_FIXED_BALANCED80" if all(gates.values()) else "FAIL_CONDITIONING_COLLAPSE",
        "effect_status": "DIAGNOSTIC_NOT_CHECKPOINT_SELECTION",
        "milestone": config.milestone,
        "checkpoint": str(config.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "restore_audit": str(config.restore_audit.resolve()),
        "restore_audit_sha256": sha256_file(config.restore_audit.resolve()),
        "training_contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest(),
        "data": {
            **actual_data,
            "validation_manifest_sha256": sha256_file(config.val_manifest.resolve()),
            "selection": selection, "split_audit": split,
            "visual_shuffle": visual_contract,
        },
        "execution": {
            "h3": "online_frozen_int8", "h3_checkpoint_sha256": TRAIN.EXPECTED_H3_SHA256,
            "disk_kv_read": False, "disk_kv_write": False,
            "disk_feature_read": False, "carrier_layers": list(LAYERS),
            "same_selected_samples_noise_solver_normalization": True,
            "seed": 42, "inference_steps": 10, "shift": 5.0,
        },
        "arm": arm,
        "conditioning_gates": gates,
        "timing": {"elapsed_seconds": time.perf_counter() - started},
        "claim_boundary": (
            "Fixed offline milestone diagnostic only. It cannot override the "
            "completed 680-pair closed-loop KEEP_C58_PARENT decision."
        ),
    }
    output = config.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return report


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--restore-audit", type=Path, required=True)
    parser.add_argument("--milestone", type=int, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    values = parser.parse_args()
    return Config(**vars(values))


def main() -> None:
    report = run(parse_args())
    metrics = report["arm"]["metrics"]
    print(json.dumps({
        "status": report["status"], "milestone": report["milestone"],
        "normalized": metrics["normalized_clip5_model_domain"],
        "physical": metrics["denormalized_official_minmax_clamp"],
        "gripper": metrics["gripper_sign"],
        "conditioning_gates": report["conditioning_gates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
