#!/usr/bin/env python3
"""Fail-closed audit of the C74 32-GPU, 32,647-step action-only trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


FORMAT = "h3wam-c56b-fact-online-training-v1"
READY_FORMAT = "h3wam-c74-action-only-multinode-three-expert-epoch-complete-v1"
MILESTONES = (
    tuple(range(1_000, 7_001, 1_000)) + (7_549,)
    + tuple(range(8_000, 32_001, 1_000)) + (32_647,)
)
C58_SHA256 = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
H3_SHA256 = "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
D0_SHA256 = "36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"
C60_DATASET_SHA256 = "1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"
C60_OBSERVATIONS_SHA256 = "b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55"
TARGET_NORM_SHA256 = "95df1f65eba1b1c3bfb9cebea90983ca54dffa69f60e6135354eb67e8551d000"
LAYERS = (
    0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24,
    25, 27, 29, 30, 32, 34, 35, 37, 39, 41, 42, 44, 46, 47, 49,
)
AUXILIARY_PREFIXES = (
    "future_state_encoder.", "value_encoder.", "future_representation_encoder.",
    "future_state_decoder.", "value_decoder.", "future_representation_decoder.",
)
CHECKPOINT_KEYS = {
    "schema_version", "completed_steps", "model", "optimizer",
    "lr_scheduler", "contract", "probe_step", "probe_predictions",
}
MIN_CHECKPOINT_BYTES = 10 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_contract(contract: dict[str, Any]) -> None:
    fixed = {
        "format": FORMAT,
        "classification": "FACT_full_backbone_port_online_frozen_int8_h3",
        "objective_mode": "action_only",
        "rank_categories": [
            "expert_demo", "expert_demo", "expert_demo", "expert_demo",
            "success_rollout", "success_rollout", "observational_failure",
            "causal_failure",
        ] * 4,
        "rank_schedule": {
            "name": "c67_4_2_1_1", "group_size": 8, "group_repetitions": 4,
        },
        "loss_weights": [10.0, 0.0, 0.0, 0.0],
        "target_norm_sha256": TARGET_NORM_SHA256,
        "h3_sha256": H3_SHA256,
        "d0_sha256": D0_SHA256,
        "c58_parent_sha256": C58_SHA256,
        "causal_failure_dataset_sha256": C60_DATASET_SHA256,
        "causal_failure_observations_sha256": C60_OBSERVATIONS_SHA256,
        "base_lr": 8e-5,
        "action_lr": 8e-4,
        "warmup_steps": 125,
        "scheduler_horizon": 32_647,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "seed": 20260816,
        "gradient_checkpointing": True,
        "action_horizon": 32,
        "action_shift": 5.0,
        "h3_carrier_layers": list(LAYERS),
        "h3_execution": "online_frozen_int8_per_rank_v1",
        "no_kv_cache": True,
    }
    mismatches = {
        key: {"actual": contract.get(key), "expected": expected}
        for key, expected in fixed.items() if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"C74 fixed contract mismatch: {mismatches}")
    if contract.get("initialization") != {
        "initialization_contract": "strict_online_c58b_parent_v1",
        "c58_completed_steps": 10_000,
    }:
        raise ValueError("C74 fixed C58b initialization mismatch")
    frozen = contract.get("frozen_auxiliary_parameters")
    if (
        not isinstance(frozen, list) or not frozen
        or len(frozen) != len(set(frozen))
        or any(not name.startswith(AUXILIARY_PREFIXES) for name in frozen)
        or any(not any(name.startswith(prefix) for name in frozen) for prefix in AUXILIARY_PREFIXES)
    ):
        raise ValueError("C74 auxiliary freeze contract mismatch")
    for name in (
        "demo_manifest_sha256", "source_manifest_sha256", "demo_stats_sha256",
        "c48_dataset_sha256", "c48_observations_sha256",
        "c59_completed_sha256", "c59_sample_labels_sha256",
    ):
        value = contract.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"C74 missing data identity: {name}")


def expected_lr_factor(step: int) -> float:
    if step < 125:
        return float(step + 1) / 125
    progress = min(1.0, (step - 125) / 32_522)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def previous_milestone(milestone: int) -> int:
    index = MILESTONES.index(milestone)
    return 0 if index == 0 else MILESTONES[index - 1]


def validate_milestone(
    root: Path, milestone: int, expected_contract: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = root / f"checkpoints/c74_action_only_s{milestone}.pt"
    train_path = root / f"reports/train_s{milestone}.json"
    restore_path = root / f"restore/restore_s{milestone}.json"
    for path in (checkpoint, train_path, restore_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    train, restore = load_json(train_path), load_json(restore_path)
    history, contract = train.get("history", []), train.get("contract", {})
    require_contract(contract)
    if expected_contract is not None and contract != expected_contract:
        raise ValueError(f"C74 contract drift at s{milestone}")
    previous_step = previous_milestone(milestone)
    previous = (
        None if previous_step == 0
        else root / f"checkpoints/c74_action_only_s{previous_step}.pt"
    )
    segment_steps = milestone - previous_step
    expected_steps = list(range(previous_step + 1, milestone + 1))
    last_lrs = history[-1].get("learning_rates", {}) if history else {}
    factor = expected_lr_factor(milestone)
    checks = {
        "status": train.get("status") == "PASS_C56B_ONLINE_TRAINING_INVOCATION",
        "step": train.get("completed_steps") == milestone,
        "checkpoint": Path(train.get("checkpoint", "")).resolve() == checkpoint.resolve()
        and train.get("checkpoint_bytes") == checkpoint.stat().st_size
        and checkpoint.stat().st_size >= MIN_CHECKPOINT_BYTES,
        "history": len(history) == segment_steps
        and [row.get("step") for row in history] == expected_steps,
        "finite": len(history) == segment_steps and all(
            all(math.isfinite(float(row.get(key, math.nan))) for key in (
                "loss", "action_loss", "future_representation_loss",
                "future_state_loss", "value_loss",
            )) for row in history
        ),
        "gradients": len(history) == segment_steps and all(
            len(row.get("block_gradient_norms_mean_across_ranks", [])) == 30
            and min(row["block_gradient_norms_mean_across_ranks"]) > 0 for row in history
        ),
        "no_leak": len(history) == segment_steps
        and max(row.get("sum_rank_future_leak_abs", math.inf) for row in history) == 0,
        "lineage": train.get("loaded_checkpoint") is None if previous is None else (
            Path(train.get("loaded_checkpoint", "")).resolve() == previous.resolve()
            and train.get("restore_at_load_max_abs") == 0
        ),
        "lr": bool(last_lrs) and all(math.isclose(
            float(value), (8e-4 if str(name).startswith("action") else 8e-5) * factor,
            rel_tol=2e-5, abs_tol=1e-12,
        ) for name, value in last_lrs.items()),
        "restore": restore.get("status") == "PASS_C56B_STRICT_RESTORE"
        and restore.get("restore_max_abs") == 0
        and Path(restore.get("checkpoint", "")).resolve() == checkpoint.resolve(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"C74 s{milestone} gate failed: {','.join(failed)}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_checks = {
        "schema": set(payload) == CHECKPOINT_KEYS and payload.get("schema_version") == 1,
        "step": payload.get("completed_steps") == milestone
        and payload.get("probe_step") == milestone,
        "contract": payload.get("contract") == contract,
        "scheduler": payload.get("lr_scheduler", {}).get("last_epoch") == milestone,
        "probes": isinstance(payload.get("probe_predictions"), list)
        and len(payload["probe_predictions"]) == 32,
    }
    failed = [name for name, passed in checkpoint_checks.items() if not passed]
    if failed:
        raise ValueError(f"C74 s{milestone} checkpoint gate failed: {','.join(failed)}")
    return {
        "milestone": milestone,
        "status": "PASS_C74_MILESTONE_STRICT_RESTORE",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "mean_action_loss": sum(float(row["action_loss"]) for row in history) / segment_steps,
        "minimum_global_block_gradient": min(
            min(row["block_gradient_norms_mean_across_ranks"]) for row in history
        ),
        "gate": {**checks, **{f"checkpoint_{key}": value for key, value in checkpoint_checks.items()}},
    }, contract


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.root.resolve()
    audits, contract = [], None
    for milestone in MILESTONES:
        audit, contract = validate_milestone(root, milestone, contract)
        audits.append(audit)
    final = audits[-1]
    report = {
        "format": READY_FORMAT,
        "status": "PASS_C74_ACTION_ONLY_THREE_EXPERT_EPOCH_COMPLETE",
        "permission": "READY_FOR_PREREGISTERED_OFFLINE_ONLY",
        "effect_status": "NOT_EVIDENCE_READY",
        "completed_steps": 32_647,
        "training_samples": 1_044_704,
        "expert_training_samples": 522_352,
        "incremental_expert_effective_epochs": 522_352 / 200_779,
        "cumulative_expert_effective_epochs_with_c58_parent": (
            80_000 + 522_352
        ) / 200_779,
        "primary_internal_control": audits[MILESTONES.index(7_549)]["checkpoint"],
        "final_checkpoint": final["checkpoint"],
        "final_checkpoint_sha256": final["checkpoint_sha256"],
        "milestone_audits": audits,
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest(),
        "claim_boundary": (
            "Complete fresh-trajectory optimizer and restore evidence only. "
            "C74-s7549 versus C74-s32647 offline and C69 champion paired LIBERO remain unproven."
        ),
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps({
        key: report[key] for key in ("status", "permission", "final_checkpoint")
    }, indent=2))


if __name__ == "__main__":
    main()
