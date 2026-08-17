#!/usr/bin/env python3
"""Audit all C70 sampler milestones and publish training-complete evidence only."""

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
READY_FORMAT = "h3wam-c70-sampler-coverage-training-complete-v1"
MILESTONES = tuple(range(1_000, 20_001, 1_000))
C58_SHA256 = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
C67_S20_SHA256 = "9ae1929e7b6ebba303e547727f58e3fd35578b17aa7d4a98da76d0b29ac1272e"
H3_SHA256 = "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
D0_SHA256 = "36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"
C60_DATASET_SHA256 = "1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"
C60_OBSERVATIONS_SHA256 = "b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55"
TARGET_NORM_SHA256 = "95df1f65eba1b1c3bfb9cebea90983ca54dffa69f60e6135354eb67e8551d000"
LAYERS = (
    0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24,
    25, 27, 29, 30, 32, 34, 35, 37, 39, 41, 42, 44, 46, 47, 49,
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


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def require_contract(contract: dict[str, Any]) -> None:
    fixed = {
        "format": FORMAT,
        "classification": "FACT_full_backbone_port_online_frozen_int8_h3",
        "rank_categories": [
            "expert_demo", "expert_demo", "expert_demo", "expert_demo",
            "expert_demo", "expert_demo", "success_rollout",
            "alternating_observational_failure_causal_failure",
        ],
        "rank_schedule": {
            "name": "c70_6_1_half_half",
            "period_steps": 2,
            "odd_step_rank7": "observational_failure",
            "even_step_rank7": "causal_failure",
            "mean_streams_per_step": {
                "expert_demo": 6.0,
                "success_rollout": 1.0,
                "observational_failure": 0.5,
                "causal_failure": 0.5,
            },
        },
        "loss_weights": [10.0, 1.0, 0.4, 0.4],
        "target_norm_sha256": TARGET_NORM_SHA256,
        "h3_sha256": H3_SHA256,
        "d0_sha256": D0_SHA256,
        "c58_parent_sha256": C58_SHA256,
        "causal_failure_dataset_sha256": C60_DATASET_SHA256,
        "causal_failure_observations_sha256": C60_OBSERVATIONS_SHA256,
        "base_lr": 2e-5,
        "action_lr": 2e-4,
        "warmup_steps": 500,
        "scheduler_horizon": 20_000,
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
        for key, expected in fixed.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"C70 fixed contract mismatch: {mismatches}")
    if contract.get("initialization") != {
        "initialization_contract": "strict_online_c58b_parent_v1",
        "c58_completed_steps": 10_000,
    }:
        raise ValueError("C70 did not start from the fixed C58 parent")
    for name in (
        "demo_manifest_sha256", "source_manifest_sha256", "demo_stats_sha256",
        "c48_dataset_sha256", "c48_observations_sha256",
        "c59_completed_sha256", "c59_sample_labels_sha256",
    ):
        value = contract.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"C70 missing immutable data identity: {name}")


def expected_lr_factor(step: int) -> float:
    progress = min(1.0, (step - 500) / (20_000 - 500))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def validate_milestone(
    root: Path, milestone: int, expected_contract: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = (root / f"checkpoints/c70_sampler_s{milestone}.pt").resolve()
    train_path = (root / f"reports/train_s{milestone}.json").resolve()
    restore_path = (root / f"restore/restore_s{milestone}.json").resolve()
    for path in (checkpoint, train_path, restore_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    train, restore = load_json(train_path), load_json(restore_path)
    history = train.get("history", [])
    contract = train.get("contract", {})
    require_contract(contract)
    if expected_contract is not None and contract != expected_contract:
        raise ValueError(f"C70 contract drift at s{milestone}")
    expected_previous = (
        None if milestone == 1_000
        else (root / f"checkpoints/c70_sampler_s{milestone - 1_000}.pt").resolve()
    )
    loaded = train.get("loaded_checkpoint")
    lineage_ok = (
        loaded is None if expected_previous is None
        else Path(loaded).resolve() == expected_previous
    )
    load_restore_ok = (
        train.get("restore_at_load_max_abs") is None if milestone == 1_000
        else train.get("restore_at_load_max_abs") == 0.0
    )
    expected_steps = list(range(milestone - 999, milestone + 1))
    losses = (
        "loss", "action_loss", "future_representation_loss",
        "future_state_loss", "value_loss",
    )
    last_lrs = history[-1].get("learning_rates", {}) if history else {}
    expected_factor = expected_lr_factor(milestone)
    lr_ok = bool(last_lrs) and all(
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and math.isclose(
            float(value),
            (2e-4 if str(name).startswith("action") else 2e-5) * expected_factor,
            rel_tol=2e-5,
            abs_tol=1e-12,
        )
        for name, value in last_lrs.items()
    )
    checks = {
        "train_status": train.get("format") == FORMAT
        and train.get("status") == "PASS_C56B_ONLINE_TRAINING_INVOCATION"
        and train.get("effect_status") == "NOT_EVIDENCE_READY",
        "completed_steps": train.get("completed_steps") == milestone,
        "checkpoint_identity": Path(train.get("checkpoint", "")).resolve() == checkpoint,
        "checkpoint_size": train.get("checkpoint_bytes") == checkpoint.stat().st_size
        and checkpoint.stat().st_size >= MIN_CHECKPOINT_BYTES,
        "exact_segment": len(history) == 1_000
        and [row.get("step") for row in history] == expected_steps,
        "finite_losses": len(history) == 1_000
        and all(
            all(math.isfinite(float(row.get(key, math.nan))) for key in losses)
            for row in history
        ),
        "all_30_gradients": len(history) == 1_000
        and all(
            len(row.get("block_gradient_norms_mean_across_ranks", [])) == 30
            and min(row["block_gradient_norms_mean_across_ranks"]) > 0
            for row in history
        ),
        "future_no_leak": len(history) == 1_000
        and max(row.get("sum_rank_future_leak_abs", math.inf) for row in history) == 0,
        "predecessor_lineage": lineage_ok,
        "predecessor_strict_restore": load_restore_ok,
        "milestone_lr": lr_ok,
        "restore_status": restore.get("format") == FORMAT
        and restore.get("status") == "PASS_C56B_STRICT_RESTORE",
        "strict_restore": restore.get("restore_max_abs") == 0.0,
        "restore_checkpoint": Path(restore.get("checkpoint", "")).resolve() == checkpoint
        and Path(restore.get("loaded_checkpoint", "")).resolve() == checkpoint,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"C70 s{milestone} gate failed: {','.join(failed)}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_checks = {
        "schema": set(payload) == CHECKPOINT_KEYS and payload.get("schema_version") == 1,
        "step": payload.get("completed_steps") == milestone
        and payload.get("probe_step") == milestone,
        "contract": canonical(payload.get("contract")) == contract,
        "state": isinstance(payload.get("model"), dict) and bool(payload["model"])
        and isinstance(payload.get("optimizer"), dict)
        and isinstance(payload.get("lr_scheduler"), dict),
        "scheduler_step": payload.get("lr_scheduler", {}).get("last_epoch") == milestone,
        "probes": isinstance(payload.get("probe_predictions"), list)
        and len(payload["probe_predictions"]) == 8,
    }
    failed = sorted(name for name, passed in checkpoint_checks.items() if not passed)
    if failed:
        raise ValueError(f"C70 s{milestone} checkpoint gate failed: {','.join(failed)}")
    audit = {
        "format": "h3wam-c70-sampler-milestone-restore-audit-v1",
        "status": "PASS_C70_SAMPLER_MILESTONE_STRICT_RESTORE",
        "effect_status": "NOT_EVIDENCE_READY",
        "milestone": milestone,
        "training_samples": milestone * 8,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "restore_max_abs": 0.0,
        "loaded_checkpoint": loaded,
        "mean_loss": sum(float(row["loss"]) for row in history) / 1_000,
        "mean_action_loss": sum(float(row["action_loss"]) for row in history) / 1_000,
        "minimum_block_gradient": min(
            min(row["block_gradient_norms_mean_across_ranks"]) for row in history
        ),
        "lr_factor": expected_factor,
        "learning_rates": last_lrs,
        "gate": {**checks, **{f"checkpoint_{key}": value for key, value in checkpoint_checks.items()}},
    }
    return audit, contract


def validate_c58_ready(path: Path) -> dict[str, Any]:
    ready = load_json(path.resolve())
    checks = {
        "status": ready.get("status") == "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE",
        "permission": ready.get("permission") == "READY_FOR_CHILD_BRANCH_AND_LIBERO_EVAL",
        "steps": ready.get("completed_steps") == 10_000,
        "sha256": ready.get("checkpoint_sha256") == C58_SHA256,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("C70 C58 READY mismatch: " + ",".join(failed))
    return ready


def finalize(
    root: Path, c58_ready_path: Path, c67_control_path: Path
) -> dict[str, Any]:
    root = root.resolve()
    validate_c58_ready(c58_ready_path)
    c67_control_path = c67_control_path.resolve()
    if sha256_file(c67_control_path) != C67_S20_SHA256:
        raise ValueError("C70 fixed C67-s20 sampler control mismatch")
    audits: list[dict[str, Any]] = []
    contract = None
    for milestone in MILESTONES:
        audit, contract = validate_milestone(root, milestone, contract)
        audits.append(audit)
    assert contract is not None
    s20 = root / "checkpoints/c70_sampler_s20000.pt"
    return {
        "format": READY_FORMAT,
        "status": "PASS_C70_SAMPLER_TRAINING_COMPLETE",
        "permission": "READY_FOR_PREREGISTERED_OFFLINE_ONLY",
        "effect_status": "NOT_EVIDENCE_READY",
        "world_size": 8,
        "global_batch": 8,
        "completed_steps": 20_000,
        "training_samples": 160_000,
        "unique_windows": 218_125,
        "effective_epochs": 0.733522,
        "scheduler": {
            "warmup_steps": 500,
            "horizon": 20_000,
            "s10000_factor": expected_lr_factor(10_000),
            "s20000_factor": expected_lr_factor(20_000),
        },
        "sampler_exposure": {
            "expert_samples": 120_000,
            "success_samples": 20_000,
            "observational_failure_samples": 10_000,
            "causal_failure_samples": 10_000,
        },
        "matched_control": {
            "variant": "C67_4_2_1_1",
            "milestone": 20_000,
            "checkpoint": str(c67_control_path),
            "checkpoint_sha256": C67_S20_SHA256,
        },
        "candidate": {
            "variant": "C70_6_1_half_half",
            "milestone": 20_000,
            "training_samples": 160_000,
            "effective_epochs": 0.733522,
            "checkpoint": str(s20.resolve()),
            "checkpoint_sha256": sha256_file(s20),
        },
        "milestone_audits": audits,
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "claim_boundary": (
            "Proves the complete C70 6/1/0.5/0.5 sampler trajectory and strict "
            "restore only. Offline and paired LIBERO effects against fixed C67-s20 "
            "remain unproven."
        ),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--c58-ready", type=Path, required=True)
    parser.add_argument("--c67-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing C70 completion marker: {args.output}")
    report = finalize(args.root, args.c58_ready, args.c67_control)
    audit_root = args.root.resolve() / "milestone-audit"
    for audit in report["milestone_audits"]:
        output = audit_root / f"s{audit['milestone']}.json"
        if output.exists():
            raise FileExistsError(f"refusing existing C70 audit: {output}")
    for audit in report["milestone_audits"]:
        atomic_json(audit_root / f"s{audit['milestone']}.json", audit)
    atomic_json(args.output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "permission": report["permission"],
        "matched_control": report["matched_control"],
        "candidate": report["candidate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
