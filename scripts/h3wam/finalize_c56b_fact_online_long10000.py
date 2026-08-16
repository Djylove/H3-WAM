#!/usr/bin/env python3
"""Publish a C56b s10000 endpoint after strict restore and identity audit."""

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
C60_DATASET_SHA256 = "1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"
C60_OBSERVATIONS_SHA256 = "b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55"
TARGET_NORM_SHA256 = "95df1f65eba1b1c3bfb9cebea90983ca54dffa69f60e6135354eb67e8551d000"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def json_canonical(value: Any) -> Any:
    """Match torch checkpoint containers to their lossless JSON form."""

    if isinstance(value, dict):
        return {str(key): json_canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_canonical(item) for item in value]
    return value


def expected_causal(causal_ready: Path | None) -> tuple[str, str, str]:
    if causal_ready is None:
        return "C60_MAIN", C60_DATASET_SHA256, C60_OBSERVATIONS_SHA256
    completed = load_json(causal_ready.resolve())
    if (
        completed.get("format") != "h3wam-c61-finalized-fact-failure-dataset-v1"
        or completed.get("status") != "PASS_C61_FINALIZED_FACT_FAILURE_DATASET"
        or not completed.get("gates")
        or any(value != "PASS" for value in completed["gates"].values())
    ):
        raise ValueError("C61 causal READY contract mismatch")
    return (
        "C61_MATCHED",
        str(completed["dataset_sha256"]),
        str(completed["observations_sha256"]),
    )


def finalize(root: Path, c58_ready_path: Path, causal_ready: Path | None) -> dict[str, Any]:
    root = root.resolve()
    checkpoint = (root / "checkpoints/c56b_online_s10000.pt").resolve()
    train_path = (root / "reports/train_s10000.json").resolve()
    restore_path = (root / "restore/restore_s10000.json").resolve()
    for path in (checkpoint, train_path, restore_path, c58_ready_path.resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)
    train, restore = load_json(train_path), load_json(restore_path)
    c58_ready = load_json(c58_ready_path.resolve())
    arm, causal_dataset_sha, causal_observations_sha = expected_causal(causal_ready)
    history = train.get("history", [])
    contract = train.get("contract", {})
    checks = {
        "train_status": train.get("status") == "PASS_C56B_ONLINE_TRAINING_INVOCATION",
        "train_effect_boundary": train.get("effect_status") == "NOT_EVIDENCE_READY",
        "completed_steps": train.get("completed_steps") == 10000,
        "checkpoint_identity": Path(train.get("checkpoint", "")).resolve() == checkpoint,
        "checkpoint_size": int(train.get("checkpoint_bytes", -1)) == checkpoint.stat().st_size,
        "final_segment": len(history) == 1000
        and [row.get("step") for row in history] == list(range(9001, 10001)),
        "finite_losses": len(history) == 1000
        and all(
            all(math.isfinite(float(row.get(key, math.nan))) for key in (
                "loss", "action_loss", "future_representation_loss",
                "future_state_loss", "value_loss",
            ))
            for row in history
        ),
        "all_30_gradients": len(history) == 1000
        and all(
            len(row.get("block_gradient_norms_mean_across_ranks", [])) == 30
            and min(row["block_gradient_norms_mean_across_ranks"]) > 0
            for row in history
        ),
        "future_no_leak": len(history) == 1000
        and max(row.get("sum_rank_future_leak_abs", math.inf) for row in history) == 0,
        "restore_status": restore.get("status") == "PASS_C56B_STRICT_RESTORE",
        "strict_restore": restore.get("restore_max_abs") == 0.0,
        "restore_checkpoint": Path(restore.get("checkpoint", "")).resolve() == checkpoint,
        "contract_format": contract.get("format") == FORMAT,
        "classification": contract.get("classification")
        == "FACT_full_backbone_port_online_frozen_int8_h3",
        "rank_mixture": contract.get("rank_categories") == [
            "expert_demo", "expert_demo", "expert_demo", "expert_demo",
            "success_rollout", "success_rollout", "observational_failure",
            "causal_failure",
        ],
        "loss_weights": contract.get("loss_weights") == [10.0, 1.0, 0.4, 0.4],
        "target_norm": contract.get("target_norm_sha256") == TARGET_NORM_SHA256,
        "optimizer": contract.get("base_lr") == 2e-5
        and contract.get("action_lr") == 2e-4
        and contract.get("warmup_steps") == 500
        and contract.get("scheduler_horizon") == 10000
        and contract.get("weight_decay") == 1e-4
        and contract.get("max_grad_norm") == 1.0
        and contract.get("seed") == 20260816,
        "shared_data_identity": all(
            isinstance(contract.get(key), str)
            and len(contract[key]) == 64
            for key in (
                "demo_manifest_sha256", "source_manifest_sha256",
                "demo_stats_sha256", "c48_dataset_sha256",
                "c48_observations_sha256", "c59_completed_sha256",
                "c59_sample_labels_sha256",
            )
        ),
        "online_no_cache": contract.get("no_kv_cache") is True
        and contract.get("h3_execution") == "online_frozen_int8_per_rank_v1"
        and contract.get("action_horizon") == 32
        and contract.get("action_shift") == 5.0
        and contract.get("h3_carrier_layers") == [
            0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24,
            25, 27, 29, 30, 32, 34, 35, 37, 39, 41, 42, 44, 46, 47, 49,
        ],
        "c58_ready": c58_ready.get("status")
        == "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE"
        and c58_ready.get("completed_steps") == 10000,
        "c58_parent": contract.get("c58_parent_sha256")
        == c58_ready.get("checkpoint_sha256"),
        "causal_dataset": contract.get("causal_failure_dataset_sha256")
        == causal_dataset_sha,
        "causal_observations": contract.get("causal_failure_observations_sha256")
        == causal_observations_sha,
    }
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise ValueError("C56b s10000 final gate failed: " + ",".join(failed))

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    expected_keys = {
        "schema_version", "completed_steps", "model", "optimizer",
        "lr_scheduler", "contract", "probe_step", "probe_predictions",
    }
    if set(payload) != expected_keys or payload.get("schema_version") != 1:
        raise ValueError("C56b checkpoint schema mismatch")
    if payload.get("completed_steps") != 10000 or payload.get("probe_step") != 10000:
        raise ValueError("C56b checkpoint milestone mismatch")
    if json_canonical(payload.get("contract")) != contract or not payload.get("model"):
        raise ValueError("C56b checkpoint/report contract mismatch")
    if not isinstance(payload.get("probe_predictions"), list) or len(payload["probe_predictions"]) != 8:
        raise ValueError("C56b checkpoint lacks eight restore probes")
    return {
        "format": "h3wam-c56b-fact-online-long10000-ready-v1",
        "status": "PASS_C56B_ONLINE_LONG10000_STRICT_RESTORE",
        "permission": "READY_FOR_PAIRED_HELDOUT",
        "effect_status": "NOT_EVIDENCE_READY",
        "arm": arm,
        "completed_steps": 10000,
        "world_size": 8,
        "global_batch": 8,
        "training_samples": 80000,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_report": str(train_path),
        "train_report_sha256": sha256_file(train_path),
        "restore_report": str(restore_path),
        "restore_report_sha256": sha256_file(restore_path),
        "c58_parent_sha256": contract["c58_parent_sha256"],
        "causal_failure_dataset_sha256": causal_dataset_sha,
        "causal_failure_observations_sha256": causal_observations_sha,
        "gate": checks,
        "claim_boundary": "Completed training and strict restore only; held-out and LIBERO effect remain unproven.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--c58-ready", type=Path, required=True)
    parser.add_argument("--causal-ready", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing C56b READY: {args.output}")
    report = finalize(args.root, args.c58_ready, args.causal_ready)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
