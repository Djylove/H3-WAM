#!/usr/bin/env python3
"""Continuously audit every C56b 1k checkpoint and independent restore."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


FORMAT = "h3wam-c56b-fact-online-training-v1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_milestone(
    root: Path,
    milestone: int,
    *,
    parent_sha256: str,
    causal_dataset_sha256: str,
    causal_observations_sha256: str,
) -> dict[str, Any]:
    root = root.resolve()
    checkpoint = (root / f"checkpoints/c56b_online_s{milestone}.pt").resolve()
    train_path = (root / f"reports/train_s{milestone}.json").resolve()
    restore_path = (root / f"restore/restore_s{milestone}.json").resolve()
    for path in (checkpoint, train_path, restore_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    train, restore = load_json(train_path), load_json(restore_path)
    history = train.get("history", [])
    contract = train.get("contract", {})
    expected_steps = list(range(milestone - 999, milestone + 1))
    losses = (
        "loss", "action_loss", "future_representation_loss",
        "future_state_loss", "value_loss",
    )
    checks = {
        "train_status": train.get("format") == FORMAT
        and train.get("status") == "PASS_C56B_ONLINE_TRAINING_INVOCATION",
        "completed_steps": train.get("completed_steps") == milestone,
        "checkpoint_identity": Path(train.get("checkpoint", "")).resolve() == checkpoint,
        "checkpoint_size": train.get("checkpoint_bytes") == checkpoint.stat().st_size,
        "exact_segment": len(history) == 1000
        and [row.get("step") for row in history] == expected_steps,
        "finite_losses": len(history) == 1000
        and all(
            all(math.isfinite(float(row.get(key, math.nan))) for key in losses)
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
        "restore_status": restore.get("format") == FORMAT
        and restore.get("status") == "PASS_C56B_STRICT_RESTORE",
        "strict_restore": restore.get("restore_max_abs") == 0.0,
        "restore_checkpoint": Path(restore.get("checkpoint", "")).resolve() == checkpoint,
        "parent": contract.get("c58_parent_sha256") == parent_sha256,
        "causal_dataset": contract.get("causal_failure_dataset_sha256")
        == causal_dataset_sha256,
        "causal_observations": contract.get("causal_failure_observations_sha256")
        == causal_observations_sha256,
        "online_no_cache": contract.get("h3_execution")
        == "online_frozen_int8_per_rank_v1"
        and contract.get("no_kv_cache") is True,
        "fixed_optimization": contract.get("base_lr") == 2e-5
        and contract.get("action_lr") == 2e-4
        and contract.get("warmup_steps") == 500
        and contract.get("scheduler_horizon") == 10000
        and contract.get("seed") == 20260816,
    }
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise ValueError(f"C56b s{milestone} milestone gate failed: {','.join(failed)}")
    return {
        "format": "h3wam-c56b-milestone-restore-audit-v1",
        "status": "PASS_C56B_MILESTONE_STRICT_RESTORE",
        "effect_status": "NOT_EVIDENCE_READY",
        "milestone": milestone,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "restore_max_abs": 0.0,
        "mean_loss": sum(float(row["loss"]) for row in history) / len(history),
        "mean_action_loss": sum(float(row["action_loss"]) for row in history) / len(history),
        "minimum_block_gradient": min(
            min(row["block_gradient_norms_mean_across_ranks"]) for row in history
        ),
        "parent_sha256": parent_sha256,
        "causal_failure_dataset_sha256": causal_dataset_sha256,
        "causal_failure_observations_sha256": causal_observations_sha256,
        "gate": checks,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--causal-dataset-sha256", required=True)
    parser.add_argument("--causal-observations-sha256", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        raise ValueError("poll-seconds must be positive")
    root = args.root.resolve()
    audit_root = root / "milestone-audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    try:
        reports = []
        for milestone in range(1000, 10001, 1000):
            output = audit_root / f"s{milestone}.json"
            if output.is_file():
                report = load_json(output)
                if report.get("status") != "PASS_C56B_MILESTONE_STRICT_RESTORE":
                    raise ValueError(f"invalid pre-existing audit: {output}")
            else:
                restore = root / f"restore/restore_s{milestone}.json"
                while not restore.is_file():
                    time.sleep(args.poll_seconds)
                report = validate_milestone(
                    root,
                    milestone,
                    parent_sha256=args.parent_sha256,
                    causal_dataset_sha256=args.causal_dataset_sha256,
                    causal_observations_sha256=args.causal_observations_sha256,
                )
                atomic_json(output, report)
            reports.append(report)
            print(json.dumps(report, sort_keys=True), flush=True)
        atomic_json(audit_root / "READY.json", {
            "format": "h3wam-c56b-all-milestone-restores-v1",
            "status": "PASS_ALL_10_C56B_MILESTONE_RESTORES",
            "milestones": [report["milestone"] for report in reports],
            "effect_status": "NOT_EVIDENCE_READY",
        })
    except BaseException as exc:
        atomic_json(audit_root / "FAILED.json", {
            "status": "FAILED_C56B_MILESTONE_AUDIT",
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


if __name__ == "__main__":
    main()
