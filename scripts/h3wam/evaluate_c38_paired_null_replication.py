#!/usr/bin/env python3
"""Aggregate four new-seed temporal paired-null consequence replications."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXPECTED_SEEDS = {161803, 271828, 8675309, 20260815}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = []
    for path in sorted(args.root.glob("*/report.json")):
        report = json.loads(path.read_text())
        if report["model_variant"] != "temporal":
            raise ValueError(f"non-temporal C38 report: {path}")
        optimization = report["optimization"]
        rows.append({
            "seed": int(optimization["seed"]),
            "passed": report["status"] == "PASS_C31_ACTION_CONDITIONED_CONSEQUENCE",
            "paired_null_gain": report["mechanism"]["conditioned_gain_over_paired_null"],
            "shuffle_degradation": report["mechanism"]["conditioned_within_state_shuffle_degradation"],
            "shuffled_train_gain": report["mechanism"]["conditioned_gain_over_shuffled_train"],
            "report": str(path.resolve()),
            "report_sha256": sha256_file(path),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "dataset_sha256": report["source"]["dataset_sha256"],
            "features_sha256": report["source"]["features_sha256"],
            "contract": [
                optimization["steps"], optimization["batch_size"],
                optimization["learning_rate"], optimization["weight_decay"],
                optimization["target_error_scaling"],
                optimization["condition_dropout_prob"], optimization["mechanism_gate"],
            ],
        })
    if {row["seed"] for row in rows} != EXPECTED_SEEDS or len(rows) != 4:
        raise ValueError("C38 requires exactly the four preregistered seeds")
    for key in ("dataset_sha256", "features_sha256", "contract"):
        if len({json.dumps(row[key], sort_keys=True) for row in rows}) != 1:
            raise ValueError(f"C38 {key} differs across runs")
    passed = all(row["passed"] for row in rows)
    result = {
        "experiment_id": "h3_c38_temporal_paired_null_replication_v1",
        "status": "PASS_C38_FOUR_SEED_PAIRED_NULL" if passed else "FAIL_C38_FOUR_SEED_PAIRED_NULL",
        "permission": "GO_FRESH_RANKING_VALIDATION" if passed else "NO_GO_FRESH_RANKING_VALIDATION",
        "claim_boundary": "Consumed consequence-validation model selection only; C33 remains the untouched ranking test.",
        "runs": sorted(rows, key=lambda row: row["seed"]),
        "minimum_paired_null_gain": min(row["paired_null_gain"] for row in rows),
        "minimum_shuffle_degradation": min(row["shuffle_degradation"] for row in rows),
        "minimum_shuffled_train_gain": min(row["shuffled_train_gain"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"status": result["status"], "permission": result["permission"]}))


if __name__ == "__main__":
    main()
