#!/usr/bin/env python3
"""Aggregate the preregistered two-variant, two-seed C31 consequence sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


EXPECTED = {
    ("flattened", 42), ("flattened", 314159),
    ("temporal", 42), ("temporal", 314159),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    reports = []
    for path in sorted(args.root.glob("*/report.json")):
        payload = json.loads(path.read_text())
        key = (str(payload["model_variant"]), int(payload["optimization"]["seed"]))
        reports.append((key, path, payload))
    keys = {key for key, _, _ in reports}
    if keys != EXPECTED or len(reports) != 4:
        raise ValueError(f"incomplete or duplicate C31 sweep: {sorted(keys)}")
    dataset_hashes = {row["source"]["dataset_sha256"] for _, _, row in reports}
    feature_hashes = {row["source"]["features_sha256"] for _, _, row in reports}
    optimization_contracts = {
        (
            row["optimization"]["steps"], row["optimization"]["batch_size"],
            row["optimization"]["learning_rate"], row["optimization"]["weight_decay"],
        )
        for _, _, row in reports
    }
    if len(dataset_hashes) != 1 or len(feature_hashes) != 1 or len(optimization_contracts) != 1:
        raise ValueError("C31 sweep data or optimization contract differs across arms")
    by_variant: dict[str, list[dict]] = defaultdict(list)
    run_rows = []
    for key, path, row in reports:
        metric = row["final_metrics"]["mse"]["conditioned_true"]
        passed = row["status"] == "PASS_C31_ACTION_CONDITIONED_CONSEQUENCE"
        item = {
            "variant": key[0], "seed": key[1], "passed": passed,
            "conditioned_true_mse": metric,
            "gain_over_independent": row["mechanism"]["conditioned_gain_over_independent"],
            "gain_over_shuffled_train": row["mechanism"]["conditioned_gain_over_shuffled_train"],
            "within_state_shuffle_degradation": row["mechanism"]["conditioned_within_state_shuffle_degradation"],
            "report": str(path.resolve()), "report_sha256": sha256_file(path),
            "checkpoint_sha256": row["checkpoint_sha256"],
        }
        by_variant[key[0]].append(item)
        run_rows.append(item)
    summaries = {}
    qualifying = []
    for variant, rows in sorted(by_variant.items()):
        rows.sort(key=lambda row: row["seed"])
        summary = {
            "runs": len(rows),
            "passes": sum(row["passed"] for row in rows),
            "mean_conditioned_true_mse": sum(row["conditioned_true_mse"] for row in rows) / len(rows),
            "minimum_gain_over_independent": min(row["gain_over_independent"] for row in rows),
            "minimum_gain_over_shuffled_train": min(row["gain_over_shuffled_train"] for row in rows),
            "minimum_within_state_shuffle_degradation": min(row["within_state_shuffle_degradation"] for row in rows),
        }
        summaries[variant] = summary
        if summary["passes"] == 2:
            qualifying.append(variant)
    winner = min(
        qualifying,
        key=lambda variant: summaries[variant]["mean_conditioned_true_mse"],
        default=None,
    )
    temporal_beats_flattened_both_seeds = all(
        next(row for row in run_rows if row["variant"] == "temporal" and row["seed"] == seed)["conditioned_true_mse"]
        < next(row for row in run_rows if row["variant"] == "flattened" and row["seed"] == seed)["conditioned_true_mse"]
        for seed in (42, 314159)
    )
    passed = winner is not None
    result = {
        "experiment_id": "h3_c31_action_conditioned_consequence_sweep_v1",
        "status": (
            "PASS_C31_REPEATED_ACTION_CONDITIONED_CONSEQUENCE"
            if passed else "FAIL_C31_REPEATED_ACTION_CONDITIONED_CONSEQUENCE"
        ),
        "permission": "GO_FROZEN_CONSEQUENCE_VALUE_RANKING" if passed else "NO_GO_VALUE_RANKING",
        "claim_boundary": "Fresh-source future-H3 mechanism; no action-ranking or closed-loop success claim.",
        "dataset_sha256": next(iter(dataset_hashes)),
        "features_sha256": next(iter(feature_hashes)),
        "optimization_contract": list(next(iter(optimization_contracts))),
        "runs": sorted(run_rows, key=lambda row: (row["variant"], row["seed"])),
        "variants": summaries,
        "selected_variant": winner,
        "temporal_beats_flattened_both_seeds": temporal_beats_flattened_both_seeds,
        "interpretation": (
            "Temporal alignment is favored only if it wins both matched seeds; otherwise the selected variant is merely the repeated mechanism survivor."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": result["status"], "selected_variant": winner,
        "temporal_beats_flattened_both_seeds": temporal_beats_flattened_both_seeds,
    }))


if __name__ == "__main__":
    main()
