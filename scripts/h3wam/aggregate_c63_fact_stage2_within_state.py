#!/usr/bin/env python3
"""Aggregate the pre-registered C63 32-pair Stage-2 ranking diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


FORMAT = "h3wam-c63-fact-stage2-within-state-result-v1"
SHARD_FORMAT = "h3wam-c63-fact-stage2-within-state-shard-v1"
PAIR_FORMAT = "h3wam-c63-fact-stage2-within-state-pairs-v1"
C60_SHA256 = "d6659c6b387f062a99f670a1d902b56df71a6bf1472aa4e46e56c9213ba75a36"
EXPECTED_SUITE_COUNTS = {"libero_object": 2, "libero_spatial": 30}
VALUE_CONTRACT = {
    "model_domain": "normalized_minus1_to1",
    "denormalization": "raw_equals_normalized_plus1",
    "raw_range": [0.0, 2.0],
    "ranking": "argmin_first_and_only_value_token",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact_binomial_greater(successes: int, trials: int) -> float:
    return sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (2**trials)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 32 or {int(row["pair_index"]) for row in rows} != set(range(32)):
        raise ValueError("C63 aggregate requires exact pair indices 0..31")
    suite_counts = dict(sorted(Counter(str(row["suite"]) for row in rows).items()))
    if suite_counts != EXPECTED_SUITE_COUNTS:
        raise ValueError("C63 aggregate suite identity drifted")
    successes = sum(bool(row["success_preferred"]) for row in rows)
    margins = [float(row["failure_minus_success"]) for row in rows]
    by_suite = {}
    for suite in EXPECTED_SUITE_COUNTS:
        selected = [row for row in rows if row["suite"] == suite]
        by_suite[suite] = {
            "success_preferred": sum(bool(row["success_preferred"]) for row in selected),
            "pairs": len(selected),
            "median_failure_minus_success": statistics.median(
                float(row["failure_minus_success"]) for row in selected
            ),
        }
    mechanics = {
        "all_scores_finite": all(bool(row["score_finite"]) for row in rows),
        "all_action_conditioned_value_deltas_nonzero": all(
            bool(row["action_conditioned_value_delta_nonzero"]) for row in rows
        ),
        "all_candidate_order_invariance": all(bool(row["order_invariance_pass"]) for row in rows),
    }
    gates = {
        "mechanics": all(mechanics.values()),
        "primary_at_least_22_of_32": successes >= 22,
        "primary_one_sided_exact_binomial_p_le_0_05": exact_binomial_greater(successes, 32) <= 0.05,
        "median_failure_minus_success_gt_0": statistics.median(margins) > 0.0,
        "spatial_at_least_20_of_30": by_suite["libero_spatial"]["success_preferred"] >= 20,
        "object_at_least_1_of_2": by_suite["libero_object"]["success_preferred"] >= 1,
    }
    passed = all(gates.values())
    return {
        "status": "PASS_C63_STAGE2_WITHIN_STATE_DIAGNOSTIC" if passed else "FAIL_C63_STAGE2_WITHIN_STATE_DIAGNOSTIC",
        "permission": "GO_COLLECT_CROSS_SUITE_C63_CONFIRMATORY_PAIRS" if passed else "NO_GO_C60_STAGE2_RANKING_KEEP_C58",
        "effect_status": "OFFLINE_SUITE_IMBALANCED_NOT_PROMOTION_EVIDENCE",
        "success_preferred": successes,
        "pairs": 32,
        "one_sided_exact_binomial_p": exact_binomial_greater(successes, 32),
        "median_failure_minus_success": statistics.median(margins),
        "mean_failure_minus_success": statistics.fmean(margins),
        "suite_counts": suite_counts,
        "by_suite": by_suite,
        "mechanics": mechanics,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, pair_path, output = args.root.resolve(), args.pairs.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing C63 aggregate: {output}")
    pair_manifest = json.loads(pair_path.read_text())
    pair_sha = sha256_file(pair_path)
    if pair_manifest.get("format") != PAIR_FORMAT or pair_manifest.get("pair_count") != 32:
        raise ValueError("C63 aggregate pair manifest gate failed")
    rows, shard_files = [], []
    for shard in range(8):
        path = root / "shards" / f"shard{shard}.json"
        report = json.loads(path.read_text())
        if (
            report.get("format") != SHARD_FORMAT
            or report.get("status") != "PASS_C63_STAGE2_SHARD_MECHANICS"
            or report.get("effect_status") != "SHARD_ONLY_NOT_INTERPRETABLE"
            or report.get("shard") != shard
            or report.get("num_shards") != 8
            or report.get("pair_manifest_sha256") != pair_sha
            or report.get("checkpoint_sha256") != C60_SHA256
            or report.get("solver") != {"inference_steps": 10, "flow_shift": 5.0}
            or report.get("value_contract") != VALUE_CONTRACT
            or report.get("same_noise_within_pair") is not True
            or report.get("candidate_order_repeated") is not True
        ):
            raise ValueError(f"C63 shard contract failed: {shard}")
        rows.extend(report["rows"])
        shard_files.append({"shard": shard, "sha256": sha256_file(path), "path": str(path)})
    result = {
        "format": FORMAT,
        **aggregate_rows(rows),
        "candidate": "C63_FACT_STAGE2_WITHIN_STATE_DIAGNOSTIC",
        "checkpoint_sha256": C60_SHA256,
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": pair_sha,
        "pre_forward_erratum": pair_manifest["pre_forward_erratum_observation_drift"],
        "shards": shard_files,
        "rows": sorted(rows, key=lambda row: row["pair_index"]),
        "claim_boundary": "Offline, 30/32 spatial and 2/32 object diagnostic only. PASS permits balanced cross-suite pair collection, not BoN deployment, training continuation, or C60 promotion.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "shards"}}, indent=2))


if __name__ == "__main__":
    main()
