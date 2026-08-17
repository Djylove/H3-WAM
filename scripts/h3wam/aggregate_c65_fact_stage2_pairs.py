#!/usr/bin/env python3
"""Aggregate the pre-registered C65 four-suite Stage-2 ranking score."""

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


FORMAT = "h3wam-c65-fact-stage2-cross-suite-result-v1"
SHARD_FORMAT = "h3wam-c65-fact-stage2-cross-suite-shard-v1"
SHARD_STATUSES = {
    "PASS_C65_STAGE2_SHARD_MECHANICS",
    "FAIL_C65_STAGE2_SHARD_MECHANICS",
}
DATA_GATE_FORMAT = "h3wam-c65-c60-deployment-pair-data-gate-v1"
PAIR_FORMAT = "h3wam-c65-c60-deployment-pairs-v1"
C60_SHA256 = "d6659c6b387f062a99f670a1d902b56df71a6bf1472aa4e46e56c9213ba75a36"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EXPECTED_SUITE_COUNTS = {suite: 20 for suite in SUITES}
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
    if trials <= 0:
        return 1.0
    return sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (2**trials)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 80 or {int(row["pair_index"]) for row in rows} != set(range(80)):
        raise ValueError("C65 aggregate requires exact pair indices 0..79")
    suite_counts = dict(sorted(Counter(str(row["suite"]) for row in rows).items()))
    if suite_counts != dict(sorted(EXPECTED_SUITE_COUNTS.items())):
        raise ValueError("C65 aggregate suite identity drifted")
    source_keys = {(str(row["suite"]), int(row["source_id"])) for row in rows}
    if len(source_keys) != 80:
        raise ValueError("C65 aggregate requires 80 source-independent pairs")

    mechanics = {
        "all_scores_finite": all(bool(row["score_finite"]) for row in rows),
        "all_candidate_order_invariance": all(
            bool(row["order_invariance_pass"]) for row in rows
        ),
        "all_identity_gates": all(bool(row["identity_pass"]) for row in rows),
        "all_candidate_actions_distinct": all(
            row["success_action_sha256"] != row["failure_action_sha256"] for row in rows
        ),
    }
    nonties = [row for row in rows if not bool(row["tie"])]
    successes = sum(bool(row["success_preferred"]) for row in nonties)
    margins = [float(row["failure_minus_success"]) for row in rows]
    by_suite: dict[str, dict[str, Any]] = {}
    for suite in SUITES:
        selected = [row for row in rows if row["suite"] == suite]
        selected_nonties = [row for row in selected if not bool(row["tie"])]
        wins = sum(bool(row["success_preferred"]) for row in selected_nonties)
        by_suite[suite] = {
            "pairs": 20,
            "non_ties": len(selected_nonties),
            "ties": 20 - len(selected_nonties),
            "success_preferred": wins,
            "conditional_preference": wins / len(selected_nonties) if selected_nonties else 0.0,
            "median_failure_minus_success": statistics.median(
                float(row["failure_minus_success"]) for row in selected
            ),
        }
    conditional_preference = successes / len(nonties) if nonties else 0.0
    p_value = exact_binomial_greater(successes, len(nonties))
    gates = {
        "mechanics": all(mechanics.values()),
        "overall_non_tie_coverage_at_least_76_of_80": len(nonties) >= 76,
        "every_suite_non_tie_coverage_at_least_19_of_20": all(
            result["non_ties"] >= 19 for result in by_suite.values()
        ),
        "overall_conditional_preference_at_least_0_65": conditional_preference >= 0.65,
        "one_sided_exact_binomial_p_le_0_05": p_value <= 0.05,
        "every_suite_conditional_preference_at_least_0_60": all(
            result["conditional_preference"] >= 0.60 for result in by_suite.values()
        ),
        "median_failure_minus_success_gt_0": statistics.median(margins) > 0.0,
    }
    passed = all(gates.values())
    return {
        "status": (
            "PASS_C65_FACT_STAGE2_CROSS_SUITE_CONFIRMATORY"
            if passed else "FAIL_C65_FACT_STAGE2_CROSS_SUITE_CONFIRMATORY"
        ),
        "permission": (
            "GO_SEPARATE_PREREGISTERED_N1_VS_N4_CLOSED_LOOP_ONLY"
            if passed else "NO_GO_C60_STAGE2_SELECTOR_KEEP_C58"
        ),
        "effect_status": (
            "OFFLINE_ACTION_RANKING_ONLY_NOT_CLOSED_LOOP"
            if passed else "NOT_EVIDENCE_READY"
        ),
        "pairs": 80,
        "non_ties": len(nonties),
        "ties": 80 - len(nonties),
        "success_preferred": successes,
        "conditional_preference": conditional_preference,
        "one_sided_exact_binomial_p": p_value,
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
    parser.add_argument("--data-gate", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data_gate_path = args.data_gate.resolve()
    pair_path = args.pairs.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing C65 aggregate: {output}")
    data_gate = json.loads(data_gate_path.read_text())
    pairs = json.loads(pair_path.read_text())
    pair_sha = sha256_file(pair_path)
    data_gate_sha = sha256_file(data_gate_path)
    if (
        data_gate.get("format") != DATA_GATE_FORMAT
        or data_gate.get("status") != "PASS_C65_FOUR_SUITE_PAIR_DATA_GATE"
        or data_gate.get("permission") != "GO_SCORE_C65"
        or data_gate.get("pairs_sha256") != pair_sha
        or pairs.get("format") != PAIR_FORMAT
        or pairs.get("pair_count") != 80
        or pairs.get("suite_counts") != EXPECTED_SUITE_COUNTS
        or pairs.get("checkpoint_sha256") != C60_SHA256
    ):
        raise ValueError("C65 aggregate data/pair gate failed")

    rows, shard_files = [], []
    expected_model_inputs = [
        "current_agentview", "current_wristview", "current_proprio",
        "task_language", "candidate_action",
    ]
    expected_forbidden = [
        "future_observation", "terminal_state", "outcome", "success_label",
    ]
    for shard in range(8):
        path = root / "shards" / f"shard{shard}.json"
        report = json.loads(path.read_text())
        if (
            report.get("format") != SHARD_FORMAT
            or report.get("status") not in SHARD_STATUSES
            or report.get("effect_status") != "SHARD_ONLY_NOT_INTERPRETABLE"
            or report.get("shard") != shard
            or report.get("num_shards") != 8
            or report.get("data_gate_sha256") != data_gate_sha
            or report.get("pair_manifest_sha256") != pair_sha
            or report.get("checkpoint_sha256") != C60_SHA256
            or report.get("solver") != {"inference_steps": 10, "flow_shift": 5.0}
            or report.get("value_contract") != VALUE_CONTRACT
            or report.get("same_noise_within_pair") is not True
            or report.get("candidate_order_repeated") is not True
            or report.get("model_input_fields") != expected_model_inputs
            or report.get("forbidden_model_input_fields") != expected_forbidden
            or len(report.get("rows", [])) != 10
        ):
            raise ValueError(f"C65 shard contract failed: {shard}")
        if {int(row["pair_index"]) % 8 for row in report["rows"]} != {shard}:
            raise ValueError(f"C65 shard ownership failed: {shard}")
        rows.extend(report["rows"])
        shard_files.append({"shard": shard, "path": str(path), "sha256": sha256_file(path)})

    result = {
        "format": FORMAT,
        **aggregate_rows(rows),
        "candidate": "C65_FACT_STAGE2_CROSS_SUITE_CONFIRMATORY",
        "checkpoint_sha256": C60_SHA256,
        "data_gate": str(data_gate_path),
        "data_gate_sha256": data_gate_sha,
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": pair_sha,
        "shards": shard_files,
        "rows": sorted(rows, key=lambda row: row["pair_index"]),
        "claim_boundary": (
            "A PASS establishes only frozen C60 Stage-2 ranking on 80 fresh, "
            "source-independent, four-suite pairs. It permits a separate preregistered "
            "N=1 versus N=4 closed-loop test; it does not promote C60 or establish LIBERO gain."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({key: value for key, value in result.items() if key not in {"rows", "shards"}}, indent=2))


if __name__ == "__main__":
    main()
