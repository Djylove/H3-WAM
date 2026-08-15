#!/usr/bin/env python3
"""Audit C23 first-action identity, continuation seeds, and outcomes."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prereg = json.loads((args.root / "preregistration.json").read_text())
    selections = [json.loads(line) for line in (args.root / "selection.jsonl").read_text().splitlines()]
    if len(selections) != prereg["branches"] or prereg["branches"] != 32:
        raise ValueError("C23 branch count mismatch")
    groups = defaultdict(list)
    for item in selections:
        slug = item["suite"].removeprefix("libero_")
        name = (
            f"{item['ordinal']}_{slug}_task{item['task']}_trial{item['trial']}_"
            f"d{item['distance_replans']}_offset{item['noise_offset']}"
        )
        path = args.root / "runs" / name / "results.json"
        payload = json.loads(path.read_text())
        episode = payload["tasks"][0]["episodes"][0]
        reference_payload = json.loads(Path(item["c22_reference_result"]).read_text())
        reference = reference_payload["tasks"][0]["episodes"][0]
        first = np.asarray(episode["first_environment_action_chunk"], dtype=np.float32)
        reference_first = np.asarray(reference["first_environment_action_chunk"], dtype=np.float32)
        seeds = [int(seed) for seed in episode["replan_noise_seeds"]]
        expected_seeds = [item["first_policy_noise_seed"]] + [
            item["continuation_policy_noise_seed_base"] + index
            for index in range(max(0, len(seeds) - 1))
        ]
        if payload["first_policy_noise_seed"] != item["first_policy_noise_seed"]:
            raise ValueError(f"first seed metadata mismatch: {path}")
        if payload["continuation_policy_noise_seed_base"] != item["continuation_policy_noise_seed_base"]:
            raise ValueError(f"continuation seed metadata mismatch: {path}")
        key = (item["suite"], item["task"], item["trial"], item["distance_replans"], item["index"])
        groups[key].append(
            {
                "ordinal": item["ordinal"], "noise_offset": item["noise_offset"],
                "first_policy_noise_seed": item["first_policy_noise_seed"],
                "continuation_policy_noise_seed_base": item["continuation_policy_noise_seed_base"],
                "first_action_chunk": first,
                "first_chunk_exact_to_c22": bool(np.array_equal(first, reference_first)),
                "seed_schedule_valid": seeds == expected_seeds,
                "replan_noise_seeds": seeds,
                "success": bool(episode["success"]), "final_step": int(episode["steps"]),
                "replans": int(episode["replans"]), "result": str(path),
                "c22_reference_result": item["c22_reference_result"],
            }
        )
    reports = []
    for key, items in sorted(groups.items()):
        if len(items) != 4:
            raise ValueError(f"invalid C23 group size: {key}")
        pair_rms = [
            float(np.sqrt(np.mean((left["first_action_chunk"] - right["first_action_chunk"]) ** 2)))
            for left, right in itertools.combinations(items, 2)
        ]
        successes = sum(item["success"] for item in items)
        reports.append(
            {
                "suite": key[0], "task": key[1], "trial": key[2],
                "distance_replans": key[3], "index": key[4],
                "branches": 4, "successes": successes,
                "mixed_outcomes": 0 < successes < 4,
                "all_first_chunks_exact_to_c22": all(item["first_chunk_exact_to_c22"] for item in items),
                "all_seed_schedules_valid": all(item["seed_schedule_valid"] for item in items),
                "min_pairwise_first_chunk_rms": min(pair_rms),
                "max_pairwise_first_chunk_rms": max(pair_rms),
                "branches_detail": [
                    {field: value for field, value in item.items() if field != "first_action_chunk"}
                    for item in items
                ],
            }
        )
    if len(reports) != prereg["groups"] or prereg["groups"] != 8:
        raise ValueError("C23 group count mismatch")
    first_exact = all(row["all_first_chunks_exact_to_c22"] for row in reports)
    schedules_valid = all(row["all_seed_schedules_valid"] for row in reports)
    action_diverse = all(row["min_pairwise_first_chunk_rms"] > 1e-6 for row in reports)
    mixed_groups = sum(row["mixed_outcomes"] for row in reports)
    passed = first_exact and schedules_valid and action_diverse and mixed_groups >= 1
    report = {
        **prereg,
        "successful_branches": sum(row["successes"] for row in reports),
        "mixed_outcome_groups": mixed_groups,
        "all_first_chunks_exact_to_c22": first_exact,
        "all_continuation_seed_schedules_valid": schedules_valid,
        "all_groups_action_diverse": action_diverse,
        "group_reports": reports,
        "status": "PASS_FIRST_ACTION_CAUSAL_CANARY" if passed else "FAIL_FIRST_ACTION_CAUSAL_CANARY",
        "data_permission": prereg["pass_permission"] if passed else prereg["fail_permission"],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({
        key: report[key] for key in (
            "status", "branches", "successful_branches", "mixed_outcome_groups",
            "all_first_chunks_exact_to_c22", "all_continuation_seed_schedules_valid",
            "all_groups_action_diverse", "data_permission", "effect_conclusion",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
