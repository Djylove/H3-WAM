#!/usr/bin/env python3
"""Evaluate preregistered C22 action diversity and outcome entropy gates."""

from __future__ import annotations

import argparse
import hashlib
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
    if len(selections) != prereg["branches"] or prereg["branches"] != 96:
        raise ValueError("C22 selection/preregistration branch count mismatch")
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
        branch = episode["branch_start"]
        if branch["index"] != item["index"] or branch["trajectory"] != item["trajectory"]:
            raise ValueError(f"branch identity mismatch: {path}")
        if payload["environment_seed"] != prereg["environment_seed"]:
            raise ValueError(f"environment seed mismatch: {path}")
        if payload["policy_noise_seed_base"] != item["policy_noise_seed"]:
            raise ValueError(f"policy seed mismatch: {path}")
        with np.load(item["trajectory"], allow_pickle=False) as source:
            state = np.asarray(source["sim_state"][item["index"]], dtype=np.float64)
        key = (
            item["suite"], item["task"], item["trial"],
            item["distance_replans"], item["index"],
        )
        groups[key].append(
            {
                "ordinal": item["ordinal"],
                "noise_offset": item["noise_offset"],
                "policy_noise_seed": item["policy_noise_seed"],
                "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                "success": bool(episode["success"]),
                "final_step": int(episode["steps"]),
                "replans": int(episode["replans"]),
                "first_action_chunk": np.asarray(
                    episode["first_environment_action_chunk"], dtype=np.float32
                ),
                "result": str(path),
            }
        )

    reports = []
    for key, items in sorted(groups.items()):
        if len(items) != 4 or len({item["state_sha256"] for item in items}) != 1:
            raise ValueError(f"invalid C22 same-state group {key}")
        pair_rms = [
            float(np.sqrt(np.mean((left["first_action_chunk"] - right["first_action_chunk"]) ** 2)))
            for left, right in itertools.combinations(items, 2)
        ]
        successes = sum(item["success"] for item in items)
        reports.append(
            {
                "suite": key[0], "task": key[1], "trial": key[2],
                "distance_replans": key[3], "index": key[4],
                "state_sha256": items[0]["state_sha256"],
                "successes": successes, "branches": 4,
                "mixed_outcomes": 0 < successes < 4,
                "min_pairwise_first_chunk_rms": min(pair_rms),
                "max_pairwise_first_chunk_rms": max(pair_rms),
                "branches_detail": [
                    {field: value for field, value in item.items() if field != "first_action_chunk"}
                    for item in items
                ],
            }
        )
    if len(reports) != prereg["groups"] or prereg["groups"] != 24:
        raise ValueError("C22 group count mismatch")
    action_diverse = all(row["min_pairwise_first_chunk_rms"] > 1e-6 for row in reports)
    mixed = [row for row in reports if row["mixed_outcomes"]]
    mixed_suites = sorted({row["suite"] for row in mixed})
    passed = action_diverse and len(mixed) >= 4 and len(mixed_suites) >= 2
    strata = []
    for suite in sorted({row["suite"] for row in reports}):
        for distance in (1, 3, 5):
            subset = [row for row in reports if row["suite"] == suite and row["distance_replans"] == distance]
            strata.append(
                {
                    "suite": suite, "distance_replans": distance,
                    "groups": len(subset),
                    "mixed_groups": sum(row["mixed_outcomes"] for row in subset),
                    "successful_branches": sum(row["successes"] for row in subset),
                    "branches": 4 * len(subset),
                }
            )
    report = {
        **prereg,
        "successful_branches": sum(row["successes"] for row in reports),
        "mixed_outcome_groups": len(mixed),
        "mixed_outcome_suites": mixed_suites,
        "all_groups_action_diverse": action_diverse,
        "strata": strata,
        "group_reports": reports,
        "status": "PASS_COUNTERFACTUAL_ENTROPY_SWEEP" if passed else "FAIL_COUNTERFACTUAL_ENTROPY_SWEEP",
        "data_permission": (
            prereg["pass_permission"] if passed else prereg["fail_permission"]
        ),
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
            "mixed_outcome_suites", "all_groups_action_diverse", "data_permission",
            "effect_conclusion",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
