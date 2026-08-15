#!/usr/bin/env python3
"""Audit paired LIBERO branch rollouts for real alternative-action outcomes."""

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
    selections = []
    for line in (args.root / "selection.txt").read_text().splitlines():
        suite, task, trial, index, base_seed, offset = line.split()
        selections.append(
            {
                "suite": suite, "task": int(task), "trial": int(trial),
                "index": int(index), "base_seed": int(base_seed),
                "offset": int(offset),
            }
        )
    if len(selections) != 16:
        raise ValueError("C21 requires exactly 16 frozen branches")
    groups = defaultdict(list)
    rows = []
    for item in selections:
        slug = item["suite"].removeprefix("libero_")
        name = (
            f"{slug}_task{item['task']}_trial{item['trial']}_"
            f"index{item['index']}_offset{item['offset']}"
        )
        path = args.root / "runs" / name / "results.json"
        payload = json.loads(path.read_text())
        episode = payload["tasks"][0]["episodes"][0]
        branch = episode["branch_start"]
        if branch["index"] != item["index"]:
            raise ValueError(f"branch index mismatch: {path}")
        if payload["environment_seed"] != 42:
            raise ValueError(f"environment seed mismatch: {path}")
        expected_seed = item["base_seed"] + item["offset"]
        if payload["policy_noise_seed_base"] != expected_seed:
            raise ValueError(f"policy seed mismatch: {path}")
        with np.load(branch["trajectory"], allow_pickle=False) as source:
            state = np.asarray(source["sim_state"][item["index"]], dtype=np.float64)
        chunk = np.asarray(episode["first_environment_action_chunk"], dtype=np.float32)
        row = {
            **item,
            "policy_noise_seed": expected_seed,
            "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            "success": bool(episode["success"]),
            "final_step": int(episode["steps"]),
            "replans": int(episode["replans"]),
            "first_action_chunk": chunk,
            "result": str(path),
        }
        key = (item["suite"], item["task"], item["trial"], item["index"])
        groups[key].append(row)
        rows.append(row)

    group_reports = []
    for key, items in sorted(groups.items()):
        if len(items) != 4:
            raise ValueError(f"C21 group {key} does not have four branches")
        hashes = {item["state_sha256"] for item in items}
        if len(hashes) != 1:
            raise ValueError(f"C21 group {key} does not share one state")
        pair_rms = [
            float(np.sqrt(np.mean((left["first_action_chunk"] - right["first_action_chunk"]) ** 2)))
            for left, right in itertools.combinations(items, 2)
        ]
        successes = sum(item["success"] for item in items)
        group_reports.append(
            {
                "suite": key[0], "task": key[1], "trial": key[2], "index": key[3],
                "state_sha256": next(iter(hashes)),
                "branches": 4, "successes": successes,
                "mixed_outcomes": 0 < successes < 4,
                "min_pairwise_first_chunk_rms": min(pair_rms),
                "max_pairwise_first_chunk_rms": max(pair_rms),
                "branches_detail": [
                    {
                        key: value for key, value in item.items()
                        if key != "first_action_chunk"
                    }
                    for item in items
                ],
            }
        )
    action_diverse = all(
        group["min_pairwise_first_chunk_rms"] > 1e-6 for group in group_reports
    )
    mixed_groups = sum(group["mixed_outcomes"] for group in group_reports)
    passed = action_diverse and mixed_groups >= 1
    report = {
        "format": "h3wam-c21-counterfactual-outcome-canary-v1",
        "experiment_class": "controlled data-collection canary",
        "falsifiable_hypothesis": (
            "At a fixed canonical LIBERO state, changing only D0 diffusion noise "
            "produces distinct action chunks and at least one mixed success/failure group."
        ),
        "parent": "D0-H32-s14000/replan8/no ensemble",
        "only_variable": "policy diffusion noise seed offset",
        "environment_seed": 42,
        "groups": len(group_reports), "branches": len(rows),
        "successful_branches": sum(row["success"] for row in rows),
        "mixed_outcome_groups": mixed_groups,
        "all_groups_action_diverse": action_diverse,
        "promotion": "all groups action-diverse and >=1 same-state mixed-outcome group",
        "status": (
            "PASS_COUNTERFACTUAL_OUTCOME_CANARY"
            if passed else "FAIL_COUNTERFACTUAL_OUTCOME_CANARY"
        ),
        "training_permission": (
            "GO_DATASET_EXPANSION" if passed else "NO_GO_CRITIC_TRAINING"
        ),
        "effect_conclusion": "NOT_EVIDENCE_READY",
        "boundary": (
            "A pass authorizes episode-disjoint outcome collection, not critic training "
            "or best-of-N deployment until train/validation ranking gates are defined."
        ),
        "group_reports": group_reports,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
