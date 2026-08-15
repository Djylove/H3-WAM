#!/usr/bin/env python3
"""Audit C27 fresh-source causal branches and label yield."""

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


def run_name(item: dict) -> str:
    slug = item["suite"].removeprefix("libero_")
    return (
        f"{item['ordinal']}_g{item['group_id']}_{slug}_task{item['task']}_"
        f"trial{item['trial']}_d{item['distance_replans']}_offset{item['noise_offset']}"
    )


def main() -> None:
    args = parse_args()
    prereg = json.loads((args.root / "preregistration.json").read_text(encoding="utf-8"))
    selections = [
        json.loads(line)
        for line in (args.root / "selection.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(selections) != prereg["branches"]:
        raise ValueError("C27 branch count mismatch")
    source_splits: dict[str, set[str]] = defaultdict(set)
    groups: dict[int, list[dict]] = defaultdict(list)
    for item in selections:
        source_splits[item["source_episode"]].add(item["split"])
        payload = json.loads(
            (args.root / "runs" / run_name(item) / "results.json").read_text(encoding="utf-8")
        )
        episode = payload["tasks"][0]["episodes"][0]
        seeds = [int(seed) for seed in episode["replan_noise_seeds"]]
        expected = [item["first_policy_noise_seed"]] + [
            item["continuation_policy_noise_seed_base"] + index
            for index in range(max(0, len(seeds) - 1))
        ]
        if payload["first_replan_steps"] != 32 or payload["replan_steps"] != 8:
            raise ValueError(f"C27 execution metadata mismatch: {run_name(item)}")
        groups[item["group_id"]].append({
            "success": bool(episode["success"]),
            "split": item["split"], "suite": item["suite"],
            "source_episode": item["source_episode"],
            "seed_schedule_valid": seeds == expected,
            "first_action_chunk": np.asarray(
                episode["first_environment_action_chunk"], dtype=np.float32
            ),
        })
    if len(groups) != prereg["groups"]:
        raise ValueError("C27 group count mismatch")
    reports = []
    for group_id, items in sorted(groups.items()):
        if len(items) != 4:
            raise ValueError(f"C27 group {group_id} size mismatch")
        pair_rms = [
            float(np.sqrt(np.mean((left["first_action_chunk"] - right["first_action_chunk"]) ** 2)))
            for left, right in itertools.combinations(items, 2)
        ]
        successes = sum(item["success"] for item in items)
        reports.append({
            "group_id": group_id, "split": items[0]["split"],
            "suite": items[0]["suite"], "source_episode": items[0]["source_episode"],
            "successes": successes, "mixed_outcomes": 0 < successes < 4,
            "all_seed_schedules_valid": all(item["seed_schedule_valid"] for item in items),
            "min_pairwise_first_chunk_rms": min(pair_rms),
            "max_pairwise_first_chunk_rms": max(pair_rms),
        })
    split_isolated = all(len(values) == 1 for values in source_splits.values())
    action_diverse = all(row["min_pairwise_first_chunk_rms"] > 1e-6 for row in reports)
    schedules_valid = all(row["all_seed_schedules_valid"] for row in reports)
    mixed = [row for row in reports if row["mixed_outcomes"]]
    train_mixed = [row for row in mixed if row["split"] == "train"]
    val_mixed = [row for row in mixed if row["split"] == "val"]
    mixed_suites = sorted({row["suite"] for row in mixed})
    passed = (
        split_isolated and action_diverse and schedules_valid
        and len(train_mixed) >= 10 and len(val_mixed) >= 4
        and len(mixed_suites) == 3
    )
    report = {
        **prereg,
        "successful_branches": sum(row["successes"] for row in reports),
        "mixed_outcome_groups": len(mixed),
        "train_mixed_groups": len(train_mixed),
        "val_mixed_groups": len(val_mixed),
        "mixed_outcome_suites": mixed_suites,
        "source_episode_split_isolated": split_isolated,
        "all_seed_schedules_valid": schedules_valid,
        "all_groups_action_diverse": action_diverse,
        "group_reports": reports,
        "status": "PASS_C27_EXPANDED_CAUSAL_DATASET" if passed else "FAIL_C27_EXPANDED_CAUSAL_DATASET",
        "training_permission": prereg["pass_permission"] if passed else prereg["fail_permission"],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({key: report[key] for key in (
        "status", "branches", "successful_branches", "mixed_outcome_groups",
        "train_mixed_groups", "val_mixed_groups", "mixed_outcome_suites",
        "training_permission",
    )}, indent=2))


if __name__ == "__main__":
    main()
