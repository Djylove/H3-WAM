#!/usr/bin/env python3
"""Audit C25 split isolation, seed schedules, action diversity, and label yield."""

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
    if len(selections) != 128 or prereg["branches"] != 128:
        raise ValueError("C25 branch count mismatch")
    source_splits = defaultdict(set)
    groups = defaultdict(list)
    for item in selections:
        source_splits[item["source_episode"]].add(item["split"])
        slug = item["suite"].removeprefix("libero_")
        name = (
            f"{item['ordinal']}_g{item['group_id']}_{slug}_task{item['task']}_"
            f"trial{item['trial']}_d{item['distance_replans']}_offset{item['noise_offset']}"
        )
        payload = json.loads((args.root / "runs" / name / "results.json").read_text())
        episode = payload["tasks"][0]["episodes"][0]
        seeds = [int(seed) for seed in episode["replan_noise_seeds"]]
        expected = [item["first_policy_noise_seed"]] + [
            item["continuation_policy_noise_seed_base"] + index
            for index in range(max(0, len(seeds) - 1))
        ]
        if payload["first_replan_steps"] != 32 or payload["replan_steps"] != 8:
            raise ValueError(f"C25 execution metadata mismatch: {name}")
        groups[item["group_id"]].append(
            {
                "ordinal": item["ordinal"], "noise_offset": item["noise_offset"],
                "suite": item["suite"], "task": item["task"], "trial": item["trial"],
                "split": item["split"], "source_episode": item["source_episode"],
                "distance_replans": item["distance_replans"], "index": item["index"],
                "success": bool(episode["success"]), "final_step": int(episode["steps"]),
                "replans": int(episode["replans"]),
                "seed_schedule_valid": seeds == expected,
                "first_action_chunk": np.asarray(episode["first_environment_action_chunk"], dtype=np.float32),
            }
        )
    split_isolated = all(len(splits) == 1 for splits in source_splits.values())
    reports = []
    for group_id, items in sorted(groups.items()):
        if len(items) != 4:
            raise ValueError(f"C25 group {group_id} size mismatch")
        pair_rms = [
            float(np.sqrt(np.mean((left["first_action_chunk"] - right["first_action_chunk"]) ** 2)))
            for left, right in itertools.combinations(items, 2)
        ]
        successes = sum(item["success"] for item in items)
        reports.append(
            {
                "group_id": group_id, "suite": items[0]["suite"],
                "task": items[0]["task"], "trial": items[0]["trial"],
                "split": items[0]["split"], "source_episode": items[0]["source_episode"],
                "distance_replans": items[0]["distance_replans"], "index": items[0]["index"],
                "branches": 4, "successes": successes, "mixed_outcomes": 0 < successes < 4,
                "all_seed_schedules_valid": all(item["seed_schedule_valid"] for item in items),
                "min_pairwise_first_chunk_rms": min(pair_rms),
                "max_pairwise_first_chunk_rms": max(pair_rms),
            }
        )
    if len(reports) != 32:
        raise ValueError("C25 group count mismatch")
    action_diverse = all(row["min_pairwise_first_chunk_rms"] > 1e-6 for row in reports)
    schedules_valid = all(row["all_seed_schedules_valid"] for row in reports)
    mixed = [row for row in reports if row["mixed_outcomes"]]
    train_mixed = sum(row["split"] == "train" for row in mixed)
    val_mixed = sum(row["split"] == "val" for row in mixed)
    mixed_suites = sorted({row["suite"] for row in mixed})
    passed = (
        split_isolated and action_diverse and schedules_valid and len(mixed) >= 8
        and train_mixed >= 4 and val_mixed >= 2 and len(mixed_suites) >= 3
    )
    report = {
        **prereg, "successful_branches": sum(row["successes"] for row in reports),
        "mixed_outcome_groups": len(mixed), "train_mixed_groups": train_mixed,
        "val_mixed_groups": val_mixed, "mixed_outcome_suites": mixed_suites,
        "source_episode_split_isolated": split_isolated,
        "all_seed_schedules_valid": schedules_valid,
        "all_groups_action_diverse": action_diverse, "group_reports": reports,
        "status": "PASS_EPISODE_DISJOINT_CAUSAL_DATASET_CANARY" if passed else "FAIL_EPISODE_DISJOINT_CAUSAL_DATASET_CANARY",
        "training_permission": prereg["pass_permission"] if passed else prereg["fail_permission"],
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
            "train_mixed_groups", "val_mixed_groups", "mixed_outcome_suites",
            "source_episode_split_isolated", "all_seed_schedules_valid",
            "all_groups_action_diverse", "training_permission", "effect_conclusion",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
