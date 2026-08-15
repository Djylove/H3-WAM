#!/usr/bin/env python3
"""Evaluate paired C24 first-action execution-horizon challengers."""

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
    parser.add_argument("--c23-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run_name(item: dict) -> str:
    slug = item["suite"].removeprefix("libero_")
    return (
        f"{item['ordinal']}_{slug}_task{item['task']}_trial{item['trial']}_"
        f"d{item['distance_replans']}_offset{item['noise_offset']}"
    )


def evaluate_candidate(root: Path, c23_root: Path, selections: list[dict], horizon: int) -> dict:
    groups = defaultdict(list)
    for item in selections:
        name = run_name(item)
        payload = json.loads((root / f"h{horizon}" / "runs" / name / "results.json").read_text())
        parent = json.loads((c23_root / "runs" / name / "results.json").read_text())
        episode = payload["tasks"][0]["episodes"][0]
        parent_episode = parent["tasks"][0]["episodes"][0]
        first = np.asarray(episode["first_environment_action_chunk"], dtype=np.float32)
        parent_first = np.asarray(parent_episode["first_environment_action_chunk"], dtype=np.float32)
        seeds = [int(seed) for seed in episode["replan_noise_seeds"]]
        expected = [item["first_policy_noise_seed"]] + [
            item["continuation_policy_noise_seed_base"] + index
            for index in range(max(0, len(seeds) - 1))
        ]
        if payload["first_replan_steps"] != horizon or payload["replan_steps"] != 8:
            raise ValueError(f"C24 execution metadata mismatch: {name}")
        key = (item["suite"], item["task"], item["trial"], item["distance_replans"], item["index"])
        groups[key].append(
            {
                "ordinal": item["ordinal"], "noise_offset": item["noise_offset"],
                "success": bool(episode["success"]), "final_step": int(episode["steps"]),
                "replans": int(episode["replans"]),
                "first_action_chunk": first,
                "first_chunk_exact_to_c23": bool(np.array_equal(first, parent_first)),
                "seed_schedule_valid": seeds == expected,
                "c23_success": bool(parent_episode["success"]),
            }
        )
    reports = []
    for key, items in sorted(groups.items()):
        pair_rms = [
            float(np.sqrt(np.mean((left["first_action_chunk"] - right["first_action_chunk"]) ** 2)))
            for left, right in itertools.combinations(items, 2)
        ]
        successes = sum(item["success"] for item in items)
        reports.append(
            {
                "suite": key[0], "task": key[1], "trial": key[2],
                "distance_replans": key[3], "index": key[4], "branches": len(items),
                "successes": successes, "mixed_outcomes": 0 < successes < 4,
                "c23_successes": sum(item["c23_success"] for item in items),
                "all_first_chunks_exact_to_c23": all(item["first_chunk_exact_to_c23"] for item in items),
                "all_seed_schedules_valid": all(item["seed_schedule_valid"] for item in items),
                "min_pairwise_first_chunk_rms": min(pair_rms),
            }
        )
    if len(reports) != 8 or any(row["branches"] != 4 for row in reports):
        raise ValueError(f"C24 h{horizon} group contract mismatch")
    mixed = [row for row in reports if row["mixed_outcomes"]]
    mixed_suites = sorted({row["suite"] for row in mixed})
    mechanical = (
        all(row["all_first_chunks_exact_to_c23"] for row in reports)
        and all(row["all_seed_schedules_valid"] for row in reports)
        and all(row["min_pairwise_first_chunk_rms"] > 1e-6 for row in reports)
    )
    return {
        "horizon": horizon, "branches": 32,
        "successful_branches": sum(row["successes"] for row in reports),
        "mixed_outcome_groups": len(mixed), "mixed_outcome_suites": mixed_suites,
        "mechanical_gate": mechanical,
        "promotion_gate": mechanical and len(mixed) >= 3 and len(mixed_suites) >= 2,
        "group_reports": reports,
    }


def main() -> None:
    args = parse_args()
    prereg = json.loads((args.root / "preregistration.json").read_text())
    selections = [json.loads(line) for line in (args.root / "selection.jsonl").read_text().splitlines()]
    if len(selections) != 32:
        raise ValueError("C24 selection count mismatch")
    candidates = [
        evaluate_candidate(args.root, args.c23_root, selections, horizon)
        for horizon in prereg["candidates"]
    ]
    promoted = [row for row in candidates if row["promotion_gate"]]
    if promoted:
        winner = sorted(promoted, key=lambda row: (-row["mixed_outcome_groups"], row["horizon"]))[0]["horizon"]
        status = "PASS_FIRST_ACTION_HORIZON_SWEEP"
        permission = prereg["pass_permission"]
    else:
        winner = None
        status = "FAIL_FIRST_ACTION_HORIZON_SWEEP"
        permission = prereg["fail_permission"]
    report = {
        **prereg, "candidate_reports": candidates, "selected_horizon": winner,
        "status": status, "data_permission": permission,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({
        "status": status, "selected_horizon": winner,
        "data_permission": permission, "effect_conclusion": report["effect_conclusion"],
        "candidates": [
            {key: row[key] for key in (
                "horizon", "successful_branches", "mixed_outcome_groups",
                "mixed_outcome_suites", "mechanical_gate", "promotion_gate",
            )}
            for row in candidates
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
