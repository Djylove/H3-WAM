#!/usr/bin/env python3
"""Audit the preregistered C52 branch execution and outcome yield."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_c27_expanded_causal_dataset import run_name


TERMINAL_FIELDS = (
    "terminal_step", "terminal_agentview_image", "terminal_wristview_image",
    "terminal_eef_pos", "terminal_eef_quat", "terminal_gripper_qpos",
    "terminal_previous_action", "terminal_sim_state",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads((args.root / "preregistration.json").read_text())
    selections = [json.loads(line) for line in (args.root / "selection.jsonl").read_text().splitlines() if line]
    if len(selections) != int(prereg["branches"]):
        raise ValueError("C52 branch inventory mismatch")
    groups: dict[int, list[dict]] = defaultdict(list)
    terminal_fallbacks = 0
    for item in selections:
        directory = args.root / "runs" / run_name(item)
        payload = json.loads((directory / "results.json").read_text())
        episode = payload["tasks"][0]["episodes"][0]
        trajectories = tuple(directory.glob("*trajectory.npz"))
        if len(trajectories) != 1:
            raise ValueError(f"trajectory count mismatch: {directory}")
        with np.load(trajectories[0], allow_pickle=False) as trajectory:
            missing = [field for field in TERMINAL_FIELDS if field not in trajectory.files]
            if missing:
                raise ValueError(f"missing terminal fields in {directory}: {missing}")
            start = int(trajectory["step"][0]); terminal = int(trajectory["terminal_step"])
            if not start < terminal <= start + 400:
                raise ValueError(f"terminal contract mismatch: {directory}")
            if len(trajectory["step"]) >= 2:
                if int(trajectory["step"][1]) != start + 32:
                    raise ValueError(f"future row is not start+32: {directory}")
            else:
                if terminal > start + 32:
                    raise ValueError(f"terminal fallback exceeds chunk32: {directory}")
                terminal_fallbacks += 1
            stored = np.asarray(trajectory["policy_actions"][0], dtype=np.float32)
        actions = np.asarray(episode["first_environment_action_chunk"], dtype=np.float32)
        if not np.array_equal(actions, stored):
            raise ValueError(f"stored/result action mismatch: {directory}")
        seeds = [int(seed) for seed in episode["replan_noise_seeds"]]
        expected = [int(item["first_policy_noise_seed"])] + [
            int(item["continuation_policy_noise_seed_base"]) + i
            for i in range(max(0, len(seeds) - 1))
        ]
        if payload["first_replan_steps"] != 32 or payload["replan_steps"] != 8:
            raise ValueError(f"execution metadata mismatch: {directory}")
        groups[int(item["group_id"])].append({
            **item, "success": bool(episode["success"]),
            "seeds_valid": seeds == expected, "actions": actions,
            "run_directory": str(directory),
        })
    if len(groups) != int(prereg["groups"]):
        raise ValueError("C52 group count mismatch")

    reports = []
    for group_id, items in sorted(groups.items()):
        items.sort(key=lambda row: row["ordinal"])
        if len(items) != 4:
            raise ValueError(f"C52 group {group_id} size mismatch")
        rms = [float(np.sqrt(np.mean((a["actions"] - b["actions"]) ** 2))) for a, b in itertools.combinations(items, 2)]
        successes = sum(int(item["success"]) for item in items)
        reports.append({
            "group_id": group_id, "suite": items[0]["suite"],
            "source_episode": items[0]["source_episode"],
            "successes": successes, "mixed_outcomes": 0 < successes < 4,
            "all_seed_schedules_valid": all(item["seeds_valid"] for item in items),
            "min_pairwise_first_chunk_rms": min(rms),
            "candidates": [{k: item[k] for k in (
                "ordinal", "noise_offset", "first_policy_noise_seed", "success",
                "run_directory", "trajectory", "index",
            )} for item in items],
        })
    mixed = [row for row in reports if row["mixed_outcomes"]]
    mixed_suites = sorted({row["suite"] for row in mixed})
    mechanics = {
        "all_240_consequences_present": len(selections) == 240,
        "all_seed_schedules_valid": all(row["all_seed_schedules_valid"] for row in reports),
        "all_groups_action_diverse": all(row["min_pairwise_first_chunk_rms"] > 1e-6 for row in reports),
    }
    passed = all(mechanics.values()) and len(mixed) >= 10 and len(mixed_suites) >= 3
    report = {
        **prereg, "successful_branches": sum(row["successes"] for row in reports),
        "mixed_groups": len(mixed), "mixed_suites": mixed_suites,
        "successes_by_suite": dict(Counter(
            row["suite"] for row in reports for _ in range(row["successes"])
        )),
        "terminal_fallback_branches": terminal_fallbacks,
        "mechanical_gate": mechanics, "group_reports": reports,
        "status": "PASS_C52_COUNTERFACTUAL_OUTCOMES" if passed else "FAIL_C52_COUNTERFACTUAL_OUTCOMES",
        "ranking_permission": "GO_C52_FROZEN_VALUE_SCORING" if passed else "NO_GO_C52_VALUE_SCORING",
    }
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({key: report[key] for key in (
        "status", "ranking_permission", "branches", "successful_branches",
        "mixed_groups", "mixed_suites", "terminal_fallback_branches", "mechanical_gate",
    )}, indent=2))


if __name__ == "__main__":
    main()
