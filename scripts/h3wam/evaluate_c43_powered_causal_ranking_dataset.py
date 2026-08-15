#!/usr/bin/env python3
"""Audit source-disjoint C43 causal action outcomes and consequences."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

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
    root = args.root.resolve()
    prereg = json.loads((root / "preregistration.json").read_text())
    selections = [
        json.loads(line) for line in (root / "selection.jsonl").read_text().splitlines()
        if line
    ]
    if len(selections) != int(prereg["branches"]):
        raise ValueError("C43 branch count mismatch")

    groups: dict[int, list[dict]] = defaultdict(list)
    consequence_rows = terminal_fallbacks = 0
    sources: dict[str, set[str]] = defaultdict(set)
    for item in selections:
        split = str(item["split"])
        if split not in {"train", "fresh_final"}:
            raise ValueError(f"unknown C43 split: {split}")
        sources[split].add(item["source_episode"])
        directory = root / "runs" / run_name(item)
        payload = json.loads((directory / "results.json").read_text())
        episode = payload["tasks"][0]["episodes"][0]
        trajectories = tuple(directory.glob("*trajectory.npz"))
        if len(trajectories) != 1:
            raise ValueError(f"expected one trajectory in {directory}")
        with np.load(trajectories[0], allow_pickle=False) as trajectory:
            missing = [field for field in TERMINAL_FIELDS if field not in trajectory.files]
            if missing:
                raise ValueError(f"terminal fields missing in {directory}: {missing}")
            start_step = int(trajectory["step"][0])
            terminal_step = int(trajectory["terminal_step"])
            if not start_step < terminal_step <= start_step + 400:
                raise ValueError(f"invalid terminal step in {directory}")
            if len(trajectory["step"]) >= 2:
                if int(trajectory["step"][1]) != start_step + 32:
                    raise ValueError(f"future row is not start+32 in {directory}")
            else:
                if terminal_step > start_step + 32:
                    raise ValueError(f"terminal fallback exceeds first chunk in {directory}")
                terminal_fallbacks += 1
            consequence_rows += 1
            stored_actions = np.asarray(trajectory["policy_actions"][0], dtype=np.float32)
        action_chunk = np.asarray(episode["first_environment_action_chunk"], dtype=np.float32)
        if not np.array_equal(action_chunk, stored_actions):
            raise ValueError(f"action/result mismatch in {directory}")
        seeds = [int(seed) for seed in episode["replan_noise_seeds"]]
        expected = [int(item["first_policy_noise_seed"])] + [
            int(item["continuation_policy_noise_seed_base"]) + index
            for index in range(max(0, len(seeds) - 1))
        ]
        if payload["first_replan_steps"] != 32 or payload["replan_steps"] != 8:
            raise ValueError(f"execution metadata mismatch in {directory}")
        groups[int(item["group_id"])].append({
            "success": bool(episode["success"]),
            "suite": item["suite"],
            "source_episode": item["source_episode"],
            "split": split,
            "seeds_valid": seeds == expected,
            "actions": action_chunk,
        })
    if len(groups) != int(prereg["groups"]):
        raise ValueError("C43 group count mismatch")

    reports = []
    for group_id, items in sorted(groups.items()):
        if len(items) != 4:
            raise ValueError(f"group {group_id} does not have four candidates")
        if len({item["split"] for item in items}) != 1:
            raise ValueError(f"group {group_id} crosses splits")
        rms = [
            float(np.sqrt(np.mean((a["actions"] - b["actions"]) ** 2)))
            for a, b in itertools.combinations(items, 2)
        ]
        successes = sum(int(item["success"]) for item in items)
        reports.append({
            "group_id": group_id,
            "split": items[0]["split"],
            "suite": items[0]["suite"],
            "source_episode": items[0]["source_episode"],
            "successes": successes,
            "mixed_outcomes": 0 < successes < 4,
            "all_seed_schedules_valid": all(item["seeds_valid"] for item in items),
            "min_pairwise_first_chunk_rms": min(rms),
        })

    mixed_by_split = Counter(row["split"] for row in reports if row["mixed_outcomes"])
    mixed_suites = {
        split: sorted({
            row["suite"] for row in reports
            if row["split"] == split and row["mixed_outcomes"]
        })
        for split in ("train", "fresh_final")
    }
    schedules_valid = all(row["all_seed_schedules_valid"] for row in reports)
    action_diverse = all(row["min_pairwise_first_chunk_rms"] > 1e-6 for row in reports)
    passed = (
        consequence_rows == len(selections)
        and schedules_valid
        and action_diverse
        and mixed_by_split["train"] >= 24
        and mixed_by_split["fresh_final"] >= 24
        and len(mixed_suites["train"]) >= 3
        and len(mixed_suites["fresh_final"]) >= 3
    )
    report = {
        **prereg,
        "successful_branches": sum(row["successes"] for row in reports),
        "mixed_groups_by_split": dict(sorted(mixed_by_split.items())),
        "mixed_suites_by_split": mixed_suites,
        "source_episodes_observed_by_split": {
            split: len(items) for split, items in sorted(sources.items())
        },
        "all_seed_schedules_valid": schedules_valid,
        "all_groups_action_diverse": action_diverse,
        "consequence_observations": consequence_rows,
        "terminal_fallback_branches": terminal_fallbacks,
        "group_reports": reports,
        "status": (
            "PASS_C43_POWERED_CAUSAL_RANKING_DATASET"
            if passed else "FAIL_C43_POWERED_CAUSAL_RANKING_DATASET"
        ),
        "training_permission": prereg["pass_permission"] if passed else prereg["fail_permission"],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({key: report[key] for key in (
        "status", "branches", "successful_branches", "mixed_groups_by_split",
        "mixed_suites_by_split", "source_episodes_observed_by_split",
        "consequence_observations", "terminal_fallback_branches", "training_permission",
    )}, indent=2))


if __name__ == "__main__":
    main()
