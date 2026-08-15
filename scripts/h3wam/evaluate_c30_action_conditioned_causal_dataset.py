#!/usr/bin/env python3
"""Audit C30 causal outcomes and post-action consequence observations."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_c27_expanded_causal_dataset import run_name


TERMINAL_FIELDS = (
    "terminal_step", "terminal_agentview_image", "terminal_wristview_image",
    "terminal_eef_pos", "terminal_eef_quat", "terminal_gripper_qpos",
    "terminal_previous_action", "terminal_sim_state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    prereg = json.loads((root / "preregistration.json").read_text(encoding="utf-8"))
    selections = [json.loads(line) for line in (root / "selection.jsonl").read_text().splitlines() if line]
    if len(selections) != int(prereg["branches"]):
        raise ValueError("C30 branch count mismatch")
    source_splits: dict[str, set[str]] = defaultdict(set)
    groups: dict[int, list[dict]] = defaultdict(list)
    consequence_rows = 0
    terminal_fallbacks = 0
    for item in selections:
        source_splits[item["source_episode"]].add(item["split"])
        directory = root / "runs" / run_name(item)
        payload = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        episode = payload["tasks"][0]["episodes"][0]
        trajectories = tuple(directory.glob("*trajectory.npz"))
        if len(trajectories) != 1:
            raise ValueError(f"expected one C30 trajectory in {directory}")
        with np.load(trajectories[0], allow_pickle=False) as trajectory:
            missing = [field for field in TERMINAL_FIELDS if field not in trajectory.files]
            if missing:
                raise ValueError(f"terminal consequence fields missing in {directory}: {missing}")
            if len(trajectory["step"]) < 1:
                raise ValueError(f"empty trajectory in {directory}")
            start_step = int(trajectory["step"][0])
            terminal_step = int(trajectory["terminal_step"])
            if not start_step < terminal_step <= start_step + 400:
                raise ValueError(f"invalid terminal step in {directory}")
            if len(trajectory["step"]) >= 2:
                if int(trajectory["step"][1]) != start_step + 32:
                    raise ValueError(f"first consequence row is not start+32 in {directory}")
                consequence_source = "replan_row_1"
            else:
                if terminal_step > start_step + 32:
                    raise ValueError(f"terminal fallback exceeds first action chunk in {directory}")
                consequence_source = "terminal_within_first_chunk"
                terminal_fallbacks += 1
            consequence_rows += 1
            stored_actions = np.asarray(trajectory["policy_actions"][0], dtype=np.float32)
        action_chunk = np.asarray(episode["first_environment_action_chunk"], dtype=np.float32)
        if not np.array_equal(action_chunk, stored_actions):
            raise ValueError(f"result/trajectory first action mismatch in {directory}")
        seeds = [int(seed) for seed in episode["replan_noise_seeds"]]
        expected = [item["first_policy_noise_seed"]] + [
            item["continuation_policy_noise_seed_base"] + index
            for index in range(max(0, len(seeds) - 1))
        ]
        if payload["first_replan_steps"] != 32 or payload["replan_steps"] != 8:
            raise ValueError(f"C30 execution metadata mismatch in {directory}")
        groups[int(item["group_id"])].append({
            "success": bool(episode["success"]), "split": item["split"],
            "suite": item["suite"], "source_episode": item["source_episode"],
            "seed_schedule_valid": seeds == expected,
            "first_action_chunk": action_chunk,
            "consequence_source": consequence_source,
        })
    if len(groups) != int(prereg["groups"]):
        raise ValueError("C30 group count mismatch")
    reports = []
    for group_id, items in sorted(groups.items()):
        if len(items) != 4:
            raise ValueError(f"C30 group {group_id} does not have four branches")
        rms = [float(np.sqrt(np.mean((a["first_action_chunk"] - b["first_action_chunk"]) ** 2))) for a,b in itertools.combinations(items,2)]
        successes = sum(int(item["success"]) for item in items)
        reports.append({
            "group_id": group_id, "split": items[0]["split"],
            "suite": items[0]["suite"], "source_episode": items[0]["source_episode"],
            "successes": successes, "mixed_outcomes": 0 < successes < 4,
            "all_seed_schedules_valid": all(item["seed_schedule_valid"] for item in items),
            "min_pairwise_first_chunk_rms": min(rms),
            "terminal_fallback_branches": sum(item["consequence_source"] == "terminal_within_first_chunk" for item in items),
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
        and consequence_rows == len(selections)
        and len(train_mixed) >= 10 and len(val_mixed) >= 4
        and len(mixed_suites) >= 3
    )
    report = {
        **prereg, "successful_branches": sum(row["successes"] for row in reports),
        "mixed_outcome_groups": len(mixed), "train_mixed_groups": len(train_mixed),
        "val_mixed_groups": len(val_mixed), "mixed_outcome_suites": mixed_suites,
        "source_episode_split_isolated": split_isolated,
        "all_seed_schedules_valid": schedules_valid,
        "all_groups_action_diverse": action_diverse,
        "consequence_observations": consequence_rows,
        "terminal_fallback_branches": terminal_fallbacks,
        "group_reports": reports,
        "status": "PASS_C30_ACTION_CONDITIONED_CAUSAL_DATASET" if passed else "FAIL_C30_ACTION_CONDITIONED_CAUSAL_DATASET",
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
        "consequence_observations", "terminal_fallback_branches", "training_permission",
    )}, indent=2))


if __name__ == "__main__":
    main()
