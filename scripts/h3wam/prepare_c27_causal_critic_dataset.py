#!/usr/bin/env python3
"""Freeze audited C27 branches into the fresh critic train/validation contract."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from prepare_c26_causal_critic_dataset import load_branch, sha256_file, tensor_sha256


FORMAT = "h3wam-c27-causal-critic-dataset-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c27-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.c27_root.resolve()
    completed = root / "COMPLETED"
    selection_path = root / "selection.jsonl"
    completion = json.loads(completed.read_text(encoding="utf-8"))
    if completion.get("status") != "PASS_C27_EXPANDED_CAUSAL_DATASET":
        raise ValueError("C27 dataset did not pass its preregistered gate")
    selections = [
        json.loads(line)
        for line in selection_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(selections) != 312:
        raise ValueError(f"expected 312 C27 branches, got {len(selections)}")
    branches = [load_branch(root, row) for row in selections]
    grouped: dict[int, list[dict]] = defaultdict(list)
    for branch in branches:
        grouped[int(branch["selection"]["group_id"])].append(branch)
    if sorted(grouped) != list(range(78)):
        raise ValueError("C27 group ids must be contiguous 0..77")

    states, mixed_groups = [], []
    source_splits: dict[str, set[str]] = defaultdict(set)
    for group_id, items in sorted(grouped.items()):
        if len(items) != 4:
            raise ValueError(f"group {group_id} must contain four branches")
        first = items[0]
        selection = first["selection"]
        for field in ("agentview_image", "wristview_image", "proprio", "sim_state"):
            if not all(torch.equal(first[field], item[field]) for item in items[1:]):
                raise ValueError(f"group {group_id} start {field} is not bit-exact")
        for field in ("task_language", "absolute_step"):
            if not all(first[field] == item[field] for item in items[1:]):
                raise ValueError(f"group {group_id} start {field} differs")
        if len({tensor_sha256(item["environment_actions"]) for item in items}) != 4:
            raise ValueError(f"group {group_id} candidate chunks are not unique")
        successes = sum(int(item["success"]) for item in items)
        if 0 < successes < 4:
            mixed_groups.append(group_id)
        source_episode, split = str(selection["source_episode"]), str(selection["split"])
        source_splits[source_episode].add(split)
        states.append({
            "group_id": group_id, "source_episode": source_episode,
            "suite": str(selection["suite"]), "task": int(selection["task"]),
            "trial": int(selection["trial"]), "split": split,
            "distance_replans": int(selection["distance_replans"]),
            "source_index": int(selection["index"]),
            "task_language": first["task_language"],
            "absolute_step": first["absolute_step"], "successes": successes,
            "mixed_outcomes": 0 < successes < 4,
            "agentview_image": first["agentview_image"],
            "wristview_image": first["wristview_image"],
            "proprio": first["proprio"], "sim_state": first["sim_state"],
            "agentview_sha256": tensor_sha256(first["agentview_image"]),
            "wristview_sha256": tensor_sha256(first["wristview_image"]),
            "sim_state_sha256": tensor_sha256(first["sim_state"]),
        })
    if not all(len(splits) == 1 for splits in source_splits.values()):
        raise ValueError("C27 source episode appears in more than one split")
    train_mixed = [group for group in mixed_groups if states[group]["split"] == "train"]
    val_mixed = [group for group in mixed_groups if states[group]["split"] == "val"]
    if len(train_mixed) != int(completion["train_mixed_groups"]):
        raise ValueError("C27 train mixed count differs from audited completion")
    if len(val_mixed) != int(completion["val_mixed_groups"]):
        raise ValueError("C27 validation mixed count differs from audited completion")

    branch_rows = []
    for branch in branches:
        row = branch["selection"]
        branch_rows.append({
            "ordinal": int(row["ordinal"]), "group_id": int(row["group_id"]),
            "split": str(row["split"]), "suite": str(row["suite"]),
            "source_episode": str(row["source_episode"]),
            "noise_offset": int(row["noise_offset"]),
            "success": branch["success"], "final_step": branch["final_step"],
            "replans": branch["replans"],
            "environment_actions": branch["environment_actions"],
            "action_sha256": tensor_sha256(branch["environment_actions"]),
            "result_sha256": branch["result_sha256"],
            "trajectory_sha256": branch["trajectory_sha256"],
        })
    payload = {
        "format": FORMAT,
        "classification": "fresh_controlled_first_action_outcome_dataset",
        "c27_root": str(root), "c27_completed_sha256": sha256_file(completed),
        "c27_selection_sha256": sha256_file(selection_path),
        "branches": branch_rows, "states": states,
        "audit": {
            "branch_count": len(branch_rows), "group_count": len(states),
            "source_episode_count": len(source_splits),
            "source_episode_split_isolated": True,
            "all_start_states_bit_exact_within_group": True,
            "all_result_actions_match_trajectory": True,
            "all_candidate_chunks_unique_within_group": True,
            "mixed_groups": mixed_groups,
            "train_mixed_groups": train_mixed, "val_mixed_groups": val_mixed,
            "train_pairwise_comparisons": sum(
                states[group]["successes"] * (4 - states[group]["successes"])
                for group in train_mixed
            ),
            "val_pairwise_comparisons": sum(
                states[group]["successes"] * (4 - states[group]["successes"])
                for group in val_mixed
            ),
        },
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print(json.dumps(payload["audit"], indent=2))


if __name__ == "__main__":
    main()
