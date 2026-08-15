#!/usr/bin/env python3
"""Freeze C25 closed-loop branches into the C26 causal critic contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from fastwam.models.h3wam import libero_observation_state


FORMAT = "h3wam-c26-causal-critic-dataset-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c25-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: np.ndarray | torch.Tensor) -> str:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run_name(row: dict) -> str:
    slug = str(row["suite"]).removeprefix("libero_")
    return (
        f"{row['ordinal']}_g{row['group_id']}_{slug}_task{row['task']}_"
        f"trial{row['trial']}_d{row['distance_replans']}_offset{row['noise_offset']}"
    )


def load_branch(root: Path, row: dict) -> dict:
    directory = root / "runs" / run_name(row)
    result_path = directory / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    episode = result["tasks"][0]["episodes"][0]
    trajectories = tuple(directory.glob("*trajectory.npz"))
    if len(trajectories) != 1:
        raise ValueError(f"expected one trajectory in {directory}, found {len(trajectories)}")
    trajectory_path = trajectories[0]
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        if len(trajectory["step"]) < 1:
            raise ValueError(f"C25 branch trajectory is empty: {directory}")
        environment_actions = np.asarray(
            episode["first_environment_action_chunk"], dtype=np.float32
        )
        # Only row zero is causally available when candidates are ranked.  The
        # remainder of the trajectory contains continuation observations and
        # must never enter the critic input.
        stored_actions = np.asarray(trajectory["policy_actions"][0], dtype=np.float32)
        if environment_actions.shape != (32, 7):
            raise ValueError(f"C25 action chunk has wrong shape in {directory}")
        if not np.array_equal(environment_actions, stored_actions):
            raise ValueError(f"result/trajectory action mismatch in {directory}")
        observation = {
            "eef_pos": trajectory["eef_pos"][0],
            "eef_quat": trajectory["eef_quat"][0],
            "gripper_qpos": trajectory["gripper_qpos"][0],
        }
        return {
            "selection": row,
            "run_name": directory.name,
            "success": bool(episode["success"]),
            "final_step": int(episode["steps"]),
            "replans": int(episode["replans"]),
            "task_language": str(result["tasks"][0]["task"]),
            "environment_actions": torch.from_numpy(environment_actions.copy()),
            "agentview_image": torch.from_numpy(
                np.asarray(trajectory["agentview_image"][0], dtype=np.uint8).copy()
            ),
            "wristview_image": torch.from_numpy(
                np.asarray(trajectory["wristview_image"][0], dtype=np.uint8).copy()
            ),
            "proprio": libero_observation_state(observation).float(),
            "sim_state": torch.from_numpy(
                np.asarray(trajectory["sim_state"][0], dtype=np.float64).copy()
            ),
            "absolute_step": int(trajectory["step"][0]),
            "result_sha256": sha256_file(result_path),
            "trajectory_sha256": sha256_file(trajectory_path),
        }


def build_dataset(root: Path) -> dict:
    root = root.resolve()
    completed = root / "COMPLETED"
    selection_path = root / "selection.jsonl"
    if not completed.is_file() or not selection_path.is_file():
        raise FileNotFoundError("C25 COMPLETED and selection.jsonl are required")
    completion = json.loads(completed.read_text(encoding="utf-8"))
    if completion.get("status") != "PASS_EPISODE_DISJOINT_CAUSAL_DATASET_CANARY":
        raise ValueError("C25 dataset did not pass its preregistered gate")
    selections = [
        json.loads(line)
        for line in selection_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(selections) != 128:
        raise ValueError(f"expected 128 C25 branches, got {len(selections)}")
    branches = [load_branch(root, row) for row in selections]
    grouped: dict[int, list[dict]] = defaultdict(list)
    for branch in branches:
        grouped[int(branch["selection"]["group_id"])].append(branch)
    if sorted(grouped) != list(range(32)):
        raise ValueError("C25 group ids must be contiguous 0..31")

    states = []
    mixed_groups = []
    source_splits: dict[str, set[str]] = defaultdict(set)
    for group_id, items in sorted(grouped.items()):
        if len(items) != 4:
            raise ValueError(f"group {group_id} must contain four branches")
        first = items[0]
        selection = first["selection"]
        for field in ("agentview_image", "wristview_image", "proprio", "sim_state"):
            reference = first[field]
            if not all(torch.equal(reference, item[field]) for item in items[1:]):
                raise ValueError(f"group {group_id} start {field} is not bit-exact")
        for field in ("task_language", "absolute_step"):
            if not all(first[field] == item[field] for item in items[1:]):
                raise ValueError(f"group {group_id} start {field} differs")
        action_hashes = {tensor_sha256(item["environment_actions"]) for item in items}
        if len(action_hashes) != 4:
            raise ValueError(f"group {group_id} candidate action chunks are not unique")
        successes = sum(int(item["success"]) for item in items)
        if 0 < successes < 4:
            mixed_groups.append(group_id)
        source_episode = str(selection["source_episode"])
        split = str(selection["split"])
        source_splits[source_episode].add(split)
        states.append(
            {
                "group_id": group_id,
                "source_episode": source_episode,
                "suite": str(selection["suite"]),
                "task": int(selection["task"]),
                "trial": int(selection["trial"]),
                "split": split,
                "distance_replans": int(selection["distance_replans"]),
                "source_index": int(selection["index"]),
                "task_language": first["task_language"],
                "absolute_step": first["absolute_step"],
                "successes": successes,
                "mixed_outcomes": 0 < successes < 4,
                "agentview_image": first["agentview_image"],
                "wristview_image": first["wristview_image"],
                "proprio": first["proprio"],
                "sim_state": first["sim_state"],
                "agentview_sha256": tensor_sha256(first["agentview_image"]),
                "wristview_sha256": tensor_sha256(first["wristview_image"]),
                "sim_state_sha256": tensor_sha256(first["sim_state"]),
            }
        )
    if not all(len(splits) == 1 for splits in source_splits.values()):
        raise ValueError("source episode appears in more than one split")

    train_mixed = [
        group_id for group_id in mixed_groups if states[group_id]["split"] == "train"
    ]
    val_mixed = [
        group_id for group_id in mixed_groups if states[group_id]["split"] == "val"
    ]
    if len(train_mixed) != 6 or len(val_mixed) != 3:
        raise ValueError(
            f"C25 label yield changed: train/val mixed={len(train_mixed)}/{len(val_mixed)}"
        )
    branch_rows = []
    for branch in branches:
        row = branch["selection"]
        branch_rows.append(
            {
                "ordinal": int(row["ordinal"]),
                "group_id": int(row["group_id"]),
                "split": str(row["split"]),
                "suite": str(row["suite"]),
                "source_episode": str(row["source_episode"]),
                "noise_offset": int(row["noise_offset"]),
                "success": branch["success"],
                "final_step": branch["final_step"],
                "replans": branch["replans"],
                "environment_actions": branch["environment_actions"],
                "action_sha256": tensor_sha256(branch["environment_actions"]),
                "result_sha256": branch["result_sha256"],
                "trajectory_sha256": branch["trajectory_sha256"],
            }
        )
    return {
        "format": FORMAT,
        "classification": "controlled_first_action_outcome_dataset",
        "c25_root": str(root),
        "c25_completed_sha256": sha256_file(completed),
        "c25_selection_sha256": sha256_file(selection_path),
        "branches": branch_rows,
        "states": states,
        "audit": {
            "branch_count": len(branch_rows),
            "group_count": len(states),
            "source_episode_count": len(source_splits),
            "source_episode_split_isolated": True,
            "all_start_states_bit_exact_within_group": True,
            "all_result_actions_match_trajectory": True,
            "all_candidate_chunks_unique_within_group": True,
            "mixed_groups": mixed_groups,
            "train_mixed_groups": train_mixed,
            "val_mixed_groups": val_mixed,
            "train_pairwise_comparisons": sum(
                states[group_id]["successes"] * (4 - states[group_id]["successes"])
                for group_id in train_mixed
            ),
            "val_pairwise_comparisons": sum(
                states[group_id]["successes"] * (4 - states[group_id]["successes"])
                for group_id in val_mixed
            ),
        },
    }


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args.c25_root)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    torch.save(dataset, temporary)
    os.replace(temporary, output)
    print(json.dumps(dataset["audit"], indent=2))


if __name__ == "__main__":
    main()
