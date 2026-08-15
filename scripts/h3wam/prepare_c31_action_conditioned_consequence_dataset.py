#!/usr/bin/env python3
"""Freeze passed C30 branches into causal current/action/future/value records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from prepare_c26_causal_critic_dataset import sha256_file, tensor_sha256
from fastwam.models.h3wam import libero_observation_state


FORMAT = "h3wam-c31-action-conditioned-consequence-dataset-v1"
TERMINAL_FIELDS = (
    "terminal_step", "terminal_agentview_image", "terminal_wristview_image", "terminal_eef_pos",
    "terminal_eef_quat", "terminal_gripper_qpos", "terminal_sim_state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c30-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run_name(row: dict) -> str:
    slug = str(row["suite"]).removeprefix("libero_")
    return (
        f"{row['ordinal']}_g{row['group_id']}_{slug}_task{row['task']}_"
        f"trial{row['trial']}_d{row['distance_replans']}_offset{row['noise_offset']}"
    )


def observation_proprio(
    eef_pos: np.ndarray, eef_quat: np.ndarray, gripper_qpos: np.ndarray
) -> torch.Tensor:
    return libero_observation_state({
        "eef_pos": eef_pos, "eef_quat": eef_quat,
        "gripper_qpos": gripper_qpos,
    }).float()


def mask_unexecuted_action_tail(
    proposed_actions: np.ndarray, *, executed_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Zero the unexecuted tail while retaining an explicit padding mask."""

    actions = np.asarray(proposed_actions, dtype=np.float32)
    if actions.shape != (32, 7):
        raise ValueError(f"proposed actions must be [32,7], got {actions.shape}")
    if not 1 <= int(executed_steps) <= 32:
        raise ValueError(f"executed_steps must be in [1,32], got {executed_steps}")
    action_is_pad = np.arange(32) >= int(executed_steps)
    executed_actions = actions.copy()
    executed_actions[action_is_pad] = 0.0
    return executed_actions, action_is_pad


def consequence_source_splits(selections: list[dict]) -> dict[str, str]:
    """Reserve original C30 val and carve tune sources from C30 train only."""

    by_suite: dict[str, set[str]] = defaultdict(set)
    result: dict[str, str] = {}
    for row in selections:
        source = str(row["source_episode"])
        if row["split"] == "val":
            result[source] = "reserved_ranking_val"
        elif row["split"] == "train":
            by_suite[str(row["suite"])].add(source)
        else:
            raise ValueError(f"unknown C30 split: {row['split']}")
    for suite, sources in sorted(by_suite.items()):
        ordered = sorted(
            sources,
            key=lambda source: hashlib.sha256(
                f"c31-consequence-validation-v1|{suite}|{source}".encode()
            ).hexdigest(),
        )
        validation_count = max(1, len(ordered) // 5)
        validation = set(ordered[:validation_count])
        for source in ordered:
            result[source] = "validation" if source in validation else "train"
    return result


def load_branch(root: Path, row: dict) -> dict:
    directory = root / "runs" / run_name(row)
    result_path = directory / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    episode = result["tasks"][0]["episodes"][0]
    trajectories = tuple(directory.glob("*trajectory.npz"))
    if len(trajectories) != 1:
        raise ValueError(f"expected one C30 trajectory in {directory}")
    trajectory_path = trajectories[0]
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        missing = [field for field in TERMINAL_FIELDS if field not in trajectory.files]
        if missing:
            raise ValueError(f"missing terminal fields in {directory}: {missing}")
        proposed_actions = np.asarray(
            episode["first_environment_action_chunk"], dtype=np.float32
        )
        stored_actions = np.asarray(trajectory["policy_actions"][0], dtype=np.float32)
        if proposed_actions.shape != (32, 7) or not np.array_equal(
            proposed_actions, stored_actions
        ):
            raise ValueError(f"first action mismatch in {directory}")
        current_obs = {
            "agent": np.asarray(trajectory["agentview_image"][0], dtype=np.uint8).copy(),
            "wrist": np.asarray(trajectory["wristview_image"][0], dtype=np.uint8).copy(),
            "proprio": observation_proprio(
                trajectory["eef_pos"][0], trajectory["eef_quat"][0],
                trajectory["gripper_qpos"][0],
            ),
            "sim_state": torch.from_numpy(
                np.asarray(trajectory["sim_state"][0], dtype=np.float64).copy()
            ),
            "step": int(trajectory["step"][0]),
        }
        if len(trajectory["step"]) >= 2:
            future_index = 1
            if int(trajectory["step"][future_index]) != current_obs["step"] + 32:
                raise ValueError(f"future row is not start+32 in {directory}")
            future = {
                "agent": np.asarray(trajectory["agentview_image"][future_index], dtype=np.uint8).copy(),
                "wrist": np.asarray(trajectory["wristview_image"][future_index], dtype=np.uint8).copy(),
                "proprio": observation_proprio(
                    trajectory["eef_pos"][future_index], trajectory["eef_quat"][future_index],
                    trajectory["gripper_qpos"][future_index],
                ),
                "sim_state": torch.from_numpy(
                    np.asarray(trajectory["sim_state"][future_index], dtype=np.float64).copy()
                ),
                "step": int(trajectory["step"][future_index]),
                "source": "replan_row_1",
            }
        else:
            terminal_step = int(trajectory["terminal_step"])
            if not current_obs["step"] < terminal_step <= current_obs["step"] + 32:
                raise ValueError(f"terminal consequence is outside first chunk in {directory}")
            future = {
                "agent": np.asarray(trajectory["terminal_agentview_image"], dtype=np.uint8).copy(),
                "wrist": np.asarray(trajectory["terminal_wristview_image"], dtype=np.uint8).copy(),
                "proprio": observation_proprio(
                    trajectory["terminal_eef_pos"], trajectory["terminal_eef_quat"],
                    trajectory["terminal_gripper_qpos"],
                ),
                "sim_state": torch.from_numpy(
                    np.asarray(trajectory["terminal_sim_state"], dtype=np.float64).copy()
                ),
                "step": terminal_step,
                "source": "terminal_within_first_chunk",
            }
    executed_steps = int(future["step"] - current_obs["step"])
    if not 1 <= executed_steps <= 32:
        raise ValueError(f"invalid executed action length in {directory}: {executed_steps}")
    executed_actions, action_is_pad = mask_unexecuted_action_tail(
        proposed_actions, executed_steps=executed_steps
    )
    return {
        "selection": row, "task_language": str(result["tasks"][0]["task"]),
        "success": bool(episode["success"]),
        # Preserve the complete policy proposal for intervention identity, but
        # expose only the prefix that the environment actually consumed to the
        # causal consequence learner.  A terminal success can stop before the
        # nominal 32-step chunk finishes; treating that tail as executed would
        # create a false action -> future edge.
        "proposed_environment_actions": torch.from_numpy(proposed_actions.copy()),
        "executed_environment_actions": torch.from_numpy(executed_actions),
        "action_is_pad": torch.from_numpy(action_is_pad),
        "executed_action_steps": executed_steps,
        "current": current_obs, "future": future,
        "result_sha256": sha256_file(result_path),
        "trajectory_sha256": sha256_file(trajectory_path),
    }


def main() -> None:
    args = parse_args()
    root = args.c30_root.resolve()
    completed = root / "COMPLETED"
    selection_path = root / "selection.jsonl"
    completion = json.loads(completed.read_text(encoding="utf-8"))
    if completion.get("status") != "PASS_C30_ACTION_CONDITIONED_CAUSAL_DATASET":
        raise ValueError("C30 data gate did not pass")
    selections = [json.loads(line) for line in selection_path.read_text().splitlines() if line]
    if len(selections) != int(completion["branches"]):
        raise ValueError("C30 completion/selection branch count mismatch")
    loaded = [load_branch(root, row) for row in selections]
    consequence_splits = consequence_source_splits(selections)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for branch in loaded:
        grouped[int(branch["selection"]["group_id"])].append(branch)
    if sorted(grouped) != list(range(int(completion["groups"]))):
        raise ValueError("C30 group ids are not contiguous")
    states, branches = [], []
    for group_id, items in sorted(grouped.items()):
        if len(items) != 4:
            raise ValueError(f"C30 group {group_id} has {len(items)} branches")
        first = items[0]
        for field in ("agent", "wrist", "proprio", "sim_state"):
            reference = first["current"][field]
            if not all(
                torch.equal(reference, item["current"][field])
                if isinstance(reference, torch.Tensor)
                else np.array_equal(reference, item["current"][field])
                for item in items[1:]
            ):
                raise ValueError(f"group {group_id} current {field} differs")
        if len({tensor_sha256(item["proposed_environment_actions"]) for item in items}) != 4:
            raise ValueError(f"group {group_id} proposed candidate actions are not unique")
        selection = first["selection"]
        successes = sum(int(item["success"]) for item in items)
        states.append({
            "group_id": group_id, "source_episode": selection["source_episode"],
            "suite": selection["suite"], "task": int(selection["task"]),
            "trial": int(selection["trial"]), "split": selection["split"],
            "consequence_split": consequence_splits[str(selection["source_episode"])],
            "distance_replans": int(selection["distance_replans"]),
            "task_language": first["task_language"],
            "absolute_step": first["current"]["step"],
            "successes": successes, "mixed_outcomes": 0 < successes < 4,
            "agentview_image": torch.from_numpy(first["current"]["agent"]),
            "wristview_image": torch.from_numpy(first["current"]["wrist"]),
            "proprio": first["current"]["proprio"],
            "sim_state": first["current"]["sim_state"],
        })
        for item in items:
            row = item["selection"]
            future = item["future"]
            branches.append({
                "ordinal": int(row["ordinal"]), "group_id": group_id,
                "source_episode": row["source_episode"], "suite": row["suite"],
                "split": row["split"], "noise_offset": int(row["noise_offset"]),
                "consequence_split": consequence_splits[str(row["source_episode"])],
                "success": item["success"],
                "proposed_environment_actions": item["proposed_environment_actions"],
                "environment_actions": item["executed_environment_actions"],
                "action_is_pad": item["action_is_pad"],
                "executed_action_steps": item["executed_action_steps"],
                "future_agentview_image": torch.from_numpy(future["agent"]),
                "future_wristview_image": torch.from_numpy(future["wrist"]),
                "future_proprio": future["proprio"],
                "future_sim_state": future["sim_state"],
                "future_step": int(future["step"]),
                "future_source": future["source"],
                "result_sha256": item["result_sha256"],
                "trajectory_sha256": item["trajectory_sha256"],
            })
    train_mixed = [s["group_id"] for s in states if s["split"] == "train" and s["mixed_outcomes"]]
    val_mixed = [s["group_id"] for s in states if s["split"] == "val" and s["mixed_outcomes"]]
    payload = {
        "format": FORMAT, "c30_root": str(root),
        "c30_completed_sha256": sha256_file(completed),
        "c30_selection_sha256": sha256_file(selection_path),
        "states": states, "branches": branches,
        "audit": {
            "states": len(states), "branches": len(branches),
            "all_current_states_bit_exact_within_group": True,
            "all_proposed_actions_match_results_and_trajectories": True,
            "unexecuted_action_tails_zero_masked": True,
            "all_branches_have_post_action_consequence": True,
            "terminal_fallback_branches": sum(b["future_source"] == "terminal_within_first_chunk" for b in branches),
            "partial_action_branches": sum(b["executed_action_steps"] < 32 for b in branches),
            "consequence_train_sources": len({
                s["source_episode"] for s in states if s["consequence_split"] == "train"
            }),
            "consequence_validation_sources": len({
                s["source_episode"] for s in states if s["consequence_split"] == "validation"
            }),
            "reserved_ranking_validation_sources": len({
                s["source_episode"] for s in states
                if s["consequence_split"] == "reserved_ranking_val"
            }),
            "train_mixed_groups": train_mixed, "val_mixed_groups": val_mixed,
            "train_pairs": sum(states[g]["successes"] * (4 - states[g]["successes"]) for g in train_mixed),
            "val_pairs": sum(states[g]["successes"] * (4 - states[g]["successes"]) for g in val_mixed),
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
