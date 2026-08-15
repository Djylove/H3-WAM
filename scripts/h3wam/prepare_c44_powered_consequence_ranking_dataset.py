#!/usr/bin/env python3
"""Combine frozen C34 ranker-train data with source-disjoint C43 train/final data."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from prepare_c26_causal_critic_dataset import sha256_file, tensor_sha256
from prepare_c31_action_conditioned_consequence_dataset import load_branch


FORMAT = "h3wam-c44-powered-consequence-ranking-dataset-v1"
EXPECTED_C34_SHA256 = "2a6c9252b8e77975f58920425bc18110fa8ea63bdc12c4c15571cfffeb9f7459"


def positive_pairs(successes: int) -> int:
    return int(successes) * (4 - int(successes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c34-dataset", type=Path, required=True)
    parser.add_argument("--c43-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    c34_path = args.c34_dataset.resolve()
    c43_root = args.c43_root.resolve()
    if sha256_file(c34_path) != EXPECTED_C34_SHA256:
        raise ValueError("C34 dataset identity mismatch")
    c34 = torch.load(c34_path, map_location="cpu", weights_only=False)
    if c34.get("format") != "h3wam-c34-combined-consequence-ranking-dataset-v1":
        raise ValueError("unexpected C34 format")
    c43_done_path = c43_root / "COMPLETED"
    c43_done = json.loads(c43_done_path.read_text())
    if c43_done.get("status") != "PASS_C43_POWERED_CAUSAL_RANKING_DATASET":
        raise ValueError("C43 data gate did not pass")
    c43_rows = [
        json.loads(line) for line in (c43_root / "selection.jsonl").read_text().splitlines()
        if line
    ]
    if len(c43_rows) != int(c43_done["branches"]):
        raise ValueError("C43 selection/completion inventory mismatch")

    states, branches = [], []

    # Retain exactly the 37-source consequence-train slice used by C40.  The
    # consumed C34 consequence-validation and C33 final sources stay excluded.
    legacy_group_map: dict[int, int] = {}
    for state in c34["states"]:
        if state["consequence_split"] != "train":
            continue
        old_group = int(state["group_id"])
        new_group = len(states)
        legacy_group_map[old_group] = new_group
        states.append({
            **state,
            "group_id": new_group,
            "origin": "c34_consequence_train",
            "split": "train",
            "consequence_split": "ranker_train",
        })
    for branch in c34["branches"]:
        old_group = int(branch["group_id"])
        if old_group not in legacy_group_map:
            continue
        branches.append({
            **branch,
            "ordinal": len(branches),
            "group_id": legacy_group_map[old_group],
            "origin": "c34_consequence_train",
            "split": "train",
            "consequence_split": "ranker_train",
        })
    if len(legacy_group_map) != 74 or len(branches) != 296:
        raise ValueError("legacy C34 ranker-train inventory mismatch")

    loaded = [(row, load_branch(c43_root, row)) for row in c43_rows]
    grouped: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for row, branch in loaded:
        grouped[int(row["group_id"])].append((row, branch))
    if sorted(grouped) != list(range(int(c43_done["groups"]))):
        raise ValueError("C43 group ids are not contiguous")

    c43_sources_by_split: dict[str, set[str]] = defaultdict(set)
    for _, items in sorted(grouped.items()):
        if len(items) != 4:
            raise ValueError("C43 group does not have four branches")
        first_row, first = items[0]
        split = str(first_row["split"])
        consequence_split = (
            "ranker_train" if split == "train" else "reserved_powered_final"
        )
        if split not in {"train", "fresh_final"}:
            raise ValueError(f"unknown C43 split: {split}")
        if any(str(row["split"]) != split for row, _ in items):
            raise ValueError("C43 group crosses splits")
        for field in ("agent", "wrist", "proprio", "sim_state"):
            reference = first["current"][field]
            for _, item in items[1:]:
                equal = (
                    torch.equal(reference, item["current"][field])
                    if isinstance(reference, torch.Tensor)
                    else np.array_equal(reference, item["current"][field])
                )
                if not equal:
                    raise ValueError(f"C43 current {field} differs within group")
        if len({tensor_sha256(item["proposed_environment_actions"]) for _, item in items}) != 4:
            raise ValueError("C43 proposed candidate actions are not unique")
        group_id = len(states)
        successes = sum(int(item["success"]) for _, item in items)
        source = str(first_row["source_episode"])
        c43_sources_by_split[split].add(source)
        states.append({
            "group_id": group_id,
            "origin": "c43_powered_train" if split == "train" else "c43_powered_final",
            "source_episode": source,
            "suite": first_row["suite"],
            "task": int(first_row["task"]),
            "trial": int(first_row["trial"]),
            "split": split,
            "consequence_split": consequence_split,
            "distance_replans": int(first_row["distance_replans"]),
            "task_language": first["task_language"],
            "absolute_step": int(first["current"]["step"]),
            "successes": successes,
            "mixed_outcomes": 0 < successes < 4,
            "agentview_image": torch.from_numpy(first["current"]["agent"]),
            "wristview_image": torch.from_numpy(first["current"]["wrist"]),
            "proprio": first["current"]["proprio"],
            "sim_state": first["current"]["sim_state"],
        })
        for row, item in items:
            future = item["future"]
            branches.append({
                "ordinal": len(branches),
                "group_id": group_id,
                "origin": "c43_powered_train" if split == "train" else "c43_powered_final",
                "source_ordinal": int(row["ordinal"]),
                "source_episode": source,
                "suite": row["suite"],
                "split": split,
                "consequence_split": consequence_split,
                "noise_offset": int(row["noise_offset"]),
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

    if [int(state["group_id"]) for state in states] != list(range(len(states))):
        raise ValueError("C44 state ids are not contiguous")
    if [int(branch["ordinal"]) for branch in branches] != list(range(len(branches))):
        raise ValueError("C44 branch ids are not contiguous")
    if len(branches) != 4 * len(states):
        raise ValueError("C44 state/branch ratio mismatch")
    train_mixed = [
        state for state in states
        if state["consequence_split"] == "ranker_train" and state["mixed_outcomes"]
    ]
    final_mixed = [
        state for state in states
        if state["consequence_split"] == "reserved_powered_final" and state["mixed_outcomes"]
    ]
    if len(train_mixed) < 46 or len(final_mixed) < 24:
        raise ValueError("C44 powered mixed-group gate was not preserved")
    if c43_sources_by_split["train"] & c43_sources_by_split["fresh_final"]:
        raise ValueError("C43 train/final source overlap")
    payload = {
        "format": FORMAT,
        "c34_dataset": str(c34_path),
        "c34_dataset_sha256": EXPECTED_C34_SHA256,
        "c43_root": str(c43_root),
        "c43_completed_sha256": sha256_file(c43_done_path),
        "states": states,
        "branches": branches,
        "audit": {
            "states": len(states),
            "branches": len(branches),
            "legacy_c34_train_states": len(legacy_group_map),
            "consumed_c34_validation_and_c33_final_excluded": True,
            "all_current_states_bit_exact_within_group": True,
            "all_proposed_actions_match_results_and_trajectories": True,
            "unexecuted_action_tails_zero_masked": True,
            "all_branches_have_post_action_consequence": True,
            "ranker_train_sources": len({
                state["source_episode"] for state in states
                if state["consequence_split"] == "ranker_train"
            }),
            "powered_final_sources": len(c43_sources_by_split["fresh_final"]),
            "train_mixed_groups": len(train_mixed),
            "final_mixed_groups": len(final_mixed),
            "train_pairs": sum(positive_pairs(state["successes"]) for state in train_mixed),
            "final_pairs": sum(positive_pairs(state["successes"]) for state in final_mixed),
            "terminal_fallback_branches": sum(
                branch["future_source"] == "terminal_within_first_chunk" for branch in branches
            ),
            "partial_action_branches": sum(
                branch["executed_action_steps"] < 32 for branch in branches
            ),
        },
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print(json.dumps(payload["audit"], indent=2))


if __name__ == "__main__":
    main()
