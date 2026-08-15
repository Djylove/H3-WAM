#!/usr/bin/env python3
"""Combine C30 train branches with untouched C33 ranking validation."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from prepare_c26_causal_critic_dataset import sha256_file, tensor_sha256
from prepare_c31_action_conditioned_consequence_dataset import (
    consequence_source_splits,
    load_branch,
)


FORMAT = "h3wam-c34-combined-consequence-ranking-dataset-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c30-root", type=Path, required=True)
    parser.add_argument("--c33-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def positive_pairs(successes: int) -> int:
    return int(successes) * (4 - int(successes))


def main() -> None:
    args = parse_args()
    c30_root, c33_root = args.c30_root.resolve(), args.c33_root.resolve()
    c30_done = json.loads((c30_root / "COMPLETED").read_text())
    c33_done = json.loads((c33_root / "COMPLETED").read_text())
    # C30 is intentionally retained as a failed preregistration.  Only its
    # training sources are reused; the insufficient old validation is dropped.
    c30_required = {
        "status": "FAIL_C30_ACTION_CONDITIONED_CAUSAL_DATASET",
        "train_mixed_groups": 25,
        "val_mixed_groups": 3,
        "consequence_observations": 488,
        "source_episode_split_isolated": True,
        "all_seed_schedules_valid": True,
        "all_groups_action_diverse": True,
    }
    for key, expected in c30_required.items():
        if c30_done.get(key) != expected:
            raise ValueError(f"unexpected C30 identity: {key}={c30_done.get(key)!r}")
    if c33_done.get("status") != "PASS_C33_FRESH_RANKING_CAUSAL_DATASET":
        raise ValueError("C33 fresh ranking data gate did not pass")

    c30_selections_all = [
        json.loads(line) for line in (c30_root / "selection.jsonl").read_text().splitlines()
        if line
    ]
    c30_splits = consequence_source_splits(c30_selections_all)
    c30_selections = [row for row in c30_selections_all if row["split"] == "train"]
    c33_selections = [
        json.loads(line) for line in (c33_root / "selection.jsonl").read_text().splitlines()
        if line
    ]
    old_to_new: dict[tuple[str, int], int] = {}
    selections = []
    for origin, rows in (("c30_train", c30_selections), ("c33_fresh_val", c33_selections)):
        for row in rows:
            key = (origin, int(row["group_id"]))
            if key not in old_to_new:
                old_to_new[key] = len(old_to_new)
            selections.append({
                **row,
                "origin": origin,
                "source_ordinal": int(row["ordinal"]),
                "source_group_id": int(row["group_id"]),
                "group_id": old_to_new[key],
                "ordinal": len(selections),
                "split": "train" if origin == "c30_train" else "val",
                "consequence_split": (
                    c30_splits[str(row["source_episode"])]
                    if origin == "c30_train" else "reserved_ranking_val"
                ),
            })
    if len(c30_selections) != 360 or len(c33_selections) != int(c33_done["branches"]):
        raise ValueError("combined C30/C33 branch inventory mismatch")

    loaded = []
    for row in selections:
        source_row = {
            **row,
            "ordinal": row["source_ordinal"],
            "group_id": row["source_group_id"],
            "split": "train" if row["origin"] == "c30_train" else "fresh_ranking_val",
        }
        root = c30_root if row["origin"] == "c30_train" else c33_root
        loaded.append((row, load_branch(root, source_row)))
    grouped: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for row, branch in loaded:
        grouped[int(row["group_id"])].append((row, branch))
    if sorted(grouped) != list(range(len(grouped))):
        raise ValueError("C34 group ids are not contiguous")

    states, branches = [], []
    for group_id, items in sorted(grouped.items()):
        if len(items) != 4:
            raise ValueError(f"C34 group {group_id} does not have four candidates")
        first_row, first = items[0]
        for field in ("agent", "wrist", "proprio", "sim_state"):
            reference = first["current"][field]
            for _, item in items[1:]:
                equal = (
                    torch.equal(reference, item["current"][field])
                    if isinstance(reference, torch.Tensor)
                    else np.array_equal(reference, item["current"][field])
                )
                if not equal:
                    raise ValueError(f"group {group_id} current {field} differs")
        if len({tensor_sha256(item["proposed_environment_actions"]) for _, item in items}) != 4:
            raise ValueError(f"group {group_id} proposed actions are not unique")
        successes = sum(int(item["success"]) for _, item in items)
        states.append({
            "group_id": group_id, "origin": first_row["origin"],
            "source_episode": first_row["source_episode"], "suite": first_row["suite"],
            "task": int(first_row["task"]), "trial": int(first_row["trial"]),
            "split": first_row["split"],
            "consequence_split": first_row["consequence_split"],
            "distance_replans": int(first_row["distance_replans"]),
            "task_language": first["task_language"],
            "absolute_step": int(first["current"]["step"]),
            "successes": successes, "mixed_outcomes": 0 < successes < 4,
            "agentview_image": torch.from_numpy(first["current"]["agent"]),
            "wristview_image": torch.from_numpy(first["current"]["wrist"]),
            "proprio": first["current"]["proprio"],
            "sim_state": first["current"]["sim_state"],
        })
        for row, item in items:
            future = item["future"]
            branches.append({
                "ordinal": int(row["ordinal"]), "group_id": group_id,
                "origin": row["origin"], "source_ordinal": row["source_ordinal"],
                "source_episode": row["source_episode"], "suite": row["suite"],
                "split": row["split"], "consequence_split": row["consequence_split"],
                "noise_offset": int(row["noise_offset"]), "success": item["success"],
                "proposed_environment_actions": item["proposed_environment_actions"],
                "environment_actions": item["executed_environment_actions"],
                "action_is_pad": item["action_is_pad"],
                "executed_action_steps": item["executed_action_steps"],
                "future_agentview_image": torch.from_numpy(future["agent"]),
                "future_wristview_image": torch.from_numpy(future["wrist"]),
                "future_proprio": future["proprio"],
                "future_sim_state": future["sim_state"],
                "future_step": int(future["step"]), "future_source": future["source"],
                "result_sha256": item["result_sha256"],
                "trajectory_sha256": item["trajectory_sha256"],
            })
    # Branches were loaded in source order; restore the declared contiguous order.
    branches.sort(key=lambda row: row["ordinal"])
    train_mixed = [s for s in states if s["split"] == "train" and s["mixed_outcomes"]]
    val_mixed = [s for s in states if s["split"] == "val" and s["mixed_outcomes"]]
    payload = {
        "format": FORMAT,
        "c30_root": str(c30_root), "c33_root": str(c33_root),
        "c30_completed_sha256": sha256_file(c30_root / "COMPLETED"),
        "c33_completed_sha256": sha256_file(c33_root / "COMPLETED"),
        "states": states, "branches": branches,
        "audit": {
            "states": len(states), "branches": len(branches),
            "c30_old_validation_excluded": True,
            "all_current_states_bit_exact_within_group": True,
            "all_proposed_actions_match_results_and_trajectories": True,
            "unexecuted_action_tails_zero_masked": True,
            "all_branches_have_post_action_consequence": True,
            "consequence_train_sources": len({s["source_episode"] for s in states if s["consequence_split"] == "train"}),
            "consequence_validation_sources": len({s["source_episode"] for s in states if s["consequence_split"] == "validation"}),
            "fresh_ranking_validation_sources": len({s["source_episode"] for s in states if s["split"] == "val"}),
            "train_mixed_groups": len(train_mixed), "val_mixed_groups": len(val_mixed),
            "train_pairs": sum(positive_pairs(s["successes"]) for s in train_mixed),
            "val_pairs": sum(positive_pairs(s["successes"]) for s in val_mixed),
            "terminal_fallback_branches": sum(b["future_source"] == "terminal_within_first_chunk" for b in branches),
            "partial_action_branches": sum(b["executed_action_steps"] < 32 for b in branches),
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
