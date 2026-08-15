#!/usr/bin/env python3
"""Freeze an episode-disjoint, multisuite causal-action dataset canary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


SOURCES = (
    ("libero_goal", 4, 1, "train", (3, 5)),
    ("libero_goal", 5, 1, "train", (3, 5)),
    ("libero_goal", 7, 0, "train", (3, 5)),
    ("libero_goal", 9, 3, "val", (3, 5)),
    ("libero_object", 1, 0, "train", (3, 5)),
    ("libero_object", 2, 0, "train", (3, 5)),
    ("libero_object", 4, 0, "train", (3, 5)),
    ("libero_object", 3, 3, "val", (3, 5)),
    ("libero_spatial", 1, 1, "train", (3, 5)),
    ("libero_spatial", 2, 1, "train", (3, 5)),
    ("libero_spatial", 3, 0, "train", (3, 5)),
    ("libero_spatial", 4, 3, "val", (3, 5)),
    ("libero_10", 5, 0, "train", (1, 3, 4, 5)),
    ("libero_10", 3, 1, "val", (1, 3, 4, 5)),
)
OFFSETS = (0, 1_000_000, 2_000_000, 3_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/mnt/h3-wam"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    (output / "runs").mkdir(parents=True)
    (output / "logs").mkdir()
    source_root = args.workspace.resolve() / "outputs/eval-dense-d0-long"
    rows = []
    groups = []
    source_splits = {}
    for suite, task, trial, split, distances in SOURCES:
        source_key = f"{suite}:task{task}:trial{trial}"
        if source_key in source_splits and source_splits[source_key] != split:
            raise ValueError(f"source episode split collision: {source_key}")
        source_splits[source_key] = split
        slug = suite.removeprefix("libero_")
        source_run = source_root / f"d0_h32_s14000_{slug}_task{task}_trial{trial}_replan8"
        payload = json.loads((source_run / "results.json").read_text())
        episode = payload["tasks"][0]["episodes"][0]
        if not episode["success"]:
            raise ValueError(f"C25 source is not successful: {source_run}")
        trajectory = source_run / f"task{task:02d}_trial{trial:02d}_trajectory.npz"
        with np.load(trajectory, allow_pickle=False) as archive:
            state_count = int(archive["step"].shape[0])
        for distance in distances:
            index = state_count - distance
            if index < 0:
                raise ValueError(f"source too short at distance {distance}: {trajectory}")
            group_id = len(groups)
            continuation_seed = 40_000_000 + group_id * 100_000
            groups.append(
                {
                    "group_id": group_id, "source_episode": source_key,
                    "suite": suite, "task": task, "trial": trial, "split": split,
                    "distance_replans": distance, "index": index,
                    "trajectory": str(trajectory.resolve()),
                    "continuation_policy_noise_seed_base": continuation_seed,
                }
            )
            base_seed = 42 + task * 100_000 + trial * 1_000 + index
            for offset in OFFSETS:
                rows.append(
                    {
                        "ordinal": len(rows), "group_id": group_id,
                        "source_episode": source_key, "suite": suite, "task": task,
                        "trial": trial, "split": split, "distance_replans": distance,
                        "index": index, "trajectory": str(trajectory.resolve()),
                        "first_policy_noise_seed": base_seed + offset,
                        "noise_offset": offset,
                        "continuation_policy_noise_seed_base": continuation_seed,
                    }
                )
    if len(groups) != 32 or len(rows) != 128:
        raise AssertionError(f"C25 expected 32 groups/128 rows, got {len(groups)}/{len(rows)}")
    (output / "selection.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    preregistration = {
        "format": "h3wam-c25-episode-disjoint-causal-dataset-canary-v1",
        "falsifiable_hypothesis": (
            "Horizon32 first-action interventions yield at least eight mixed groups, with "
            "at least four train and two held-out validation groups, over at least three suites."
        ),
        "parent": "C24 selected horizon32 + D0-H32-s14000/replan8/no ensemble",
        "only_variable_within_group": "first-replan policy diffusion noise seed",
        "execution_contract": "first chunk32, all continuation replans8, fixed continuation seeds",
        "split_contract": "all states/branches from one source episode remain in one split",
        "source_episodes": len(source_splits), "groups": 32, "branches": 128,
        "train_groups": sum(group["split"] == "train" for group in groups),
        "val_groups": sum(group["split"] == "val" for group in groups),
        "max_environment_steps": 51_200,
        "groups_detail": groups,
        "promotion": (
            "mechanical gates pass; >=8 mixed groups total; >=4 train and >=2 val mixed; "
            "mixed groups span >=3 suites"
        ),
        "pass_permission": "GO_FROZEN_H3_ACTION_CRITIC_CANARY",
        "fail_permission": "NO_GO_GENERAL_ACTION_CRITIC_DATA",
        "effect_conclusion": "NOT_EVIDENCE_READY",
    }
    atomic_json(output / "preregistration.json", preregistration)
    print(json.dumps({
        "output_root": str(output), "source_episodes": len(source_splits),
        "groups": len(groups), "branches": len(rows),
        "train_groups": preregistration["train_groups"],
        "val_groups": preregistration["val_groups"],
    }, indent=2))


if __name__ == "__main__":
    main()
