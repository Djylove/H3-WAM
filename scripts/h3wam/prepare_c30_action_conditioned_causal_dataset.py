#!/usr/bin/env python3
"""Freeze fresh C29 successes into C30 action-conditioned causal branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


FORMAT = "h3wam-c30-action-conditioned-causal-dataset-v1"
SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")
TRIALS = (4, 5, 6, 7)
TASKS = tuple(range(10))
DISTANCES = (3, 5)
OFFSETS = (0, 1_000_000, 2_000_000, 3_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/mnt/h3-wam"))
    parser.add_argument("--c29-completed", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def deterministic_split(records: list[dict]) -> dict[str, str]:
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_suite[record["suite"]].append(record)
    result = {}
    for suite, items in sorted(by_suite.items()):
        ordered = sorted(
            items,
            key=lambda row: hashlib.sha256(
                ("c30-consequence-v1:" + row["source_episode"]).encode()
            ).digest(),
        )
        validation_count = max(1, round(len(ordered) * 0.25))
        for index, row in enumerate(ordered):
            result[row["source_episode"]] = "val" if index < validation_count else "train"
    return result


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    c29_path = args.c29_completed.resolve()
    c29 = json.loads(c29_path.read_text(encoding="utf-8"))
    if c29.get("status") != "PASS_C29_FRESH_PARENT_SOURCE_EXPANSION":
        raise ValueError("C29 source expansion did not pass")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    (output / "runs").mkdir(parents=True)
    (output / "logs").mkdir()

    result_root = workspace / "outputs" / "eval-dense-d0-long"
    records = []
    ineligible_short = []
    for suite in SUITES:
        slug = suite.removeprefix("libero_")
        for task in TASKS:
            for trial in TRIALS:
                directory = result_root / f"d0_h32_s14000_{slug}_task{task}_trial{trial}_replan8"
                result_path = directory / "results.json"
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                episode = payload["tasks"][0]["episodes"][0]
                if not bool(episode["success"]):
                    continue
                trajectory = directory / f"task{task:02d}_trial{trial:02d}_trajectory.npz"
                with np.load(trajectory, allow_pickle=False) as archive:
                    state_count = int(archive["step"].shape[0])
                available = [distance for distance in DISTANCES if state_count > distance]
                source_episode = f"{suite}:task{task}:trial{trial}"
                if not available:
                    ineligible_short.append(source_episode)
                    continue
                records.append({
                    "source_episode": source_episode, "suite": suite,
                    "task": task, "trial": trial, "trajectory": str(trajectory),
                    "source_result": str(result_path), "state_count": state_count,
                    "available_distances": available,
                })
    suite_counts = defaultdict(int)
    for record in records:
        suite_counts[record["suite"]] += 1
    if len(records) < 32:
        raise ValueError(f"fewer than 32 eligible successful sources: {len(records)}")
    splits = deterministic_split(records)
    groups, selections = [], []
    for record in sorted(records, key=lambda row: row["source_episode"]):
        for distance in record["available_distances"]:
            group_id = len(groups)
            index = int(record["state_count"]) - int(distance)
            continuation = 80_000_000 + group_id * 100_000
            group = {
                **record, "group_id": group_id,
                "split": splits[record["source_episode"]],
                "distance_replans": distance, "index": index,
                "continuation_policy_noise_seed_base": continuation,
            }
            groups.append(group)
            base_seed = 42 + record["task"] * 100_000 + record["trial"] * 1_000 + index
            for offset in OFFSETS:
                selections.append({
                    "ordinal": len(selections), "group_id": group_id,
                    "source_episode": record["source_episode"], "suite": record["suite"],
                    "task": record["task"], "trial": record["trial"],
                    "split": group["split"], "distance_replans": distance,
                    "index": index, "trajectory": record["trajectory"],
                    "first_policy_noise_seed": base_seed + offset,
                    "noise_offset": offset,
                    "continuation_policy_noise_seed_base": continuation,
                })
    if len(groups) < 50 or len(selections) != 4 * len(groups):
        raise ValueError(f"insufficient C30 groups/branches: {len(groups)}/{len(selections)}")
    selection_path = output / "selection.jsonl"
    selection_path.write_text(
        "".join(json.dumps(row) + "\n" for row in selections), encoding="utf-8"
    )
    train_sources = {key for key, split in splits.items() if split == "train"}
    val_sources = set(splits) - train_sources
    prereg = {
        "format": FORMAT,
        "hypothesis": "Fresh terminal-complete causal branches provide action-conditioned future targets and at least ten train/four validation mixed groups across three suites.",
        "parent": "D0-H32-s14000/replan8/no-ensemble on fresh C29 trials4..7",
        "only_variable_within_group": "first-replan policy diffusion noise seed",
        "execution_contract": "first chunk32; continuation replan8 with identical seed schedule",
        "consequence_contract": "row1 observation at start+32 when present, otherwise post-action terminal observation; terminal fields required for every branch",
        "split_contract": "suite-stratified source-episode hash split frozen before branch outcomes",
        "source_episodes": len(records), "train_source_episodes": len(train_sources),
        "val_source_episodes": len(val_sources), "groups": len(groups),
        "branches": len(selections),
        "train_groups": sum(g["split"] == "train" for g in groups),
        "val_groups": sum(g["split"] == "val" for g in groups),
        "suite_source_counts": dict(sorted(suite_counts.items())),
        "ineligible_short_sources": sorted(ineligible_short),
        "groups_detail": groups,
        "promotion": "all mechanical and consequence observation checks; train mixed>=10; val mixed>=4; mixed covers>=3 suites",
        "pass_permission": "GO_ACTION_CONDITIONED_CONSEQUENCE_TRAINING",
        "fail_permission": "NO_GO_ACTION_CONDITIONED_CONSEQUENCE_TRAINING",
        "effect_conclusion": "NOT_EVIDENCE_READY",
        "c29_completed": str(c29_path),
        "max_environment_steps": len(selections) * 400,
    }
    atomic_json(output / "preregistration.json", prereg)
    print(json.dumps({
        "output_root": str(output), "sources": len(records),
        "suite_source_counts": dict(sorted(suite_counts.items())),
        "groups": len(groups), "branches": len(selections),
        "train_sources": len(train_sources), "val_sources": len(val_sources),
    }, indent=2))


if __name__ == "__main__":
    main()
