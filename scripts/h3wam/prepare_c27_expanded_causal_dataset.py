#!/usr/bin/env python3
"""Freeze fresh successful source episodes for the expanded C27 causal set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


FORMAT = "h3wam-c27-expanded-causal-dataset-v1"
SOURCE_PATTERN = re.compile(
    r"d0_h32_s14000_(goal|object|spatial|10)_task(\d+)_trial(\d+)_replan8"
)
DISTANCES = (3, 5)
OFFSETS = (0, 1_000_000, 2_000_000, 3_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/mnt/h3-wam"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--c22-preregistration",
        type=Path,
        default=Path("/mnt/h3-wam/eval/c22-counterfactual-entropy-sweep-v1/preregistration.json"),
    )
    parser.add_argument(
        "--c25-preregistration",
        type=Path,
        default=Path("/mnt/h3-wam/eval/c25-episode-disjoint-causal-dataset-v1/preregistration.json"),
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_key(suite: str, task: int, trial: int) -> str:
    return f"{suite}:task{task}:trial{trial}"


def previously_used(c22_path: Path, c25_path: Path) -> set[str]:
    c22 = json.loads(c22_path.read_text(encoding="utf-8"))
    c25 = json.loads(c25_path.read_text(encoding="utf-8"))
    used = {
        source_key(str(row["suite"]), int(row["task"]), int(row["trial"]))
        for row in c22["source_episodes"]
    }
    used.update(str(row["source_episode"]) for row in c25["groups_detail"])
    return used


def deterministic_split(records: list[dict]) -> dict[str, str]:
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_suite[record["suite"]].append(record)
    splits = {}
    for suite, items in sorted(by_suite.items()):
        ordered = sorted(
            items,
            key=lambda row: hashlib.sha256(row["source_episode"].encode()).digest(),
        )
        validation_count = max(1, round(len(ordered) * 0.25))
        for index, row in enumerate(ordered):
            splits[row["source_episode"]] = "val" if index < validation_count else "train"
    return splits


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    (output / "runs").mkdir(parents=True)
    (output / "logs").mkdir()
    source_root = args.workspace.resolve() / "outputs/eval-dense-d0-long"
    used = previously_used(args.c22_preregistration.resolve(), args.c25_preregistration.resolve())
    records = []
    for directory in sorted(source_root.iterdir()):
        match = SOURCE_PATTERN.fullmatch(directory.name)
        if match is None:
            continue
        suite = f"libero_{match.group(1)}"
        task, trial = int(match.group(2)), int(match.group(3))
        key = source_key(suite, task, trial)
        if key in used:
            continue
        result_path = directory / "results.json"
        trajectory = directory / f"task{task:02d}_trial{trial:02d}_trajectory.npz"
        if not result_path.is_file() or not trajectory.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        episode = result["tasks"][0]["episodes"][0]
        if not episode["success"]:
            continue
        with np.load(trajectory, allow_pickle=False) as archive:
            state_count = int(archive["step"].shape[0])
        if state_count <= max(DISTANCES):
            raise ValueError(f"source trajectory is too short: {trajectory}")
        records.append({
            "source_episode": key, "suite": suite, "task": task, "trial": trial,
            "trajectory": str(trajectory.resolve()), "state_count": state_count,
            "source_result": str(result_path.resolve()), "source_success": True,
        })
    suite_counts = defaultdict(int)
    for record in records:
        suite_counts[record["suite"]] += 1
    expected = {"libero_goal": 5, "libero_object": 22, "libero_spatial": 12}
    if dict(suite_counts) != expected or len(records) != 39:
        raise ValueError(
            f"fresh successful source inventory changed: {dict(suite_counts)}, total={len(records)}"
        )
    splits = deterministic_split(records)
    groups, rows = [], []
    for record in sorted(records, key=lambda row: row["source_episode"]):
        split = splits[record["source_episode"]]
        for distance in DISTANCES:
            index = int(record["state_count"]) - distance
            group_id = len(groups)
            continuation_seed = 60_000_000 + group_id * 100_000
            group = {
                **record, "group_id": group_id, "split": split,
                "distance_replans": distance, "index": index,
                "continuation_policy_noise_seed_base": continuation_seed,
            }
            groups.append(group)
            base_seed = 42 + int(record["task"]) * 100_000 + int(record["trial"]) * 1_000 + index
            for offset in OFFSETS:
                rows.append({
                    "ordinal": len(rows), "group_id": group_id,
                    "source_episode": record["source_episode"], "suite": record["suite"],
                    "task": record["task"], "trial": record["trial"], "split": split,
                    "distance_replans": distance, "index": index,
                    "trajectory": record["trajectory"],
                    "first_policy_noise_seed": base_seed + offset,
                    "noise_offset": offset,
                    "continuation_policy_noise_seed_base": continuation_seed,
                })
    if len(groups) != 78 or len(rows) != 312:
        raise AssertionError(f"expected 78 groups/312 branches, got {len(groups)}/{len(rows)}")
    (output / "selection.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    train_sources = {record["source_episode"] for record in records if splits[record["source_episode"]] == "train"}
    val_sources = set(splits) - train_sources
    preregistration = {
        "format": FORMAT,
        "falsifiable_hypothesis": (
            "Fresh episode-disjoint horizon32 branches yield at least ten train and four "
            "validation mixed groups over Goal/Object/Spatial, sufficient to re-test a "
            "training-only-selected frozen action critic on untouched episodes."
        ),
        "parent": "C25 execution contract + D0-H32-s14000/replan8/no ensemble",
        "only_variable_within_group": "first-replan policy diffusion noise seed",
        "execution_contract": "first chunk32, all continuation replans8, fixed continuation seeds",
        "freshness_contract": "all C22 and C25 source episodes excluded before outcome inspection",
        "split_contract": "suite-stratified deterministic source-episode split frozen before rollouts",
        "source_episodes": len(records), "train_source_episodes": len(train_sources),
        "val_source_episodes": len(val_sources), "groups": len(groups), "branches": len(rows),
        "train_groups": sum(group["split"] == "train" for group in groups),
        "val_groups": sum(group["split"] == "val" for group in groups),
        "suite_source_counts": dict(sorted(suite_counts.items())),
        "excluded_source_episodes": sorted(used),
        "groups_detail": groups,
        "max_environment_steps": len(rows) * 400,
        "promotion": (
            "mechanical gates pass; >=10 train and >=4 val mixed groups; mixed groups span all 3 suites"
        ),
        "pass_permission": "GO_C27_FRESH_HELDOUT_CRITIC_CONFIRMATION",
        "fail_permission": "NO_GO_GENERAL_ACTION_CRITIC_DATA",
        "effect_conclusion": "NOT_EVIDENCE_READY",
    }
    atomic_json(output / "preregistration.json", preregistration)
    print(json.dumps({
        "output_root": str(output), "source_episodes": len(records),
        "suite_source_counts": dict(sorted(suite_counts.items())),
        "train_source_episodes": len(train_sources), "val_source_episodes": len(val_sources),
        "groups": len(groups), "branches": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
