#!/usr/bin/env python3
"""Freeze C32 successes into a wholly fresh ranking-validation branch set."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


FORMAT = "h3wam-c33-fresh-ranking-causal-dataset-v1"
DISTANCES = (3, 5)
OFFSETS = (0, 1_000_000, 2_000_000, 3_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c32-completed", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    c32_path = args.c32_completed.resolve()
    c32 = json.loads(c32_path.read_text(encoding="utf-8"))
    if c32.get("status") != "PASS_C32_FRESH_PARENT_SOURCE_EXPANSION":
        raise ValueError("C32 source expansion did not pass")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    (output / "runs").mkdir(parents=True)
    (output / "logs").mkdir()

    records, ineligible = [], []
    for row in c32["rows"]:
        if not bool(row["success"]):
            continue
        result_path = Path(row["path"]).resolve()
        trajectory = result_path.parent / (
            f"task{int(row['task']):02d}_trial{int(row['trial']):02d}_trajectory.npz"
        )
        with np.load(trajectory, allow_pickle=False) as archive:
            state_count = int(archive["step"].shape[0])
        available = [distance for distance in DISTANCES if state_count > distance]
        source_episode = f"{row['suite']}:task{row['task']}:trial{row['trial']}"
        if not available:
            ineligible.append(source_episode)
            continue
        records.append({
            "source_episode": source_episode, "suite": row["suite"],
            "task": int(row["task"]), "trial": int(row["trial"]),
            "trajectory": str(trajectory), "source_result": str(result_path),
            "state_count": state_count, "available_distances": available,
        })
    if len(records) < 32:
        raise ValueError(f"fewer than 32 eligible C33 sources: {len(records)}")
    suite_counts: dict[str, int] = defaultdict(int)
    groups, selections = [], []
    for record in sorted(records, key=lambda row: row["source_episode"]):
        suite_counts[record["suite"]] += 1
        for distance in record["available_distances"]:
            group_id = len(groups)
            index = int(record["state_count"]) - int(distance)
            continuation = 120_000_000 + group_id * 100_000
            groups.append({
                **record, "group_id": group_id, "split": "fresh_ranking_val",
                "distance_replans": distance, "index": index,
                "continuation_policy_noise_seed_base": continuation,
            })
            base_seed = (
                42 + int(record["task"]) * 100_000
                + int(record["trial"]) * 1_000 + index
            )
            for offset in OFFSETS:
                selections.append({
                    "ordinal": len(selections), "group_id": group_id,
                    "source_episode": record["source_episode"],
                    "suite": record["suite"], "task": record["task"],
                    "trial": record["trial"], "split": "fresh_ranking_val",
                    "distance_replans": distance, "index": index,
                    "trajectory": record["trajectory"],
                    "first_policy_noise_seed": base_seed + offset,
                    "noise_offset": offset,
                    "continuation_policy_noise_seed_base": continuation,
                })
    if len(groups) < 50 or len(selections) != 4 * len(groups):
        raise ValueError(f"insufficient C33 groups/branches: {len(groups)}/{len(selections)}")
    (output / "selection.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in selections), encoding="utf-8"
    )
    prereg = {
        "format": FORMAT,
        "hypothesis": "Never-consumed trial8..11 sources yield at least eight fresh mixed ranking groups across at least three LIBERO suites.",
        "parent": "D0-H32-s14000/replan8/no-ensemble C32 successes",
        "only_variable_within_group": "first-replan policy diffusion noise seed",
        "execution_contract": "first chunk32; continuation replan8 with identical seed schedule",
        "consequence_contract": "row1 at start+32 or terminal within first chunk; terminal fields required",
        "split_contract": "all C33 sources are fresh ranking validation; none may train or select the consequence model",
        "source_episodes": len(records), "groups": len(groups),
        "branches": len(selections), "suite_source_counts": dict(sorted(suite_counts.items())),
        "ineligible_short_sources": sorted(ineligible), "groups_detail": groups,
        "promotion": "all mechanical/consequence checks; fresh mixed groups>=8 across>=3 suites",
        "pass_permission": "GO_FRESH_CONSEQUENCE_VALUE_RANKING",
        "fail_permission": "NO_GO_FRESH_CONSEQUENCE_VALUE_RANKING",
        "effect_conclusion": "NOT_EVIDENCE_READY",
        "c32_completed": str(c32_path), "max_environment_steps": len(selections) * 400,
    }
    atomic_json(output / "preregistration.json", prereg)
    print(json.dumps({
        "output_root": str(output), "sources": len(records),
        "suite_source_counts": dict(sorted(suite_counts.items())),
        "groups": len(groups), "branches": len(selections),
    }, indent=2))


if __name__ == "__main__":
    main()
