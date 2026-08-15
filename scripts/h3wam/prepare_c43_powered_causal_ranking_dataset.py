#!/usr/bin/env python3
"""Freeze C42 successes into source-disjoint train and final causal branches."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


FORMAT = "h3wam-c43-powered-causal-ranking-dataset-v1"
DISTANCES = (3, 5)
OFFSETS = (0, 1_000_000, 2_000_000, 3_000_000)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c42-completed", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    c42_path = args.c42_completed.resolve()
    c42 = json.loads(c42_path.read_text())
    if c42.get("status") != "PASS_C42_FRESH_PARENT_SOURCE_EXPANSION":
        raise ValueError("C42 source expansion did not pass")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    (output / "runs").mkdir(parents=True)
    (output / "logs").mkdir()

    records, ineligible = [], []
    for row in c42["rows"]:
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
            "source_episode": source_episode,
            "suite": row["suite"],
            "task": int(row["task"]),
            "trial": int(row["trial"]),
            "trajectory": str(trajectory),
            "source_result": str(result_path),
            "state_count": state_count,
            "available_distances": available,
            "split": "train" if row["role"] == "ranker_train_source" else "fresh_final",
        })

    source_counts = Counter(record["split"] for record in records)
    if source_counts["train"] < 36 or source_counts["fresh_final"] < 36:
        raise ValueError(f"insufficient eligible sources by split: {dict(source_counts)}")
    groups, selections = [], []
    suite_source_counts: dict[str, Counter[str]] = {
        "train": Counter(), "fresh_final": Counter()
    }
    for record in sorted(records, key=lambda row: row["source_episode"]):
        suite_source_counts[record["split"]][record["suite"]] += 1
        for distance in record["available_distances"]:
            group_id = len(groups)
            index = int(record["state_count"]) - int(distance)
            continuation = 160_000_000 + group_id * 100_000
            groups.append({
                **record,
                "group_id": group_id,
                "distance_replans": distance,
                "index": index,
                "continuation_policy_noise_seed_base": continuation,
            })
            base_seed = (
                42 + int(record["task"]) * 100_000
                + int(record["trial"]) * 1_000 + index
            )
            for offset in OFFSETS:
                selections.append({
                    "ordinal": len(selections),
                    "group_id": group_id,
                    "source_episode": record["source_episode"],
                    "suite": record["suite"],
                    "task": record["task"],
                    "trial": record["trial"],
                    "split": record["split"],
                    "distance_replans": distance,
                    "index": index,
                    "trajectory": record["trajectory"],
                    "first_policy_noise_seed": base_seed + offset,
                    "noise_offset": offset,
                    "continuation_policy_noise_seed_base": continuation,
                })
    if len(selections) != 4 * len(groups):
        raise ValueError("C43 group/branch inventory mismatch")
    (output / "selection.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in selections)
    )
    prereg = {
        "format": FORMAT,
        "hypothesis": (
            "Five train and five untouched final source trials yield at least24 mixed "
            "counterfactual groups across at least3 suites in each split."
        ),
        "parent": "D0-H32-s14000/replan8/no-ensemble C42 successes",
        "only_variable_within_group": "first-replan policy diffusion noise seed",
        "execution_contract": "first chunk32; continuation replan8 with identical seed schedule",
        "consequence_contract": "row1 at start+32 or terminal within first chunk; terminal fields required",
        "split_contract": (
            "train trials12,13,14,16,17; fresh_final trials15,18,19,20,21; "
            "fresh_final cannot fit or select any model"
        ),
        "source_episodes": len(records),
        "source_counts": dict(sorted(source_counts.items())),
        "suite_source_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in suite_source_counts.items()
        },
        "groups": len(groups),
        "branches": len(selections),
        "ineligible_short_sources": sorted(ineligible),
        "groups_detail": groups,
        "promotion": (
            "all mechanical/consequence checks; train mixed groups>=24 across>=3 suites; "
            "fresh_final mixed groups>=24 across>=3 suites"
        ),
        "pass_permission": "GO_C44_POWERED_CONSEQUENCE_VALUE_RANKING",
        "fail_permission": "NO_GO_C44_POWERED_CONSEQUENCE_VALUE_RANKING",
        "effect_conclusion": "NOT_EVIDENCE_READY",
        "c42_completed": str(c42_path),
        "max_environment_steps": len(selections) * 400,
    }
    atomic_json(output / "preregistration.json", prereg)
    print(json.dumps({
        "output_root": str(output),
        "sources": len(records),
        "source_counts": dict(sorted(source_counts.items())),
        "groups": len(groups),
        "branches": len(selections),
    }, indent=2))


if __name__ == "__main__":
    main()
