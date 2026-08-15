#!/usr/bin/env python3
"""Freeze a multisuite temporal-distance sweep for counterfactual outcomes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


SOURCES = (
    ("libero_goal", 2, 2),
    ("libero_goal", 7, 0),
    ("libero_object", 0, 1),
    ("libero_object", 4, 2),
    ("libero_spatial", 0, 3),
    ("libero_spatial", 3, 0),
    ("libero_10", 3, 1),
    ("libero_10", 5, 0),
)
DISTANCES = (1, 3, 5)
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
    source_records = []
    for suite, task, trial in SOURCES:
        slug = suite.removeprefix("libero_")
        run_root = source_root / (
            f"d0_h32_s14000_{slug}_task{task}_trial{trial}_replan8"
        )
        results = json.loads((run_root / "results.json").read_text())
        episode = results["tasks"][0]["episodes"][0]
        if not episode["success"]:
            raise ValueError(f"C22 source is not successful: {run_root}")
        trajectory = run_root / f"task{task:02d}_trial{trial:02d}_trajectory.npz"
        with np.load(trajectory, allow_pickle=False) as archive:
            state_count = int(archive["step"].shape[0])
        source_record = {
            "suite": suite,
            "task": task,
            "trial": trial,
            "trajectory": str(trajectory.resolve()),
            "state_count": state_count,
            "source_success": True,
        }
        source_records.append(source_record)
        for distance in DISTANCES:
            index = state_count - distance
            if index < 0:
                raise ValueError(f"source too short for distance {distance}: {trajectory}")
            original_seed = 42 + task * 100_000 + trial * 1_000 + index
            for offset in OFFSETS:
                rows.append(
                    {
                        "ordinal": len(rows),
                        **source_record,
                        "distance_replans": distance,
                        "index": index,
                        "policy_noise_seed": original_seed + offset,
                        "noise_offset": offset,
                    }
                )
    if len(rows) != 96:
        raise AssertionError(f"expected 96 branches, got {len(rows)}")
    selection = output / "selection.jsonl"
    selection.write_text("".join(json.dumps(row) + "\n" for row in rows))
    preregistration = {
        "format": "h3wam-c22-counterfactual-entropy-sweep-v1",
        "falsifiable_hypothesis": (
            "Across 24 fixed canonical states, diffusion noise produces distinct first "
            "action chunks in every group and at least four mixed-outcome groups spanning "
            "at least two LIBERO suites."
        ),
        "parent": "D0-H32-s14000/replan8/no ensemble",
        "within_group_only_variable": "policy diffusion noise seed offset",
        "calibration_strata": "suite/source episode/distance of 1, 3, or 5 replans",
        "source_episode_split_contract": (
            "All later train/validation rows derived from one source episode remain in the "
            "same split; this calibration sweep is not itself a training split."
        ),
        "environment_seed": 42,
        "source_episodes": source_records,
        "distances_replans": list(DISTANCES),
        "noise_offsets": list(OFFSETS),
        "groups": 24,
        "branches": 96,
        "max_environment_steps": 38_400,
        "promotion": (
            "all 24 groups action-diverse, >=4 mixed-outcome groups, and mixed groups in "
            ">=2 suites"
        ),
        "pass_permission": "GO_TARGETED_COUNTERFACTUAL_DATASET",
        "fail_permission": "NO_GO_UNIFORM_DATASET_EXPANSION",
        "effect_conclusion": "NOT_EVIDENCE_READY",
    }
    atomic_json(output / "preregistration.json", preregistration)
    print(json.dumps({"output_root": str(output), "branches": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
