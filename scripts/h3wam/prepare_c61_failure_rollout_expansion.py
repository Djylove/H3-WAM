#!/usr/bin/env python3
"""Freeze a four-candidate failure-rollout expansion from C48 train successes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


FORMAT = "h3wam-c61-failure-rollout-expansion-v1"
DISTANCES = (3, 5)
OFFSETS = (0, 1_000_000, 2_000_000, 3_000_000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c48-dataset", type=Path, required=True)
    parser.add_argument("--c48-observations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite C61 root: {output_root}")
    dataset_path = args.c48_dataset.resolve()
    observations_path = args.c48_observations.resolve()
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if dataset.get("format") != "h3wam-c48-fact-dense-value-dataset-v1":
        raise ValueError("C61 requires the immutable C48 dataset")
    by_episode = {}
    for row in dataset["samples"]:
        by_episode.setdefault(int(row["episode_id"]), row)
    trajectories = {}
    for line in observations_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        trajectories.setdefault(int(row["episode_id"]), str(Path(row["trajectory"]).resolve()))

    jobs = []
    groups = []
    sources = []
    for episode_id, row in sorted(by_episode.items()):
        if row["split"] != "train" or not bool(row["success"]):
            continue
        trajectory = Path(trajectories[episode_id])
        with np.load(trajectory, allow_pickle=False) as archive:
            state_count = int(archive["step"].shape[0])
            if state_count < max(DISTANCES):
                continue
            steps = archive["step"].astype(np.int64).tolist()
        source = {
            "source_id": len(sources),
            "episode_id": episode_id,
            "suite": str(row["suite"]),
            "task": int(row["task"]),
            "trial": int(row["trial"]),
            "trajectory": str(trajectory),
            "trajectory_sha256": sha256_file(trajectory),
            "state_count": state_count,
        }
        sources.append(source)
        for distance in DISTANCES:
            index = state_count - distance
            group_id = len(groups)
            continuation_seed = 361_000_000 + group_id * 10_000
            base_seed = 61_000_000 + group_id * 100
            group = {
                **source,
                "group_id": group_id,
                "distance_replans": distance,
                "index": index,
                "start_step": int(steps[index]),
                "continuation_policy_noise_seed_base": continuation_seed,
            }
            groups.append(group)
            for candidate, offset in enumerate(OFFSETS):
                jobs.append(
                    {
                        **group,
                        "ordinal": len(jobs),
                        "candidate": candidate,
                        "first_policy_noise_seed": base_seed + offset,
                    }
                )
    if len(sources) < 100 or len(groups) != 2 * len(sources) or len(jobs) != 4 * len(groups):
        raise ValueError("C61 source inventory is unexpectedly small or inconsistent")
    output_root.mkdir(parents=True)
    jobs_path = output_root / "jobs.jsonl"
    jobs_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in jobs),
        encoding="utf-8",
    )
    frozen = {
        "format": FORMAT,
        "status": "PASS_C61_FROZEN_NOT_EXECUTED",
        "source_contract": "C48 train-split successful parent rollouts only",
        "intervention_contract": (
            "exact d3/d5 sim-state restore; four first-chunk diffusion seeds; "
            "identical continuation seed schedule within each group"
        ),
        "failure_contract": (
            "retain only terminal failures after collection; action imitation masked; "
            "branch start is the explicit causal intervention boundary"
        ),
        "sources": len(sources),
        "groups": len(groups),
        "jobs": len(jobs),
        "offsets": list(OFFSETS),
        "c48_dataset_sha256": sha256_file(dataset_path),
        "c48_observations_sha256": sha256_file(observations_path),
        "jobs_sha256": sha256_file(jobs_path),
    }
    (output_root / "FROZEN.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (output_root / "runs").mkdir()
    (output_root / "logs").mkdir()
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
