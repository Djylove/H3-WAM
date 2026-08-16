#!/usr/bin/env python3
"""Freeze deterministic tri-arm C55 LIBERO jobs after offline selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-final", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--action-only-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-aux-checkpoint", type=Path, required=True)
    parser.add_argument("--trials", type=int, nargs="+", required=True)
    parser.add_argument("--stage", choices=("mechanical_canary", "fresh_final"), required=True)
    parser.add_argument("--total-workers", type=int, default=32)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    final_path = args.offline_final.resolve()
    final = json.loads(final_path.read_text())
    if (
        final.get("status") != "PASS_C55_OFFLINE_GATE"
        or final.get("permission") != "GO_FRESH_CLOSED_LOOP"
        or int(final.get("selected", {}).get("step", -1)) != 1000
    ):
        raise ValueError("C55 fresh rollout requires the frozen step1000 offline winner")
    trials = tuple(sorted(set(args.trials)))
    if args.total_workers < 1:
        raise ValueError("total workers must be positive")
    allowed = set(range(33, 37)) if args.stage == "mechanical_canary" else set(range(37, 50))
    if not trials or set(trials) != allowed:
        raise ValueError(f"{args.stage} requires exactly trials {sorted(allowed)}")
    checkpoints = {
        "d0_parent": args.parent_checkpoint.resolve(),
        "action_only": args.action_only_checkpoint.resolve(),
        "joint_aux": args.joint_aux_checkpoint.resolve(),
    }
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to reuse C55 rollout stage: {root}")
    root.mkdir(parents=True)
    jobs = []
    for trial in trials:
        for suite in SUITES:
            for task in range(10):
                for arm in ("d0_parent", "action_only", "joint_aux"):
                    job_id = len(jobs)
                    jobs.append(
                        {
                            "job_id": job_id,
                            "stage": args.stage,
                            "arm": arm,
                            "suite": suite,
                            "task": task,
                            "trial": trial,
                            "checkpoint": str(checkpoints[arm]),
                            "output": str(
                                root / "runs" / arm / suite
                                / f"task{task:02d}_trial{trial:02d}"
                            ),
                        }
                    )
    manifest = root / "jobs.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in jobs),
        encoding="utf-8",
    )
    identity = hashlib.sha256(manifest.read_bytes()).hexdigest()
    report = {
        "format": "h3wam-c55-fresh-jobs-v1",
        "stage": args.stage,
        "status": "PREPARED_NOT_EXECUTED",
        "jobs": len(jobs),
        "trials": list(trials),
        "suites": list(SUITES),
        "tasks_per_suite": 10,
        "arms": list(checkpoints),
        "total_workers": args.total_workers,
        "manifest": str(manifest),
        "manifest_sha256": identity,
        "offline_final_sha256": sha256_file(final_path),
        "checkpoints": {
            arm: {"path": str(path), "sha256": sha256_file(path)}
            for arm, path in checkpoints.items()
        },
        "effect_read_boundary": (
            "Mechanical canary may inspect completeness/state/seed contracts only; "
            "success aggregation is prohibited before fresh_final is complete."
        ),
    }
    (root / "PREPARED.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
