#!/usr/bin/env python3
"""Prepare a direct, strictly paired C69 versus C58b LIBERO evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TRIALS = tuple(range(33, 50))
C69_SHA = "20914729d340b05768ec99e152cc026313d5a0dab064c963df90ac8184d8a12a"
C58_SHA = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--c69-checkpoint", type=Path, required=True)
    parser.add_argument("--c58-checkpoint", type=Path, required=True)
    parser.add_argument("--c69-authorization", type=Path, required=True)
    parser.add_argument("--c58-ready", type=Path, required=True)
    args = parser.parse_args()

    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(root)
    snapshot = args.snapshot.resolve()
    freeze = snapshot / "SOURCE_FREEZE.json"
    if not freeze.is_file() or os.stat(snapshot).st_mode & 0o222:
        raise ValueError("rollout source snapshot is missing or writable")

    endpoints = {
        "c69_action_only": (args.c69_checkpoint.resolve(), C69_SHA),
        "c58b_fastwam": (args.c58_checkpoint.resolve(), C58_SHA),
    }
    for name, (path, expected) in endpoints.items():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"checkpoint identity mismatch: {name}")

    c69_gate = load(args.c69_authorization.resolve())
    c58_gate = load(args.c58_ready.resolve())
    if (
        c69_gate.get("format") != "h3wam-c67-c69-paired-rollout-authorization-v1"
        or c69_gate.get("status") != "AUTHORIZED_C67_C69_FIXED_S20_PAIRED_680"
        or c69_gate.get("endpoints", {}).get("c69_action_only", {}).get("checkpoint_sha256") != C69_SHA
    ):
        raise ValueError("C69 inner authorization mismatch")
    if (
        c58_gate.get("format") != "h3wam-c58b-online-balanced80-ready-v1"
        or c58_gate.get("status") != "PASS"
        or c58_gate.get("permission") != "GO_FRESH_LIBERO"
        or c58_gate.get("checkpoint_sha256") != C58_SHA
    ):
        raise ValueError("C58b readiness gate mismatch")

    jobs = []
    for trial in TRIALS:
        for suite in SUITES:
            for task in range(10):
                pair_id = len(jobs) // 2
                for arm, (checkpoint, digest) in endpoints.items():
                    jobs.append({
                        "job_id": len(jobs), "pair_id": pair_id, "arm": arm,
                        "suite": suite, "tasks": [task], "trials": [trial],
                        "episodes": 1, "checkpoint": str(checkpoint),
                        "checkpoint_sha256": digest,
                        "output": str(root / "episodes" / arm / suite / f"task{task:02d}_trial{trial:02d}"),
                    })
    if len(jobs) != 1360 or len({row["pair_id"] for row in jobs}) != 680:
        raise AssertionError("paired grid is not exact")

    root.mkdir(parents=True)
    manifest = root / "jobs.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in jobs), encoding="utf-8")
    authorization = {
        "format": "h3wam-c69-c58b-direct-paired680-authorization-v1",
        "status": "AUTHORIZED_DIRECT_PAIRED_RECHECK",
        "permission": "GO_1360_FRESH_PROCESSES_NO_INTERMEDIATE_PROMOTION",
        "hypothesis": "C69-s20000 directly outperforms C58b-s10000 under an identical 680-episode LIBERO paired grid.",
        "classification": "evaluation_only_controlled_promotion_confirmation",
        "optimizer_steps": 0,
        "source_snapshot": str(snapshot),
        "source_freeze_sha256": sha256(freeze),
        "manifest_sha256": sha256(manifest),
        "jobs": 1360,
        "pairs": 680,
        "trials": list(TRIALS),
        "suites": list(SUITES),
        "inner_gates": {
            "c69": {"path": str(args.c69_authorization.resolve()), "sha256": sha256(args.c69_authorization.resolve())},
            "c58b": {"path": str(args.c58_ready.resolve()), "sha256": sha256(args.c58_ready.resolve())},
        },
        "endpoints": {name: {"checkpoint": str(path), "checkpoint_sha256": digest} for name, (path, digest) in endpoints.items()},
        "claim_boundary": "Confirmation rerun on previously used LIBERO trial indices 33..49; direct paired evidence, not unseen-state generalization.",
    }
    (root / "AUTHORIZATION.json").write_text(json.dumps(authorization, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(authorization, indent=2))


if __name__ == "__main__":
    main()
