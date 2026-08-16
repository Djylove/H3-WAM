#!/usr/bin/env python3
"""Freeze the C60 FACT trials34..49 isolated-process evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


C60_SHA256 = "d6659c6b387f062a99f670a1d902b56df71a6bf1472aa4e46e56c9213ba75a36"
C60_READY_SHA256 = "0d7ce236c8e1d2fcbce78f064ce89536592a6b04a5e2ef2ab2100a33e0b9b081"
PAIRED_GATE_SHA256 = "c68b2f8bfff6308f97a5f181facd9841c84cfcf9ec22e5e2e561196084337220"
TRIAL33_SHA256 = "fe4c7c49c6fd7e7ce0abf56c1f863c604fc05862dfc98fb5f0b4f8d00417ebe2"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TRIALS = tuple(range(34, 50))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    checkpoint: Path, ready_path: Path, paired_gate_path: Path,
    trial33_path: Path, output_root: Path,
) -> dict:
    paths = [checkpoint, ready_path, paired_gate_path, trial33_path]
    checkpoint, ready_path, paired_gate_path, trial33_path = (
        path.resolve() for path in paths
    )
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    expected = (
        (checkpoint, C60_SHA256, "C60 checkpoint"),
        (ready_path, C60_READY_SHA256, "C60 READY"),
        (paired_gate_path, PAIRED_GATE_SHA256, "paired gate"),
        (trial33_path, TRIAL33_SHA256, "trial33 result"),
    )
    for path, digest, label in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"{label} SHA256 mismatch")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if (
        ready.get("status") != "PASS_C56B_ONLINE_LONG10000_STRICT_RESTORE"
        or ready.get("permission") != "READY_FOR_PAIRED_HELDOUT"
        or ready.get("arm") != "C60_MAIN"
        or ready.get("checkpoint_sha256") != C60_SHA256
        or Path(ready.get("checkpoint", "")).resolve() != checkpoint
    ):
        raise ValueError("C60 READY contract mismatch")
    gate = json.loads(paired_gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("status") != "PASS_PAIRED_BALANCED80"
        or gate.get("permission") != "GO_PAIRED_LIBERO"
        or gate.get("checkpoint_identity", {}).get("c60_main_checkpoint_sha256")
        != C60_SHA256
    ):
        raise ValueError("paired heldout gate mismatch")
    trial33 = json.loads(trial33_path.read_text(encoding="utf-8"))
    effects = trial33.get("paired_effects", {}).get("main_vs_c58", {})
    suites = trial33.get("suites", [])
    safety = all(
        row.get("main_successes", -99) - row.get("c58_parent_successes", 99) > -3
        for row in suites
    )
    if (
        trial33.get("format") != "h3wam-c56b-fact-paired-libero-trial33-v1"
        or trial33.get("status") != "COMPLETE"
        or trial33.get("paired_episodes_per_arm") != 40
        or trial33.get("main_successes", 0) <= trial33.get("c58_parent_successes", 0)
        or effects.get("first_wins", 0) <= effects.get("second_wins", 0)
        or not safety
    ):
        raise ValueError("trial33 did not pass preregistered C60 expansion screen")

    jobs = []
    for trial in TRIALS:
        for suite in SUITES:
            for task in range(10):
                job_id = len(jobs)
                jobs.append({
                    "job_id": job_id,
                    "gpu": job_id % 8,
                    "suite": suite,
                    "tasks": [task],
                    "trials": [trial],
                    "episodes": 1,
                    "output": str(
                        output_root / "candidate_c60" / suite
                        / f"task{task:02d}_trial{trial:02d}"
                    ),
                })
    if len(jobs) != 640 or len({
        (row["suite"], row["tasks"][0], row["trials"][0]) for row in jobs
    }) != 640:
        raise AssertionError("C60 expanded grid is not exact")
    output_root.mkdir(parents=True)
    manifest = output_root / "jobs.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in jobs),
        encoding="utf-8",
    )
    report = {
        "format": "h3wam-c56b-fact-expanded-prepared-v1",
        "status": "PREPARED_NOT_EXECUTED",
        "permission": "GO_MECHANICAL_CANARY_THEN_8GPU_640_FRESH_PROCESSES",
        "candidate": "C60_FULL_FACT_PORT",
        "candidate_checkpoint": str(checkpoint),
        "candidate_checkpoint_sha256": C60_SHA256,
        "candidate_ready": str(ready_path),
        "candidate_ready_sha256": C60_READY_SHA256,
        "paired_gate": str(paired_gate_path),
        "paired_gate_sha256": PAIRED_GATE_SHA256,
        "trial33_results": str(trial33_path),
        "trial33_results_sha256": TRIAL33_SHA256,
        "jobs": 640,
        "candidate_episodes": 640,
        "one_episode_per_process": True,
        "trials": list(TRIALS),
        "suites": list(SUITES),
        "tasks_per_suite": 10,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "protocol": {
            "wait_steps": 30, "max_steps": 400, "replan_steps": 8,
            "action_horizon": 32, "model_evaluations": 10, "seed": 42,
            "environment_seed": None, "policy_noise_seed_base": None,
            "episode_seed_contract": "42+task*100000+trial*1000",
            "normalized_action_pre_clamp": True,
            "save_trajectories": True,
        },
        "evaluation_gate": {
            "absolute_gain_at_least_0_03": 0.03,
            "net_wins_at_least": 20,
            "one_sided_exact_mcnemar_p_at_most": 0.05,
            "no_suite_regression_below": -0.03,
        },
        "stopping": (
            "Run every one of the 640 C60 episodes. Never inspect intermediate "
            "successes for stopping or checkpoint selection."
        ),
        "process_contract": "One fresh simulator and policy process per episode.",
    }
    temporary = output_root / f".PREPARED.json.{os.getpid()}.partial"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_root / "PREPARED.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--paired-gate", type=Path, required=True)
    parser.add_argument("--trial33-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(
        args.checkpoint, args.ready, args.paired_gate,
        args.trial33_results, args.output_root,
    )
    print(json.dumps({key: report[key] for key in (
        "status", "permission", "jobs", "manifest_sha256"
    )}, indent=2))


if __name__ == "__main__":
    main()
