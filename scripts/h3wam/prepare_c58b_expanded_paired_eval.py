#!/usr/bin/env python3
"""Pre-register the eight C58b candidate-only expanded LIBERO jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


C58_SHA256 = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
D0_READY_FORMAT = "h3wam-c58b-expanded-d0-control-freeze-v1"
TRIAL33_SHA256 = "f7e9c8f65c177d33a3b168d0e0a47e79034d0054c99866a66ba09f82ee916ab3"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TRIAL_GROUPS = (tuple(range(34, 42)), tuple(range(42, 50)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    checkpoint: Path,
    balanced_gate: Path,
    d0_ready: Path,
    trial33_results: Path,
    output_root: Path,
) -> dict:
    checkpoint = checkpoint.resolve()
    balanced_gate = balanced_gate.resolve()
    d0_ready = d0_ready.resolve()
    trial33_results = trial33_results.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if sha256_file(checkpoint) != C58_SHA256:
        raise ValueError("C58b checkpoint SHA256 mismatch")
    if sha256_file(trial33_results) != TRIAL33_SHA256:
        raise ValueError("corrected trial33 result SHA256 mismatch")
    gate = json.loads(balanced_gate.read_text(encoding="utf-8"))
    if (
        gate.get("permission") != "GO_FRESH_LIBERO"
        or Path(gate.get("checkpoint", "")).resolve() != checkpoint
        or gate.get("checkpoint_sha256") != C58_SHA256
        or gate.get("closed_loop_protocol", {}).get("wait_steps") != 30
    ):
        raise ValueError("C58b balanced gate mismatch")
    control = json.loads(d0_ready.read_text(encoding="utf-8"))
    if (
        control.get("format") != D0_READY_FORMAT
        or control.get("status") != "PASS_D0_CONTROL_REUSE"
        or control.get("permission") != "GO_C58B_CANDIDATE_ONLY_TRIALS34_49"
        or control.get("controls") != 640
        or control.get("trials") != list(range(34, 50))
    ):
        raise ValueError("D0 historical control did not pass reuse gate")
    trial33 = json.loads(trial33_results.read_text(encoding="utf-8"))
    if (
        trial33.get("status") != "COMPLETE"
        or trial33.get("paired_episodes_per_arm") != 40
        or trial33.get("candidate_successes") != 18
        or trial33.get("control_successes") != 16
    ):
        raise ValueError("trial33 exact-reproduction bridge mismatch")
    jobs = []
    gpu = 0
    for suite in SUITES:
        for trials in TRIAL_GROUPS:
            jobs.append({
                "job_id": len(jobs),
                "gpu": gpu,
                "suite": suite,
                "trials": list(trials),
                "tasks": list(range(10)),
                "episodes": 80,
                "output": str(
                    output_root / "candidate_c58b" / suite
                    / f"trials{trials[0]:02d}-{trials[-1]:02d}"
                ),
            })
            gpu += 1
    output_root.mkdir(parents=True)
    manifest = output_root / "jobs.jsonl"
    manifest.write_text(
        "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    report = {
        "format": "h3wam-c58b-expanded-paired-prepared-v1",
        "status": "PREPARED_NOT_EXECUTED",
        "permission": "GO_8GPU_CANDIDATE_ONLY_NO_INTERMEDIATE_STOP",
        "hypothesis": (
            "C58b full layer-wise H3 carriers improve paired LIBERO success over "
            "the exact D0 parent across trials33..49."
        ),
        "candidate_checkpoint": str(checkpoint),
        "candidate_checkpoint_sha256": C58_SHA256,
        "balanced_gate": str(balanced_gate),
        "balanced_gate_sha256": sha256_file(balanced_gate),
        "d0_control_ready": str(d0_ready),
        "d0_control_ready_sha256": sha256_file(d0_ready),
        "trial33_results": str(trial33_results),
        "trial33_results_sha256": TRIAL33_SHA256,
        "jobs": len(jobs),
        "candidate_episodes": 640,
        "trials": list(range(34, 50)),
        "suites": list(SUITES),
        "tasks_per_suite": 10,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "protocol": {
            "wait_steps": 30,
            "max_steps": 400,
            "replan_steps": 8,
            "action_horizon": 32,
            "model_evaluations": 10,
            "seed": 42,
            "environment_seed": None,
            "policy_noise_seed_base": None,
            "normalized_action_pre_clamp": True,
            "save_trajectories": True,
        },
        "evaluation_gate": {
            "absolute_gain_at_least_0_03": 0.03,
            "net_wins_at_least": 20,
            "one_sided_exact_mcnemar_p_at_most": 0.05,
            "no_suite_regression_below": -0.03,
        },
        "stopping": "Run all 640 candidate episodes; intermediate success is never read for stopping.",
        "fallback": "If D0 reuse audit fails, discard this preparation and rerun both arms.",
    }
    temporary = output_root / f".PREPARED.json.{os.getpid()}.partial"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_root / "PREPARED.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--balanced-gate", type=Path, required=True)
    parser.add_argument("--d0-ready", type=Path, required=True)
    parser.add_argument("--trial33-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(
        args.checkpoint, args.balanced_gate, args.d0_ready,
        args.trial33_results, args.output_root,
    )
    print(json.dumps({
        "status": report["status"], "permission": report["permission"],
        "jobs": report["jobs"], "candidate_episodes": report["candidate_episodes"],
        "manifest_sha256": report["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
