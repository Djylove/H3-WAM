#!/usr/bin/env python3
"""Freeze paired fresh-process C58/D0 jobs for descriptive trials 0..32."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TRIALS = tuple(range(33))
C58_SHA256 = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
D0_SHA256 = "36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"
CONFIRMATORY_FINAL_SHA256 = "53a06ac5c3c36298ed2ee397688eb03e6918219d32f469897e9139530d954f88"
CONFIRMATORY_EVIDENCE_SHA256 = "e44a32833c1d9f71485f3cca37785b5d813f59c7af4eea12311dfd1ed14f1e3c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    c58_checkpoint: Path,
    d0_checkpoint: Path,
    balanced_gate: Path,
    confirmatory_final: Path,
    snapshot_manifest: Path,
    output_root: Path,
) -> dict:
    paths = [
        c58_checkpoint, d0_checkpoint, balanced_gate, confirmatory_final,
        snapshot_manifest,
    ]
    c58_checkpoint, d0_checkpoint, balanced_gate, confirmatory_final, snapshot_manifest = (
        path.resolve() for path in paths
    )
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if sha256_file(c58_checkpoint) != C58_SHA256:
        raise ValueError("C58 checkpoint SHA256 mismatch")
    if sha256_file(d0_checkpoint) != D0_SHA256:
        raise ValueError("D0 checkpoint SHA256 mismatch")
    if sha256_file(confirmatory_final) != CONFIRMATORY_FINAL_SHA256:
        raise ValueError("confirmatory FINAL SHA256 mismatch")
    snapshot = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
    snapshot_root = snapshot_manifest.parent
    required_snapshot_hashes = {
        "preparer": "scripts/h3wam/prepare_c58b_full50_descriptive_eval.py",
        "launcher": "scripts/h3wam/launch_c58b_full50_descriptive_arm.sh",
        "aggregator": "scripts/h3wam/aggregate_c58b_full50_descriptive_eval.py",
        "finalizer": "scripts/h3wam/watch_c58b_full50_descriptive_finalizer.sh",
        "starwam_wan_block": "third_party/StarWAM/starwam/modules/wan_block.py",
    }
    if (
        snapshot.get("format") != "h3wam-c58b-full50-runtime-snapshot-v1"
        or snapshot.get("status") != "VERIFIED_FOR_READ_ONLY_FREEZE"
        or not isinstance(snapshot.get("source_commit"), str)
        or not snapshot["source_commit"]
    ):
        raise ValueError("runtime snapshot manifest contract mismatch")
    for name, relative in required_snapshot_hashes.items():
        path = snapshot_root / relative
        if (
            not path.is_file()
            or sha256_file(path) != snapshot.get("hashes", {}).get(name)
            or path.stat().st_mode & 0o222
        ):
            raise ValueError(f"runtime snapshot source mismatch: {name}")
    final = json.loads(confirmatory_final.read_text(encoding="utf-8"))
    if (
        final.get("status") != "PASS_C58B_EXPANDED_PAIRED"
        or final.get("effect_status") != "EVIDENCE_READY"
        or final.get("pairs") != 680
        or final.get("pair_evidence_sha256") != CONFIRMATORY_EVIDENCE_SHA256
        or not all(final.get("gates", {}).values())
    ):
        raise ValueError("confirmatory C58 carrier promotion contract mismatch")
    gate = json.loads(balanced_gate.read_text(encoding="utf-8"))
    if (
        gate.get("permission") != "GO_FRESH_LIBERO"
        or Path(gate.get("checkpoint", "")).resolve() != c58_checkpoint
        or gate.get("checkpoint_sha256") != C58_SHA256
    ):
        raise ValueError("C58 balanced gate mismatch")

    jobs = []
    arm_counts = {"candidate_c58b": 0, "control_d0": 0}
    arm_specs = (
        ("candidate_c58b", "n0", "h3_fastwam_online_int8", c58_checkpoint),
        ("control_d0", "n3", "h3_dreamwam_kv_int8", d0_checkpoint),
    )
    for trial in TRIALS:
        for suite in SUITES:
            for task in range(10):
                for arm, node, policy, checkpoint in arm_specs:
                    ordinal = arm_counts[arm]
                    arm_counts[arm] += 1
                    jobs.append({
                        "job_id": len(jobs),
                        "arm_ordinal": ordinal,
                        "arm": arm,
                        "node": node,
                        "gpu": ordinal % 8,
                        "policy": policy,
                        "checkpoint": str(checkpoint),
                        "suite": suite,
                        "task": task,
                        "trial": trial,
                        "episodes": 1,
                        "output": str(
                            output_root / "supplement_trials00_32" / arm / suite
                            / f"task{task:02d}_trial{trial:02d}"
                        ),
                    })
    identities = {
        (job["arm"], job["suite"], job["task"], job["trial"])
        for job in jobs
    }
    if len(jobs) != 2640 or len(identities) != 2640:
        raise AssertionError("full50 job grid identity failure")
    if arm_counts != {"candidate_c58b": 1320, "control_d0": 1320}:
        raise AssertionError("full50 per-arm job count failure")

    output_root.mkdir(parents=True)
    manifest = output_root / "jobs.jsonl"
    manifest.write_text(
        "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    report = {
        "format": "h3wam-c58b-full50-descriptive-prepared-v1",
        "status": "PREPARED_NOT_EXECUTED",
        "permission": "GO_DESCRIPTIVE_FULL50_SUPPLEMENT_NO_PROMOTION",
        "interpretation": (
            "Trials0..32 were consumed during historical development. Their paired "
            "rerun completes a descriptive 50-state benchmark and cannot create, "
            "revoke or strengthen the confirmatory C58 promotion from trials33..49."
        ),
        "confirmatory_trials": list(range(33, 50)),
        "descriptive_supplement_trials": list(TRIALS),
        "confirmatory_final": str(confirmatory_final),
        "confirmatory_final_sha256": CONFIRMATORY_FINAL_SHA256,
        "confirmatory_pair_evidence_sha256": CONFIRMATORY_EVIDENCE_SHA256,
        "runtime_snapshot_manifest": str(snapshot_manifest),
        "runtime_snapshot_manifest_sha256": sha256_file(snapshot_manifest),
        "runtime_snapshot_source_commit": snapshot["source_commit"],
        "checkpoints": {
            "candidate_c58b": {"path": str(c58_checkpoint), "sha256": C58_SHA256},
            "control_d0": {"path": str(d0_checkpoint), "sha256": D0_SHA256},
        },
        "jobs": 2640,
        "episodes_per_arm": 1320,
        "jobs_per_gpu": {"candidate_c58b_n0": 165, "control_d0_n3": 165},
        "nodes": {"candidate_c58b": "n0:32611", "control_d0": "n3:30234"},
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "process_contract": "one fresh simulator and policy process per episode",
        "protocol": {
            "wait_steps": 30, "max_steps": 400, "replan_steps": 8,
            "action_horizon": 32, "model_evaluations": 10, "seed": 42,
            "environment_seed": None, "policy_noise_seed_base": None,
            "normalized_action_pre_clamp": True, "save_trajectories": True,
        },
        "stopping": "Run all 2640 episodes; never read intermediate success for stopping or selection.",
        "final_gate": (
            "Both 1320-episode arm markers, exact 1320 initial-state pairs, all source "
            "hashes and exact 2000 pair identities. Output is DESCRIPTIVE_ONLY."
        ),
    }
    tmp = output_root / f".PREPARED.json.{os.getpid()}.partial"
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, output_root / "PREPARED.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c58-checkpoint", type=Path, required=True)
    parser.add_argument("--d0-checkpoint", type=Path, required=True)
    parser.add_argument("--balanced-gate", type=Path, required=True)
    parser.add_argument("--confirmatory-final", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(
        args.c58_checkpoint, args.d0_checkpoint, args.balanced_gate,
        args.confirmatory_final, args.snapshot_manifest, args.output_root,
    )
    print(json.dumps({
        "permission": report["permission"], "jobs": report["jobs"],
        "episodes_per_arm": report["episodes_per_arm"],
        "manifest_sha256": report["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
