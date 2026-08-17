#!/usr/bin/env python3
"""Authorize and freeze the C67 s10/s20 680-pair rollout grid."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


def load_sibling(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen sibling: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SOURCE = load_sibling("_c67_rollout_source_freeze", "freeze_c67_rollout_source.py")
FORMAT = "h3wam-c67-budget-rollout-authorization-v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TRIALS = tuple(range(33, 50))
MIN_CHECKPOINT_BYTES = 10 * 1024**3
HISTORICAL_C60_DATA_SHA256 = {
    "demo_manifest_sha256": "b0d611c21059fa7da6fb08162b03efadd59aff68354bb101be41d3ae20d98eb1",
    "source_manifest_sha256": "cab8876f067114dce41d16ca52cb0bafddf17da33c92d0adde5f11d7ac9555b9",
    "demo_stats_sha256": "6f7e9f4a2232a798e4e30ad26f5748e71aeeda7fa54cb6ea2d0a3ec7d290e814",
    "c48_dataset_sha256": "d416d86c09ba334fae449a131510b84fa1d111e665a77eabfb248f1c79a5bc61",
    "c48_observations_sha256": "399d93f31a8f26297145942387a233b9667049efc60ac1f46514a3f7ce77a638",
    "c59_completed_sha256": "4e67bb95b69ada2a854d3b2bf4ba434c6b3072c2bba11a91df2c30c6de5eeb99",
    "c59_sample_labels_sha256": "f2be6801cac2f1c5b680b30c5e089f47e2bf428f179ee13c1ae283e2d47a9d53",
}
CAUSAL_DATA_SHA256 = {
    "causal_failure_dataset_sha256": "1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4",
    "causal_failure_observations_sha256": "b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55",
}
DATA_ARGUMENTS = {
    "demo_manifest_sha256": "demo_manifest",
    "source_manifest_sha256": "source_manifest",
    "demo_stats_sha256": "demo_stats",
    "c48_dataset_sha256": "c48_dataset",
    "c48_observations_sha256": "c48_observations",
    "c59_completed_sha256": "c59_completed",
    "c59_sample_labels_sha256": "c59_sample_labels",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_checkpoint(path: Path, *, milestone: int) -> tuple[str, dict[str, Any]]:
    path = path.resolve()
    if path.stat().st_size < MIN_CHECKPOINT_BYTES:
        raise ValueError(f"C67 checkpoint is unexpectedly small: {path}")
    digest = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    contract = payload.get("contract", {})
    required = {
        "schema_version", "completed_steps", "model", "optimizer",
        "lr_scheduler", "contract", "probe_step", "probe_predictions",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != 1
        or payload.get("completed_steps") != milestone
        or payload.get("probe_step") != milestone
        or payload.get("lr_scheduler", {}).get("last_epoch") != milestone
        or not isinstance(payload.get("probe_predictions"), list)
        or len(payload["probe_predictions"]) != 8
    ):
        raise ValueError(f"C67 s{milestone} checkpoint schema/step mismatch")
    fixed = {
        "format": "h3wam-c56b-fact-online-training-v1",
        "classification": "FACT_full_backbone_port_online_frozen_int8_h3",
        "scheduler_horizon": 20_000,
        "warmup_steps": 500,
        "base_lr": 2e-5,
        "action_lr": 2e-4,
        "seed": 20260816,
        "action_horizon": 32,
        "action_shift": 5.0,
        "h3_execution": "online_frozen_int8_per_rank_v1",
        "no_kv_cache": True,
        **HISTORICAL_C60_DATA_SHA256,
        **CAUSAL_DATA_SHA256,
    }
    mismatch = {
        key: {"actual": contract.get(key), "expected": value}
        for key, value in fixed.items() if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"C67 s{milestone} contract mismatch: {mismatch}")
    return digest, contract


def validate_offline(
    path: Path, training_complete_path: Path,
    endpoints: dict[str, tuple[Path, str, int]],
) -> dict[str, Any]:
    report = load_json(path.resolve())
    gates = report.get("gates")
    if (
        report.get("format") != "h3wam-c67-budget-balanced80-result-v1"
        or report.get("status") != "PASS_C67_BUDGET_BALANCED80_GATE"
        or report.get("permission") != "GO_C67_PAIRED_680_ROLLOUT"
        or not isinstance(gates, dict) or not gates or not all(gates.values())
    ):
        raise ValueError("C67 offline gate does not authorize paired rollout")
    completed = report.get("training_complete", {})
    training_marker = load_json(training_complete_path.resolve())
    if (
        Path(completed.get("path", "")).resolve() != training_complete_path.resolve()
        or completed.get("sha256") != sha256_file(training_complete_path.resolve())
        or not isinstance(completed.get("contract_sha256"), str)
        or len(completed["contract_sha256"]) != 64
        or completed["contract_sha256"] != training_marker.get("contract_sha256")
    ):
        raise ValueError("C67 offline gate/training-complete identity mismatch")
    identity = report.get("endpoint_identity", {})
    for name, (checkpoint, digest, milestone) in endpoints.items():
        row = identity.get(name, {})
        restore = Path(row.get("restore_audit", "")).resolve()
        if (
            row.get("milestone") != milestone
            or Path(row.get("checkpoint", "")).resolve() != checkpoint
            or row.get("checkpoint_sha256") != digest
            or not restore.is_file()
            or row.get("restore_audit_sha256") != sha256_file(restore)
        ):
            raise ValueError(f"C67 offline endpoint identity mismatch: {name}")
    return report


def build_jobs(
    output_root: Path, endpoints: dict[str, tuple[Path, str, int]]
) -> list[dict[str, Any]]:
    jobs = []
    for trial in TRIALS:
        for suite in SUITES:
            for task in range(10):
                pair_id = len(jobs) // 2
                for arm, (checkpoint, digest, milestone) in endpoints.items():
                    jobs.append({
                        "job_id": len(jobs), "pair_id": pair_id,
                        "gpu": pair_id % 8, "arm": arm,
                        "milestone": milestone,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": digest,
                        "suite": suite, "tasks": [task], "trials": [trial],
                        "episodes": 1,
                        "output": str(
                            output_root / "episodes" / arm / suite
                            / f"task{task:02d}_trial{trial:02d}"
                        ),
                    })
    pair_keys = {
        (row["pair_id"], row["suite"], row["tasks"][0], row["trials"][0])
        for row in jobs
    }
    if len(jobs) != 1_360 or len(pair_keys) != 680:
        raise AssertionError("C67 paired rollout grid is not exact")
    return jobs


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    snapshot = args.snapshot.resolve()
    source_freeze = SOURCE.verify(
        snapshot, args.source_freeze_sha256
    )
    data_paths = {
        key: Path(getattr(args, attribute)).resolve()
        for key, attribute in DATA_ARGUMENTS.items()
    }
    for key, path in data_paths.items():
        if not path.is_file() or sha256_file(path) != HISTORICAL_C60_DATA_SHA256[key]:
            raise ValueError(f"historical C60 data SHA256 mismatch: {key}")
    control_path, treatment_path = args.s10000.resolve(), args.s20000.resolve()
    control_sha, control_contract = validate_checkpoint(control_path, milestone=10_000)
    treatment_sha, treatment_contract = validate_checkpoint(
        treatment_path, milestone=20_000
    )
    if control_contract != treatment_contract:
        raise ValueError("C67 s10/s20 training contracts differ")
    training_complete_path = args.training_complete.resolve()
    training_complete = load_json(training_complete_path)
    completed_control = training_complete.get("matched_control", {})
    completed_treatment = training_complete.get("treatment", {})
    if (
        training_complete.get("format")
        != "h3wam-c67-c60-budget-ablation-training-complete-v1"
        or training_complete.get("status") != "PASS_C67_BUDGET_TRAINING_COMPLETE"
        or training_complete.get("permission") != "READY_FOR_PREREGISTERED_OFFLINE_ONLY"
        or training_complete.get("effect_status") != "NOT_EVIDENCE_READY"
        or training_complete.get("matched_control", {}).get("checkpoint_sha256")
        != control_sha
        or training_complete.get("treatment", {}).get("checkpoint_sha256")
        != treatment_sha
        or completed_control.get("milestone") != 10_000
        or Path(completed_control.get("checkpoint", "")).resolve() != control_path
        or completed_treatment.get("milestone") != 20_000
        or Path(completed_treatment.get("checkpoint", "")).resolve()
        != treatment_path
    ):
        raise ValueError("C67 training-complete contract mismatch")
    endpoints = {
        "matched_control": (control_path, control_sha, 10_000),
        "treatment": (treatment_path, treatment_sha, 20_000),
    }
    offline = validate_offline(
        args.offline_results.resolve(), training_complete_path, endpoints
    )
    jobs = build_jobs(output_root, endpoints)
    output_root.mkdir(parents=True)
    jobs_path = output_root / "jobs.jsonl"
    jobs_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in jobs),
        encoding="utf-8",
    )
    source_freeze_path = snapshot / SOURCE.MANIFEST_NAME
    report = {
        "format": FORMAT,
        "status": "AUTHORIZED_C67_S10_S20_PAIRED_680",
        "permission": "GO_C67_1360_FRESH_PROCESSES_NO_INTERMEDIATE_STOP",
        "effect_status": "NOT_EVIDENCE_READY",
        "release_signed": False,
        "endpoints": {
            name: {
                "milestone": milestone, "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
            }
            for name, (checkpoint, digest, milestone) in endpoints.items()
        },
        "training_complete": {
            "path": str(training_complete_path),
            "sha256": sha256_file(training_complete_path),
        },
        "offline_results": {
            "path": str(args.offline_results.resolve()),
            "sha256": sha256_file(args.offline_results.resolve()),
            "status": offline["status"], "permission": offline["permission"],
        },
        "source_freeze": {
            "snapshot": str(snapshot), "path": str(source_freeze_path),
            "sha256": sha256_file(source_freeze_path),
            "git_commit": source_freeze["git_commit"],
            "git_tree": source_freeze["git_tree"],
            "tracked_file_count": source_freeze["tracked_file_count"],
            "dynamic_execution_sha256": source_freeze["dynamic_execution_sha256"],
        },
        "historical_c60_data_sha256": HISTORICAL_C60_DATA_SHA256,
        "historical_c60_data_paths": {
            key: str(path) for key, path in data_paths.items()
        },
        "causal_data_sha256": CAUSAL_DATA_SHA256,
        "jobs": 1_360, "pairs": 680, "episodes_per_arm": 680,
        "one_episode_per_process": True,
        "suites": list(SUITES), "tasks_per_suite": 10,
        "trials": list(TRIALS), "manifest": str(jobs_path),
        "manifest_sha256": sha256_file(jobs_path),
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
            "treatment_successes_at_least_historical_c60": 313,
        },
        "stopping": (
            "Run all 1360 fresh processes. Never inspect intermediate success, "
            "stop early, or select a checkpoint."
        ),
        "claim_boundary": (
            "This unsigned artifact authorizes only the preregistered C67 s20-vs-s10 "
            "budget diagnostic. It cannot promote C67 or rewrite historical C60."
        ),
    }
    temporary = output_root / f".AUTHORIZATION.json.{os.getpid()}.partial"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_root / "AUTHORIZATION.json")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-complete", type=Path, required=True)
    parser.add_argument("--offline-results", type=Path, required=True)
    parser.add_argument("--s10000", type=Path, required=True)
    parser.add_argument("--s20000", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-freeze-sha256", required=True)
    parser.add_argument("--demo-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--demo-stats", type=Path, required=True)
    parser.add_argument("--c48-dataset", type=Path, required=True)
    parser.add_argument("--c48-observations", type=Path, required=True)
    parser.add_argument("--c59-completed", type=Path, required=True)
    parser.add_argument("--c59-sample-labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report = prepare(parse_args())
    print(json.dumps({
        "status": report["status"], "permission": report["permission"],
        "pairs": report["pairs"], "jobs": report["jobs"],
        "manifest_sha256": report["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
