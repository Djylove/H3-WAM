#!/usr/bin/env python3
"""Authorize the fixed C67-joint versus C69-action-only paired LIBERO grid."""

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


BASE = load_sibling("_c67_c69_rollout_base", "prepare_c67_budget_rollout.py")
SOURCE = load_sibling("_c67_c69_rollout_source", "freeze_c67_rollout_source.py")
FORMAT = "h3wam-c67-c69-paired-rollout-authorization-v1"
SUITES = BASE.SUITES
# LIBERO exposes exactly fifty initial states per task.  Trials 33..49 were
# preregistered for C67's budget rollout but never executed because its offline
# gate failed, so they remain the only audited fresh 17-trial block.
TRIALS = tuple(range(33, 50))
HISTORICAL_C60_DATA_SHA256 = BASE.HISTORICAL_C60_DATA_SHA256
DATA_ARGUMENTS = BASE.DATA_ARGUMENTS
CAUSAL_DATA_SHA256 = BASE.CAUSAL_DATA_SHA256
MIN_CHECKPOINT_BYTES = BASE.MIN_CHECKPOINT_BYTES


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


def validate_endpoint(
    path: Path, *, arm: str, expected_sha256: str
) -> tuple[str, dict[str, Any]]:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size < MIN_CHECKPOINT_BYTES:
        raise ValueError(f"{arm} checkpoint is missing or unexpectedly small: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"{arm} checkpoint SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    contract = payload.get("contract", {})
    required = {
        "schema_version", "completed_steps", "model", "optimizer",
        "lr_scheduler", "contract", "probe_step", "probe_predictions",
    }
    common = {
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
        for key, value in common.items() if contract.get(key) != value
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != 1
        or payload.get("completed_steps") != 20_000
        or payload.get("probe_step") != 20_000
        or payload.get("lr_scheduler", {}).get("last_epoch") != 20_000
        or not isinstance(payload.get("probe_predictions"), list)
        or len(payload["probe_predictions"]) != 8
        or mismatch
    ):
        raise ValueError(f"{arm} checkpoint contract mismatch: {mismatch}")
    if arm == "c67_fact_joint":
        if (
            contract.get("objective_mode", "fact_joint") != "fact_joint"
            or contract.get("loss_weights") != [10.0, 1.0, 0.4, 0.4]
            or contract.get("frozen_auxiliary_parameters", []) != []
        ):
            raise ValueError("C67 endpoint is not the fixed joint objective")
    elif arm == "c69_action_only":
        frozen = contract.get("frozen_auxiliary_parameters", [])
        if (
            contract.get("objective_mode") != "action_only"
            or contract.get("loss_weights") != [10.0, 0.0, 0.0, 0.0]
            or not isinstance(frozen, list)
            or not frozen
        ):
            raise ValueError("C69 endpoint is not the fixed action-only objective")
    else:
        raise ValueError(f"unknown endpoint arm: {arm}")
    return digest, contract


def build_jobs(
    output_root: Path, endpoints: dict[str, tuple[Path, str, int]]
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for trial in TRIALS:
        for suite in SUITES:
            for task in range(10):
                pair_id = len(jobs) // 2
                for arm, (checkpoint, digest, milestone) in endpoints.items():
                    jobs.append({
                        "job_id": len(jobs),
                        "pair_id": pair_id,
                        "arm": arm,
                        "milestone": milestone,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": digest,
                        "suite": suite,
                        "tasks": [task],
                        "trials": [trial],
                        "episodes": 1,
                        "output": str(
                            output_root / "episodes" / arm / suite
                            / f"task{task:02d}_trial{trial:02d}"
                        ),
                    })
    if len(jobs) != 1_360 or len({row["pair_id"] for row in jobs}) != 680:
        raise AssertionError("C67/C69 paired rollout grid is not exact")
    return jobs


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    snapshot = args.snapshot.resolve()
    source_freeze = SOURCE.verify(snapshot, args.source_freeze_sha256)
    data_paths = {
        key: Path(getattr(args, attribute)).resolve()
        for key, attribute in DATA_ARGUMENTS.items()
    }
    for key, path in data_paths.items():
        if not path.is_file() or sha256_file(path) != HISTORICAL_C60_DATA_SHA256[key]:
            raise ValueError(f"historical C60 data SHA mismatch: {key}")

    offline_path = args.offline_attribution.resolve()
    offline = load_json(offline_path)
    if (
        offline.get("format") != "h3wam-c67-c69-fixed-s20-attribution-gate-v1"
        or offline.get("status") != "PASS_C67_C69_FIXED_S20_ATTRIBUTION_CHAIN"
        or offline.get("permission")
        != "GO_C67_VS_C69_FIXED_S20_PAIRED_LIBERO_ATTRIBUTION"
        or offline.get("effect_status")
        != "OFFLINE_ATTRIBUTION_NOT_WINNER_NOT_CLOSED_LOOP_EVIDENCE"
        or not offline.get("evidence_gates")
        or not all(offline["evidence_gates"].values())
    ):
        raise ValueError("offline attribution does not authorize paired rollout")

    offline_endpoints = offline.get("fixed_endpoint_identity", {})
    paths = {
        "c67_fact_joint": args.c67_s20000.resolve(),
        "c69_action_only": args.c69_s20000.resolve(),
    }
    endpoints: dict[str, tuple[Path, str, int]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    for arm, path in paths.items():
        offline_name = "c67" if arm == "c67_fact_joint" else "c69"
        row = offline_endpoints.get(offline_name, {})
        expected_sha = row.get("checkpoint_sha256", "")
        if (
            row.get("milestone") != 20_000
            or Path(row.get("checkpoint", "")).resolve() != path
            or len(expected_sha) != 64
        ):
            raise ValueError(f"offline endpoint identity mismatch: {arm}")
        digest, contract = validate_endpoint(
            path, arm=arm, expected_sha256=expected_sha
        )
        endpoints[arm] = (path, digest, 20_000)
        contracts[arm] = contract

    complete_paths = {
        "c67": args.c67_training_complete.resolve(),
        "c69": args.c69_training_complete.resolve(),
    }
    for name, path in complete_paths.items():
        expected = offline.get("training_complete", {}).get(name, {})
        if (
            not path.is_file()
            or Path(expected.get("path", "")).resolve() != path
            or expected.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"training-complete identity mismatch: {name}")
    c67_complete = load_json(complete_paths["c67"])
    c69_complete = load_json(complete_paths["c69"])
    if (
        c67_complete.get("format")
        != "h3wam-c67-c60-budget-ablation-training-complete-v1"
        or c67_complete.get("status") != "PASS_C67_BUDGET_TRAINING_COMPLETE"
        or c67_complete.get("treatment", {}).get("checkpoint_sha256")
        != endpoints["c67_fact_joint"][1]
        or c69_complete.get("format")
        != "h3wam-c69-matched-action-only-training-complete-v1"
        or c69_complete.get("status")
        != "PASS_C69_MATCHED_ACTION_ONLY_TRAINING_COMPLETE"
        or c69_complete.get("final_checkpoint_sha256")
        != endpoints["c69_action_only"][1]
    ):
        raise ValueError("final training-complete endpoint binding failed")

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
        "status": "AUTHORIZED_C67_C69_FIXED_S20_PAIRED_680",
        "permission": "GO_C67_C69_1360_FRESH_PROCESSES_NO_INTERMEDIATE_STOP",
        "effect_status": "NOT_EVIDENCE_READY",
        "release_signed": False,
        "hypothesis": (
            "At an identical 20k action-tower budget, C67 joint consequence "
            "supervision changes closed-loop LIBERO success relative to C69 action-only."
        ),
        "endpoints": {
            name: {
                "milestone": milestone,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
                "objective_mode": contracts[name].get(
                    "objective_mode", "fact_joint"
                ),
                "loss_weights": contracts[name].get("loss_weights"),
            }
            for name, (checkpoint, digest, milestone) in endpoints.items()
        },
        "training_complete": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in complete_paths.items()
        },
        "offline_attribution": {
            "path": str(offline_path),
            "sha256": sha256_file(offline_path),
            "status": offline["status"],
            "permission": offline["permission"],
        },
        "source_freeze": {
            "snapshot": str(snapshot),
            "path": str(source_freeze_path),
            "sha256": sha256_file(source_freeze_path),
            "git_commit": source_freeze["git_commit"],
            "git_tree": source_freeze["git_tree"],
            "tracked_file_count": source_freeze["tracked_file_count"],
            "dynamic_execution_sha256": source_freeze[
                "dynamic_execution_sha256"
            ],
        },
        "historical_c60_data_sha256": HISTORICAL_C60_DATA_SHA256,
        "historical_c60_data_paths": {
            key: str(path) for key, path in data_paths.items()
        },
        "causal_data_sha256": CAUSAL_DATA_SHA256,
        "jobs": 1_360,
        "pairs": 680,
        "episodes_per_arm": 680,
        "one_episode_per_process": True,
        "suites": list(SUITES),
        "tasks_per_suite": 10,
        "trials": list(TRIALS),
        "manifest": str(jobs_path),
        "manifest_sha256": sha256_file(jobs_path),
        "protocol": {
            "wait_steps": 30,
            "max_steps": 400,
            "replan_steps": 8,
            "action_horizon": 32,
            "model_evaluations": 10,
            "seed": 42,
            "episode_seed_contract": "42+task*100000+trial*1000",
            "normalized_action_pre_clamp": True,
            "save_trajectories": True,
        },
        "attribution_threshold": {
            "absolute_delta": 0.03,
            "net_wins": 20,
            "one_sided_exact_mcnemar_p": 0.05,
            "suite_regression_floor": -0.03,
        },
        "stopping": (
            "Run all 1360 fresh processes; never inspect intermediate success, "
            "stop early, or replace either fixed s20000 endpoint."
        ),
        "claim_boundary": (
            "This authorization tests consequence-objective attribution only. "
            "It does not promote either endpoint over the C58 champion."
        ),
    }
    temporary = output_root / f".AUTHORIZATION.json.{os.getpid()}.partial"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_root / "AUTHORIZATION.json")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c67-training-complete", type=Path, required=True)
    parser.add_argument("--c69-training-complete", type=Path, required=True)
    parser.add_argument("--offline-attribution", type=Path, required=True)
    parser.add_argument("--c67-s20000", type=Path, required=True)
    parser.add_argument("--c69-s20000", type=Path, required=True)
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
        "status": report["status"],
        "permission": report["permission"],
        "pairs": report["pairs"],
        "jobs": report["jobs"],
        "manifest_sha256": report["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
