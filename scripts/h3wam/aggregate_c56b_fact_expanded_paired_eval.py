#!/usr/bin/env python3
"""Aggregate C60 FACT against C58b over paired LIBERO trials33..49."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from aggregate_c58b_expanded_paired_eval import (
    initial_state_digest, paired_summary, same_object_joints, sha256_file,
    validate_result_contract, episode_map,
)


C60_SHA256 = "d6659c6b387f062a99f670a1d902b56df71a6bf1472aa4e46e56c9213ba75a36"
C58_SHA256 = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
TRIAL33_SHA256 = "fe4c7c49c6fd7e7ce0abf56c1f863c604fc05862dfc98fb5f0b4f8d00417ebe2"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TRIALS = tuple(range(33, 50))


def _load_trial33(
    path: Path, c60_checkpoint: Path, c58_checkpoint: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(path) != TRIAL33_SHA256:
        raise ValueError("C60 trial33 report SHA256 mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("format") != "h3wam-c56b-fact-paired-libero-trial33-v1"
        or report.get("status") != "COMPLETE"
        or report.get("paired_episodes_per_arm") != 40
        or report.get("main_checkpoint_sha256") != C60_SHA256
        or report.get("shared_c58_parent_sha256") != C58_SHA256
    ):
        raise ValueError("C60 trial33 report contract mismatch")
    rows = []
    for suite in SUITES:
        payloads = {}
        for arm, policy, checkpoint in (
            ("c60_main", "h3_fact_online_int8", c60_checkpoint),
            ("c58_parent", "h3_fastwam_online_int8", c58_checkpoint),
        ):
            source = report["sources"][arm][suite]
            source_path = Path(source["path"]).resolve()
            if sha256_file(source_path) != source["sha256"]:
                raise ValueError(f"trial33 source changed: {arm}/{suite}")
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            validate_result_contract(
                payload, policy=policy, checkpoint=checkpoint, suite=suite,
                tasks=list(range(10)), trials=[33], save_trajectories=False,
            )
            payloads[arm] = episode_map(payload)
        if set(payloads["c60_main"]) != set(payloads["c58_parent"]):
            raise ValueError(f"trial33 pair identity mismatch: {suite}")
        for (task, trial), candidate in payloads["c60_main"].items():
            control = payloads["c58_parent"][(task, trial)]
            if not same_object_joints(
                candidate["initial_object_joints"], control["initial_object_joints"]
            ):
                raise ValueError(f"trial33 initial state mismatch: {suite}/{task}")
            rows.append({
                "trial": trial, "suite": suite, "task": task,
                "candidate": bool(candidate["success"]),
                "control": bool(control["success"]),
                "mechanical_identity": "initial_object_joints_exact",
            })
    if len(rows) != 40:
        raise ValueError("trial33 does not contain exactly 40 pairs")
    return rows, report


def _job_evidence(spec: tuple[dict, str, Path]) -> dict[str, Any]:
    job, policy, checkpoint = spec
    result_path = Path(job["output"]) / "results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    validate_result_contract(
        payload, policy=policy, checkpoint=checkpoint, suite=job["suite"],
        tasks=job["tasks"], trials=job["trials"], save_trajectories=True,
    )
    episodes = episode_map(payload)
    key = (job["tasks"][0], job["trials"][0])
    if set(episodes) != {key}:
        raise ValueError(f"isolated episode identity mismatch: {result_path}")
    episode = episodes[key]
    trajectory = Path(episode.get("trajectory", "")).resolve()
    if not trajectory.is_file():
        raise FileNotFoundError(trajectory)
    return {
        "suite": job["suite"], "task": key[0], "trial": key[1],
        "success": bool(episode["success"]),
        "initial_object_joints": episode["initial_object_joints"],
        "initial_state_sha256": initial_state_digest(trajectory),
        "result": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "trajectory": str(trajectory),
        "trajectory_sha256": sha256_file(trajectory),
    }


def _load_arm(
    root: Path, *, label: str, candidate_dir: str, policy: str,
    checkpoint: Path, checkpoint_sha256: str, prepared_format: str,
    prepared_permission: str, completed_format: str, workers: int,
) -> tuple[dict[str, Any], dict[tuple[str, int, int], dict[str, Any]]]:
    root = root.resolve()
    if (root / "INVALID.json").exists():
        raise ValueError(f"{label} root is marked INVALID")
    prepared_path, manifest_path, completed_path = (
        root / "PREPARED.json", root / "jobs.jsonl", root / "COMPLETED.json"
    )
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    jobs = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    if (
        prepared.get("format") != prepared_format
        or prepared.get("permission") != prepared_permission
        or prepared.get("candidate_checkpoint_sha256") != checkpoint_sha256
        or prepared.get("jobs") != 640
        or prepared.get("candidate_episodes") != 640
        or prepared.get("one_episode_per_process") is not True
        or sha256_file(manifest_path) != prepared.get("manifest_sha256")
        or completed.get("format") != completed_format
        or completed.get("status") != "COMPLETE"
        or completed.get("episodes") != 640
        or completed.get("manifest_sha256") != prepared.get("manifest_sha256")
        or len(jobs) != 640
    ):
        raise ValueError(f"{label} preparation/completion mismatch")
    expected = {
        (suite, task, trial)
        for trial in range(34, 50) for suite in SUITES for task in range(10)
    }
    identities = {
        (job.get("suite"), job.get("tasks", [None])[0], job.get("trials", [None])[0])
        for job in jobs
        if job.get("episodes") == 1
        and len(job.get("tasks", [])) == 1
        and len(job.get("trials", [])) == 1
        and Path(job.get("output", "")).parts[-3] == candidate_dir
    }
    if identities != expected or len(identities) != len(jobs):
        raise ValueError(f"{label} manifest grid/path mismatch")
    specs = [(job, policy, checkpoint) for job in jobs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_job_evidence, specs))
    mapped = {
        (row["suite"], int(row["task"]), int(row["trial"])): row for row in rows
    }
    if set(mapped) != expected:
        raise ValueError(f"{label} evidence grid mismatch")
    return prepared, mapped


def aggregate(
    c60_root: Path, c58_root: Path, trial33_path: Path,
    c60_checkpoint: Path, c58_checkpoint: Path, workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    c60_checkpoint, c58_checkpoint = c60_checkpoint.resolve(), c58_checkpoint.resolve()
    if sha256_file(c60_checkpoint) != C60_SHA256:
        raise ValueError("C60 checkpoint SHA256 mismatch")
    if sha256_file(c58_checkpoint) != C58_SHA256:
        raise ValueError("C58 checkpoint SHA256 mismatch")
    trial33, trial33_report = _load_trial33(
        trial33_path.resolve(), c60_checkpoint, c58_checkpoint
    )
    c60_prepared, c60 = _load_arm(
        c60_root, label="C60", candidate_dir="candidate_c60",
        policy="h3_fact_online_int8", checkpoint=c60_checkpoint,
        checkpoint_sha256=C60_SHA256,
        prepared_format="h3wam-c56b-fact-expanded-prepared-v1",
        prepared_permission="GO_MECHANICAL_CANARY_THEN_8GPU_640_FRESH_PROCESSES",
        completed_format="h3wam-c56b-fact-expanded-isolated-complete-v1",
        workers=workers,
    )
    c58_prepared, c58 = _load_arm(
        c58_root, label="C58", candidate_dir="candidate_c58b",
        policy="h3_fastwam_online_int8", checkpoint=c58_checkpoint,
        checkpoint_sha256=C58_SHA256,
        prepared_format="h3wam-c58b-expanded-paired-prepared-v1",
        prepared_permission="GO_8GPU_640_FRESH_PROCESSES_NO_INTERMEDIATE_STOP",
        completed_format="h3wam-c58b-expanded-isolated-candidate-complete-v1",
        workers=workers,
    )
    if set(c60) != set(c58):
        raise ValueError("C60/C58 expanded pair identities differ")
    expanded, evidence = [], []
    for key in sorted(c60):
        candidate, control = c60[key], c58[key]
        if (
            candidate["initial_state_sha256"] != control["initial_state_sha256"]
            or not same_object_joints(
                candidate["initial_object_joints"], control["initial_object_joints"]
            )
        ):
            raise ValueError(f"expanded initial state mismatch: {key}")
        suite, task, trial = key
        expanded.append({
            "trial": trial, "suite": suite, "task": task,
            "candidate": candidate["success"], "control": control["success"],
            "mechanical_identity": "full_trajectory_initial_state_exact",
        })
        evidence.append({
            "trial": trial, "suite": suite, "task": task,
            "candidate_result": candidate["result"],
            "candidate_result_sha256": candidate["result_sha256"],
            "candidate_trajectory": candidate["trajectory"],
            "candidate_trajectory_sha256": candidate["trajectory_sha256"],
            "control_result": control["result"],
            "control_result_sha256": control["result_sha256"],
            "control_trajectory": control["trajectory"],
            "control_trajectory_sha256": control["trajectory_sha256"],
            "initial_state_sha256": candidate["initial_state_sha256"],
        })
    pairs = sorted(trial33 + expanded, key=lambda row: (
        row["trial"], row["suite"], row["task"]
    ))
    if len(pairs) != 680 or len({
        (row["trial"], row["suite"], row["task"]) for row in pairs
    }) != 680:
        raise ValueError("final 680-pair identity mismatch")
    overall = paired_summary(pairs)
    per_suite = {
        suite: paired_summary([row for row in pairs if row["suite"] == suite])
        for suite in SUITES
    }
    per_trial = {
        str(trial): paired_summary([row for row in pairs if row["trial"] == trial])
        for trial in TRIALS
    }
    threshold = c60_prepared["evaluation_gate"]
    gates = {
        "all_680_pairs_complete": len(pairs) == 680,
        "trial33_preregistered_bridge": (
            trial33_report.get("main_successes") == 20
            and trial33_report.get("c58_parent_successes") == 18
        ),
        "expanded_initial_states_exact": all(
            row["mechanical_identity"] == "full_trajectory_initial_state_exact"
            for row in expanded
        ),
        "both_arms_one_episode_per_process": (
            c60_prepared.get("jobs") == 640 and c58_prepared.get("jobs") == 640
            and c60_prepared.get("one_episode_per_process") is True
            and c58_prepared.get("one_episode_per_process") is True
        ),
        "absolute_gain_at_least_0_03": (
            overall["success_rate_delta"] >= threshold["absolute_gain_at_least_0_03"]
        ),
        "net_wins_at_least_20": (
            overall["candidate_wins"] - overall["control_wins"]
            >= threshold["net_wins_at_least"]
        ),
        "one_sided_exact_mcnemar_p_at_most_0_05": (
            overall["one_sided_p_candidate_better"]
            <= threshold["one_sided_exact_mcnemar_p_at_most"]
        ),
        "no_suite_regression_below_minus_0_03": all(
            value["success_rate_delta"] >= threshold["no_suite_regression_below"]
            for value in per_suite.values()
        ),
    }
    passed = all(gates.values())
    report = {
        "format": "h3wam-c60-fact-vs-c58-expanded-paired-libero-trials33-49-v1",
        "status": "PASS_C60_FACT_EXPANDED_PAIRED" if passed else "FAIL_C60_FACT_EXPANDED_PAIRED",
        "permission": "GO_PROMOTE_C60_FACT" if passed else "KEEP_C58_PARENT",
        "effect_status": "EVIDENCE_READY" if passed else "NOT_EVIDENCE_READY",
        "candidate": "C60_FULL_FACT_PORT", "control": "C58B_FASTWAM_FULL30_H3_LAYERWISE",
        "candidate_checkpoint": str(c60_checkpoint),
        "candidate_checkpoint_sha256": C60_SHA256,
        "control_checkpoint": str(c58_checkpoint),
        "control_checkpoint_sha256": C58_SHA256,
        "trial33_report": str(trial33_path.resolve()),
        "trial33_report_sha256": TRIAL33_SHA256,
        "candidate_prepared_sha256": sha256_file(c60_root.resolve() / "PREPARED.json"),
        "candidate_manifest_sha256": c60_prepared["manifest_sha256"],
        "control_prepared_sha256": sha256_file(c58_root.resolve() / "PREPARED.json"),
        "control_manifest_sha256": c58_prepared["manifest_sha256"],
        "pairs": 680, "trials": list(TRIALS), "overall": overall,
        "per_suite": per_suite, "per_trial": per_trial, "gates": gates,
        "pair_outcomes": pairs,
        "claim_boundary": (
            "Paired LIBERO trials33..49 over four suites and forty tasks under "
            "fresh-process wait30/replan8/horizon32/eval10 execution."
        ),
    }
    return report, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c60-root", type=Path, required=True)
    parser.add_argument("--c58-root", type=Path, required=True)
    parser.add_argument("--trial33-results", type=Path, required=True)
    parser.add_argument("--c60-checkpoint", type=Path, required=True)
    parser.add_argument("--c58-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report, evidence = aggregate(
        args.c60_root, args.c58_root, args.trial33_results,
        args.c60_checkpoint, args.c58_checkpoint, args.workers,
    )
    evidence_path = output.with_name("PAIR_EVIDENCE.jsonl")
    if evidence_path.exists():
        raise FileExistsError(evidence_path)
    evidence_tmp = evidence_path.with_name(f".{evidence_path.name}.{os.getpid()}.partial")
    evidence_tmp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in evidence),
        encoding="utf-8",
    )
    report["pair_evidence"] = str(evidence_path)
    report["pair_evidence_sha256"] = sha256_file(evidence_tmp)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_name(f".{output.name}.{os.getpid()}.partial")
    output_tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(evidence_tmp, evidence_path)
    os.replace(output_tmp, output)
    print(json.dumps({
        "status": report["status"], "effect_status": report["effect_status"],
        "overall": report["overall"], "gates": report["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
