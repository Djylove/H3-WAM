#!/usr/bin/env python3
"""Aggregate the preregistered C67 s20-vs-s10 680-pair budget diagnostic."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

def load_sibling(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen sibling: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_sibling(
    "_c67_aggregate_paired_protocol", "aggregate_c58b_expanded_paired_eval.py"
)
SOURCE = load_sibling("_c67_aggregate_source_freeze", "freeze_c67_rollout_source.py")
PREPARE = load_sibling("_c67_aggregate_prepare", "prepare_c67_budget_rollout.py")
episode_map = BASE.episode_map
initial_state_digest = BASE.initial_state_digest
paired_summary = BASE.paired_summary
same_object_joints = BASE.same_object_joints
sha256_file = BASE.sha256_file
validate_result_contract = BASE.validate_result_contract
SUITES = PREPARE.SUITES
TRIALS = PREPARE.TRIALS


def _job_evidence(spec: tuple[dict[str, Any], str, Path]) -> dict[str, Any]:
    job, authorization_sha, authorization_path = spec
    checkpoint = Path(job["checkpoint"]).resolve()
    result_path = Path(job["output"]) / "results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    validate_result_contract(
        payload, policy="h3_fact_online_int8", checkpoint=checkpoint,
        suite=job["suite"], tasks=job["tasks"], trials=job["trials"],
        save_trajectories=True,
    )
    if (
        Path(payload.get("c67_budget_rollout_authorization", "")).resolve()
        != authorization_path.resolve()
        or payload.get("c67_budget_rollout_authorization_sha256")
        != authorization_sha
    ):
        raise ValueError(f"C67 result authorization identity mismatch: {result_path}")
    episodes = episode_map(payload)
    key = (job["tasks"][0], job["trials"][0])
    if set(episodes) != {key}:
        raise ValueError(f"C67 isolated episode identity mismatch: {result_path}")
    episode = episodes[key]
    trajectory = Path(episode.get("trajectory", "")).resolve()
    if not trajectory.is_file():
        raise FileNotFoundError(trajectory)
    return {
        "pair_id": int(job["pair_id"]), "arm": job["arm"],
        "suite": job["suite"], "task": key[0], "trial": key[1],
        "success": bool(episode["success"]),
        "initial_object_joints": episode["initial_object_joints"],
        "initial_state_sha256": initial_state_digest(trajectory),
        "result": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "trajectory": str(trajectory),
        "trajectory_sha256": sha256_file(trajectory),
    }


def decision_gates(
    pairs: list[dict[str, Any]], threshold: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, bool]]:
    overall = paired_summary(pairs)
    per_suite = {
        suite: paired_summary([row for row in pairs if row["suite"] == suite])
        for suite in SUITES
    }
    gates = {
        "all_680_pairs_complete": len(pairs) == 680,
        "absolute_gain_at_least_0_03": (
            overall["success_rate_delta"]
            >= threshold["absolute_gain_at_least_0_03"]
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
            row["success_rate_delta"] >= threshold["no_suite_regression_below"]
            for row in per_suite.values()
        ),
        "treatment_successes_at_least_historical_c60_313": (
            overall["candidate_successes"]
            >= threshold["treatment_successes_at_least_historical_c60"]
        ),
    }
    return overall, per_suite, gates


def aggregate(root: Path, workers: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve()
    if (root / "INVALID.json").exists():
        raise ValueError("C67 rollout root is marked INVALID")
    authorization_path = root / "AUTHORIZATION.json"
    manifest_path = root / "jobs.jsonl"
    completed_path = root / "COMPLETED.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    jobs = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    authorization_sha = sha256_file(authorization_path)
    if (
        authorization.get("format") != PREPARE.FORMAT
        or authorization.get("status") != "AUTHORIZED_C67_S10_S20_PAIRED_680"
        or authorization.get("permission")
        != "GO_C67_1360_FRESH_PROCESSES_NO_INTERMEDIATE_STOP"
        or authorization.get("effect_status") != "NOT_EVIDENCE_READY"
        or authorization.get("release_signed") is not False
        or authorization.get("jobs") != 1_360
        or authorization.get("pairs") != 680
        or authorization.get("episodes_per_arm") != 680
        or authorization.get("one_episode_per_process") is not True
        or authorization.get("historical_c60_data_sha256")
        != PREPARE.HISTORICAL_C60_DATA_SHA256
        or sha256_file(manifest_path) != authorization.get("manifest_sha256")
        or completed.get("format")
        != "h3wam-c67-budget-paired680-isolated-complete-v1"
        or completed.get("status") != "COMPLETE"
        or completed.get("jobs") != 1_360
        or completed.get("pairs") != 680
        or completed.get("episodes_per_arm") != 680
        or completed.get("authorization_sha256") != authorization_sha
        or completed.get("manifest_sha256") != authorization.get("manifest_sha256")
        or len(jobs) != 1_360
    ):
        raise ValueError("C67 authorization/manifest/completion contract mismatch")
    source = authorization["source_freeze"]
    frozen = SOURCE.verify(Path(source["snapshot"]), source["sha256"])
    if (
        frozen["git_commit"] != source["git_commit"]
        or frozen["git_tree"] != source["git_tree"]
        or frozen["dynamic_execution_sha256"]
        != source["dynamic_execution_sha256"]
    ):
        raise ValueError("C67 complete source identity changed")
    for key, path in authorization["historical_c60_data_paths"].items():
        if (
            key not in PREPARE.HISTORICAL_C60_DATA_SHA256
            or sha256_file(Path(path)) != PREPARE.HISTORICAL_C60_DATA_SHA256[key]
        ):
            raise ValueError(f"C67 historical data changed before aggregation: {key}")
    expected = {
        (arm, suite, task, trial)
        for arm in ("matched_control", "treatment")
        for trial in TRIALS for suite in SUITES for task in range(10)
    }
    actual = {
        (row.get("arm"), row.get("suite"), row.get("tasks", [None])[0],
         row.get("trials", [None])[0])
        for row in jobs if row.get("episodes") == 1
    }
    if actual != expected or len(actual) != len(jobs):
        raise ValueError("C67 exact 1360-job grid mismatch")
    endpoints = authorization["endpoints"]
    for name, endpoint in endpoints.items():
        checkpoint = Path(endpoint["checkpoint"])
        if (
            not checkpoint.is_file()
            or sha256_file(checkpoint) != endpoint["checkpoint_sha256"]
            or endpoint["milestone"]
            != {"matched_control": 10_000, "treatment": 20_000}[name]
        ):
            raise ValueError(f"C67 endpoint changed before aggregation: {name}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(
            _job_evidence,
            ((job, authorization_sha, authorization_path) for job in jobs),
        ))
    mapped = {
        (row["arm"], row["suite"], row["task"], row["trial"]): row
        for row in rows
    }
    if set(mapped) != expected:
        raise ValueError("C67 result evidence grid mismatch")
    pairs, evidence = [], []
    for trial in TRIALS:
        for suite in SUITES:
            for task in range(10):
                key = (suite, task, trial)
                control = mapped[("matched_control", *key)]
                treatment = mapped[("treatment", *key)]
                if (
                    treatment["pair_id"] != control["pair_id"]
                    or treatment["initial_state_sha256"]
                    != control["initial_state_sha256"]
                    or not same_object_joints(
                        treatment["initial_object_joints"],
                        control["initial_object_joints"],
                    )
                ):
                    raise ValueError(f"C67 paired initial-state mismatch: {key}")
                pairs.append({
                    "trial": trial, "suite": suite, "task": task,
                    "candidate": treatment["success"],
                    "control": control["success"],
                    "mechanical_identity": "full_trajectory_initial_state_exact",
                })
                evidence.append({
                    "trial": trial, "suite": suite, "task": task,
                    "treatment_result": treatment["result"],
                    "treatment_result_sha256": treatment["result_sha256"],
                    "treatment_trajectory": treatment["trajectory"],
                    "treatment_trajectory_sha256": treatment["trajectory_sha256"],
                    "control_result": control["result"],
                    "control_result_sha256": control["result_sha256"],
                    "control_trajectory": control["trajectory"],
                    "control_trajectory_sha256": control["trajectory_sha256"],
                    "initial_state_sha256": treatment["initial_state_sha256"],
                })
    if len(pairs) != 680 or len({
        (row["suite"], row["task"], row["trial"]) for row in pairs
    }) != 680:
        raise ValueError("C67 final 680-pair identity mismatch")
    overall, per_suite, effect_gates = decision_gates(
        pairs, authorization["evaluation_gate"]
    )
    gates = {
        "authorization_and_completion_exact": True,
        "complete_commit_tree_dynamic_source_frozen": True,
        "historical_seven_data_sha256_exact": True,
        "both_arms_680_fresh_processes": True,
        "all_initial_states_exact": True,
        **effect_gates,
    }
    passed = all(gates.values())
    report = {
        "format": "h3wam-c67-budget-s20-vs-s10-paired680-result-v1",
        "status": (
            "PASS_C67_BUDGET_PAIRED_680" if passed
            else "FAIL_C67_BUDGET_PAIRED_680"
        ),
        "permission": (
            "EVIDENCE_READY_BUDGET_ABLATION_ONLY" if passed
            else "KEEP_C58_PARENT_NO_BUDGET_EFFECT"
        ),
        "effect_status": (
            "EVIDENCE_READY_BUDGET_ABLATION_ONLY" if passed
            else "NOT_EVIDENCE_READY"
        ),
        "candidate": "C67_S20000_TREATMENT",
        "control": "C67_S10000_MATCHED_CONTROL",
        "endpoints": endpoints,
        "historical_c60_external_anchor": {
            "checkpoint_sha256": (
                "d6659c6b387f062a99f670a1d902b56df71a6bf1472aa4e46e56c9213ba75a36"
            ),
            "successes": 313, "pairs": 680,
        },
        "authorization": str(authorization_path),
        "authorization_sha256": authorization_sha,
        "manifest_sha256": authorization["manifest_sha256"],
        "source_freeze": source,
        "historical_c60_data_sha256": PREPARE.HISTORICAL_C60_DATA_SHA256,
        "pairs": 680, "trials": list(TRIALS), "overall": overall,
        "per_suite": per_suite,
        "per_trial": {
            str(trial): paired_summary([row for row in pairs if row["trial"] == trial])
            for trial in TRIALS
        },
        "gates": gates,
        "pair_outcomes": pairs,
        "claim_boundary": (
            "A pass supports only the same-trajectory C67 20k-vs-10k budget effect "
            "on reused trials33..49. It does not promote C67, alter historical C60, "
            "or establish fresh benchmark generalization."
        ),
    }
    return report, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    output = args.output.resolve()
    evidence_path = output.with_name("PAIR_EVIDENCE.jsonl")
    if output.exists() or evidence_path.exists():
        raise FileExistsError("refusing existing C67 aggregate/evidence")
    report, evidence = aggregate(args.root, args.workers)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_tmp = evidence_path.with_name(f".{evidence_path.name}.{os.getpid()}.partial")
    evidence_tmp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in evidence),
        encoding="utf-8",
    )
    report["pair_evidence"] = str(evidence_path)
    report["pair_evidence_sha256"] = sha256_file(evidence_tmp)
    output_tmp = output.with_name(f".{output.name}.{os.getpid()}.partial")
    output_tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(evidence_tmp, evidence_path)
    os.replace(output_tmp, output)
    print(json.dumps({
        "status": report["status"], "permission": report["permission"],
        "overall": report["overall"], "gates": report["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
