#!/usr/bin/env python3
"""Aggregate fixed C67-joint versus C69-action-only paired LIBERO evidence."""

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
    "_c67_c69_paired_base", "aggregate_c58b_expanded_paired_eval.py"
)
PREPARE = load_sibling(
    "_c67_c69_paired_prepare", "prepare_c67_c69_attribution_rollout.py"
)
SOURCE = load_sibling("_c67_c69_paired_source", "freeze_c67_rollout_source.py")


def job_evidence(spec: tuple[dict[str, Any], str, Path]) -> dict[str, Any]:
    job, authorization_sha, authorization_path = spec
    checkpoint = Path(job["checkpoint"]).resolve()
    result_path = Path(job["output"]) / "results.json"
    payload = json.loads(result_path.read_text())
    BASE.validate_result_contract(
        payload,
        policy="h3_fact_online_int8",
        checkpoint=checkpoint,
        suite=job["suite"],
        tasks=job["tasks"],
        trials=job["trials"],
        save_trajectories=True,
    )
    if (
        Path(payload.get("c67_c69_attribution_authorization", "")).resolve()
        != authorization_path.resolve()
        or payload.get("c67_c69_attribution_authorization_sha256")
        != authorization_sha
    ):
        raise ValueError(f"C67/C69 result authorization mismatch: {result_path}")
    episodes = BASE.episode_map(payload)
    key = (job["tasks"][0], job["trials"][0])
    if set(episodes) != {key}:
        raise ValueError(f"C67/C69 isolated result identity mismatch: {result_path}")
    episode = episodes[key]
    trajectory = Path(episode.get("trajectory", "")).resolve()
    if not trajectory.is_file():
        raise FileNotFoundError(trajectory)
    return {
        "pair_id": int(job["pair_id"]),
        "arm": job["arm"],
        "suite": job["suite"],
        "task": key[0],
        "trial": key[1],
        "success": bool(episode["success"]),
        "initial_object_joints": episode["initial_object_joints"],
        "initial_state_sha256": BASE.initial_state_digest(trajectory),
        "result": str(result_path),
        "result_sha256": BASE.sha256_file(result_path),
        "trajectory": str(trajectory),
        "trajectory_sha256": BASE.sha256_file(trajectory),
    }


def directional_decision(
    overall: dict[str, Any], per_suite: dict[str, dict[str, Any]], threshold: dict[str, Any]
) -> tuple[str, dict[str, bool], dict[str, bool]]:
    delta = overall["success_rate_delta"]
    c67_gates = {
        "absolute_gain": delta >= threshold["absolute_delta"],
        "net_wins": overall["candidate_wins"] - overall["control_wins"]
        >= threshold["net_wins"],
        "one_sided_p": overall["one_sided_p_candidate_better"]
        <= threshold["one_sided_exact_mcnemar_p"],
        "no_suite_regression": all(
            row["success_rate_delta"] >= threshold["suite_regression_floor"]
            for row in per_suite.values()
        ),
    }
    reverse_p = BASE.exact_mcnemar(
        overall["control_wins"], overall["candidate_wins"]
    )["one_sided_p_candidate_better"]
    c69_gates = {
        "absolute_gain": -delta >= threshold["absolute_delta"],
        "net_wins": overall["control_wins"] - overall["candidate_wins"]
        >= threshold["net_wins"],
        "one_sided_p": reverse_p <= threshold["one_sided_exact_mcnemar_p"],
        "no_suite_regression": all(
            -row["success_rate_delta"] >= threshold["suite_regression_floor"]
            for row in per_suite.values()
        ),
    }
    if all(c67_gates.values()):
        decision = "SUPPORT_C67_CONSEQUENCE_OBJECTIVE"
    elif all(c69_gates.values()):
        decision = "SUPPORT_C69_ACTION_ONLY_REJECT_INCREMENTAL_CONSEQUENCE_VALUE"
    else:
        decision = "NO_DETECTABLE_INCREMENTAL_CONSEQUENCE_EFFECT"
    return decision, c67_gates, c69_gates


def aggregate(root: Path, workers: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve()
    authorization_path = root / "AUTHORIZATION.json"
    manifest_path = root / "jobs.jsonl"
    completed_path = root / "COMPLETED.json"
    authorization = json.loads(authorization_path.read_text())
    completed = json.loads(completed_path.read_text())
    jobs = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    authorization_sha = BASE.sha256_file(authorization_path)
    if (
        authorization.get("format") != PREPARE.FORMAT
        or authorization.get("status")
        != "AUTHORIZED_C67_C69_FIXED_S20_PAIRED_680"
        or authorization.get("jobs") != 1_360
        or authorization.get("pairs") != 680
        or BASE.sha256_file(manifest_path) != authorization.get("manifest_sha256")
        or completed.get("format")
        != "h3wam-c67-c69-paired680-isolated-complete-v1"
        or completed.get("status") != "COMPLETE"
        or completed.get("authorization_sha256") != authorization_sha
        or completed.get("manifest_sha256") != authorization.get("manifest_sha256")
        or len(jobs) != 1_360
    ):
        raise ValueError("C67/C69 authorization/manifest/completion mismatch")
    source = authorization["source_freeze"]
    frozen = SOURCE.verify(Path(source["snapshot"]), source["sha256"])
    if (
        frozen["git_commit"] != source["git_commit"]
        or frozen["git_tree"] != source["git_tree"]
        or frozen["dynamic_execution_sha256"] != source["dynamic_execution_sha256"]
    ):
        raise ValueError("C67/C69 source identity changed")
    expected = {
        (arm, suite, task, trial)
        for arm in ("c67_fact_joint", "c69_action_only")
        for trial in PREPARE.TRIALS
        for suite in PREPARE.SUITES
        for task in range(10)
    }
    actual = {
        (row.get("arm"), row.get("suite"), row.get("tasks", [None])[0],
         row.get("trials", [None])[0])
        for row in jobs if row.get("episodes") == 1
    }
    if actual != expected or len(actual) != len(jobs):
        raise ValueError("C67/C69 exact 1360-job grid mismatch")
    for endpoint in authorization["endpoints"].values():
        checkpoint = Path(endpoint["checkpoint"])
        if (
            endpoint.get("milestone") != 20_000
            or not checkpoint.is_file()
            or BASE.sha256_file(checkpoint) != endpoint["checkpoint_sha256"]
        ):
            raise ValueError("C67/C69 endpoint changed before aggregation")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(
            job_evidence,
            ((job, authorization_sha, authorization_path) for job in jobs),
        ))
    mapped = {
        (row["arm"], row["suite"], row["task"], row["trial"]): row
        for row in rows
    }
    if set(mapped) != expected:
        raise ValueError("C67/C69 result evidence grid mismatch")
    pairs, evidence = [], []
    for trial in PREPARE.TRIALS:
        for suite in PREPARE.SUITES:
            for task in range(10):
                joint = mapped[("c67_fact_joint", suite, task, trial)]
                action = mapped[("c69_action_only", suite, task, trial)]
                if (
                    joint["pair_id"] != action["pair_id"]
                    or joint["initial_state_sha256"] != action["initial_state_sha256"]
                    or not BASE.same_object_joints(
                        joint["initial_object_joints"], action["initial_object_joints"]
                    )
                ):
                    raise ValueError(f"C67/C69 paired initial-state mismatch: {suite}/{task}/{trial}")
                pairs.append({
                    "trial": trial,
                    "suite": suite,
                    "task": task,
                    "candidate": joint["success"],
                    "control": action["success"],
                    "candidate_arm": "c67_fact_joint",
                    "control_arm": "c69_action_only",
                    "mechanical_identity": "full_trajectory_initial_state_exact",
                })
                evidence.append({
                    "trial": trial,
                    "suite": suite,
                    "task": task,
                    "c67_result": joint["result"],
                    "c67_result_sha256": joint["result_sha256"],
                    "c67_trajectory": joint["trajectory"],
                    "c67_trajectory_sha256": joint["trajectory_sha256"],
                    "c69_result": action["result"],
                    "c69_result_sha256": action["result_sha256"],
                    "c69_trajectory": action["trajectory"],
                    "c69_trajectory_sha256": action["trajectory_sha256"],
                    "initial_state_sha256": joint["initial_state_sha256"],
                })
    overall = BASE.paired_summary(pairs)
    per_suite = {
        suite: BASE.paired_summary([row for row in pairs if row["suite"] == suite])
        for suite in PREPARE.SUITES
    }
    decision, c67_gates, c69_gates = directional_decision(
        overall, per_suite, authorization["attribution_threshold"]
    )
    report = {
        "format": "h3wam-c67-c69-fixed-s20-paired680-attribution-result-v1",
        "status": "PASS_C67_C69_PAIRED_680_ATTRIBUTION_EVIDENCE",
        "permission": "EVIDENCE_READY_ATTRIBUTION_ONLY_KEEP_C58_CHAMPION",
        "effect_status": "EVIDENCE_READY_ATTRIBUTION_ONLY",
        "decision": decision,
        "candidate": "C67_FACT_JOINT_S20000",
        "control": "C69_ACTION_ONLY_S20000",
        "endpoints": authorization["endpoints"],
        "authorization": str(authorization_path),
        "authorization_sha256": authorization_sha,
        "source_freeze": source,
        "pairs": 680,
        "trials": list(PREPARE.TRIALS),
        "overall": overall,
        "per_suite": per_suite,
        "per_trial": {
            str(trial): BASE.paired_summary(
                [row for row in pairs if row["trial"] == trial]
            )
            for trial in PREPARE.TRIALS
        },
        "directional_gates": {"c67_joint_better": c67_gates, "c69_action_only_better": c69_gates},
        "identity_gates": {
            "authorization_and_completion_exact": True,
            "complete_source_frozen": True,
            "both_arms_680_fresh_processes": True,
            "all_initial_states_exact": True,
            "fixed_s20000_no_checkpoint_selection": True,
        },
        "pair_outcomes": pairs,
        "claim_boundary": (
            "This result isolates the incremental closed-loop value of the C67 "
            "consequence objective versus its matched C69 action-only control. "
            "It does not establish either as champion or replace C58."
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
        raise FileExistsError("refusing existing C67/C69 aggregate/evidence")
    report, evidence = aggregate(args.root, args.workers)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_tmp = evidence_path.with_name(f".{evidence_path.name}.{os.getpid()}.partial")
    evidence_tmp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in evidence)
    )
    report["pair_evidence"] = str(evidence_path)
    report["pair_evidence_sha256"] = BASE.sha256_file(evidence_tmp)
    output_tmp = output.with_name(f".{output.name}.{os.getpid()}.partial")
    output_tmp.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(evidence_tmp, evidence_path)
    os.replace(output_tmp, output)
    print(json.dumps({
        "status": report["status"],
        "permission": report["permission"],
        "decision": report["decision"],
        "overall": report["overall"],
    }, indent=2))


if __name__ == "__main__":
    main()
