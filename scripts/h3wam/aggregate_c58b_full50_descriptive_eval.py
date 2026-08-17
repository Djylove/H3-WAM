#!/usr/bin/env python3
"""Aggregate descriptive C58/D0 LIBERO trials0..49 without a new promotion claim."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from aggregate_c58b_expanded_paired_eval import (
    C58_SHA256,
    D0_SHA256,
    initial_state_digest,
    paired_summary,
    same_object_joints,
    sha256_file,
    validate_episode,
    validate_result_contract,
)
from prepare_c58b_full50_descriptive_eval import (
    CONFIRMATORY_EVIDENCE_SHA256,
    CONFIRMATORY_FINAL_SHA256,
    SUITES,
)


def audit_job(spec: tuple[dict[str, Any], Path]) -> dict[str, Any]:
    job, expected_checkpoint = spec
    result = (Path(job["output"]) / "results.json").resolve()
    payload = json.loads(result.read_text(encoding="utf-8"))
    validate_result_contract(
        payload, policy=job["policy"], checkpoint=expected_checkpoint,
        suite=job["suite"], tasks=[job["task"]], trials=[job["trial"]],
        save_trajectories=True,
    )
    tasks = payload.get("tasks", [])
    if len(tasks) != 1 or tasks[0].get("task_id") != job["task"]:
        raise ValueError(f"full50 task payload mismatch: {result}")
    episodes = tasks[0].get("episodes", [])
    if len(episodes) != 1:
        raise ValueError(f"full50 episode payload mismatch: {result}")
    episode = episodes[0]
    validate_episode(episode, job["task"], job["trial"])
    trajectory = Path(episode.get("trajectory", "")).resolve()
    if not trajectory.is_file():
        raise FileNotFoundError(trajectory)
    return {
        "arm": job["arm"], "suite": job["suite"], "task": job["task"],
        "trial": job["trial"], "success": bool(episode["success"]),
        "initial_object_joints": episode["initial_object_joints"],
        "initial_state_sha256": initial_state_digest(trajectory),
        "result": str(result), "result_sha256": sha256_file(result),
        "trajectory": str(trajectory), "trajectory_sha256": sha256_file(trajectory),
    }


def aggregate(root: Path, workers: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve()
    prepared_path = root / "PREPARED.json"
    manifest_path = root / "jobs.jsonl"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if (
        prepared.get("permission") != "GO_DESCRIPTIVE_FULL50_SUPPLEMENT_NO_PROMOTION"
        or prepared.get("jobs") != 2640
        or prepared.get("episodes_per_arm") != 1320
        or sha256_file(manifest_path) != prepared.get("manifest_sha256")
        or not (root / "candidate_c58b.COMPLETED.json").is_file()
        or not (root / "control_d0.COMPLETED.json").is_file()
    ):
        raise ValueError("full50 preparation/completion contract mismatch")
    confirmatory_path = Path(prepared["confirmatory_final"]).resolve()
    if sha256_file(confirmatory_path) != CONFIRMATORY_FINAL_SHA256:
        raise ValueError("confirmatory FINAL changed")
    confirmatory = json.loads(confirmatory_path.read_text(encoding="utf-8"))
    confirmatory_evidence = Path(confirmatory["pair_evidence"]).resolve()
    if (
        sha256_file(confirmatory_evidence) != CONFIRMATORY_EVIDENCE_SHA256
        or confirmatory.get("status") != "PASS_C58B_EXPANDED_PAIRED"
        or confirmatory.get("effect_status") != "EVIDENCE_READY"
        or confirmatory.get("pairs") != 680
        or not all(confirmatory.get("gates", {}).values())
    ):
        raise ValueError("confirmatory C58 evidence contract mismatch")
    checkpoints = {
        arm: Path(spec["path"]).resolve()
        for arm, spec in prepared["checkpoints"].items()
    }
    if (
        sha256_file(checkpoints["candidate_c58b"]) != C58_SHA256
        or sha256_file(checkpoints["control_d0"]) != D0_SHA256
    ):
        raise ValueError("full50 checkpoint identity mismatch")
    jobs = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    identities = {(j["arm"], j["suite"], j["task"], j["trial"]) for j in jobs}
    if len(jobs) != len(identities) or len(jobs) != 2640:
        raise ValueError("full50 manifest identity mismatch")
    specs = [(job, checkpoints[job["arm"]]) for job in jobs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(audit_job, specs))
    mapped = {
        (row["arm"], row["suite"], row["task"], row["trial"]): row
        for row in rows
    }
    supplement = []
    evidence = []
    for trial in range(33):
        for suite in SUITES:
            for task in range(10):
                candidate = mapped[("candidate_c58b", suite, task, trial)]
                control = mapped[("control_d0", suite, task, trial)]
                if (
                    candidate["initial_state_sha256"] != control["initial_state_sha256"]
                    or not same_object_joints(
                        candidate["initial_object_joints"],
                        control["initial_object_joints"],
                    )
                ):
                    raise ValueError(f"full50 initial state mismatch: {(suite, task, trial)}")
                supplement.append({
                    "trial": trial, "suite": suite, "task": task,
                    "candidate": candidate["success"], "control": control["success"],
                    "evidence_class": "descriptive_consumed_trial_fresh_process_pair",
                })
                evidence.append({
                    "trial": trial, "suite": suite, "task": task,
                    "initial_state_sha256": candidate["initial_state_sha256"],
                    "candidate_result": candidate["result"],
                    "candidate_result_sha256": candidate["result_sha256"],
                    "candidate_trajectory": candidate["trajectory"],
                    "candidate_trajectory_sha256": candidate["trajectory_sha256"],
                    "control_result": control["result"],
                    "control_result_sha256": control["result_sha256"],
                    "control_trajectory": control["trajectory"],
                    "control_trajectory_sha256": control["trajectory_sha256"],
                })
    historical = [dict(row, evidence_class="confirmatory_trials33_49") for row in confirmatory["pair_outcomes"]]
    pairs = sorted(supplement + historical, key=lambda row: (row["trial"], row["suite"], row["task"]))
    if len(pairs) != 2000 or len({(r["trial"], r["suite"], r["task"]) for r in pairs}) != 2000:
        raise ValueError("full50 exact 2000-pair identity mismatch")
    report = {
        "format": "h3wam-c58b-vs-d0-libero-full50-descriptive-v1",
        "status": "COMPLETE_DESCRIPTIVE_BENCHMARK",
        "permission": "NO_NEW_PROMOTION_CLAIM",
        "effect_status": "DESCRIPTIVE_ONLY",
        "pairs": 2000,
        "trials": list(range(50)),
        "confirmatory_trials33_49": {
            "final": str(confirmatory_path),
            "final_sha256": CONFIRMATORY_FINAL_SHA256,
            "pair_evidence_sha256": CONFIRMATORY_EVIDENCE_SHA256,
            "status": confirmatory["status"], "overall": confirmatory["overall"],
        },
        "descriptive_supplement_trials0_32": paired_summary(supplement),
        "descriptive_full50": paired_summary(pairs),
        "per_suite": {
            suite: paired_summary([row for row in pairs if row["suite"] == suite])
            for suite in SUITES
        },
        "per_trial": {
            str(trial): paired_summary([row for row in pairs if row["trial"] == trial])
            for trial in range(50)
        },
        "gates": {
            "all_2640_supplement_episodes_complete": len(rows) == 2640,
            "all_1320_supplement_initial_states_exact": len(supplement) == 1320,
            "all_2000_pair_identities_exact": len(pairs) == 2000,
            "confirmatory_result_unchanged": True,
            "descriptive_boundary_enforced": True,
        },
        "pair_outcomes": pairs,
        "claim_boundary": (
            "Trials0..32 were previously consumed during development and are reported "
            "descriptively. Only the frozen trials33..49 result supports C58 carrier "
            "promotion; full50 statistics cannot create a new confirmatory claim."
        ),
    }
    return report, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report, evidence = aggregate(args.root, args.workers)
    evidence_path = args.output.with_name("SUPPLEMENT_PAIR_EVIDENCE.jsonl")
    if evidence_path.exists():
        raise FileExistsError(evidence_path)
    evidence_tmp = evidence_path.with_name(f".{evidence_path.name}.{os.getpid()}.partial")
    evidence_tmp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in evidence),
        encoding="utf-8",
    )
    report["supplement_pair_evidence"] = str(evidence_path.resolve())
    report["supplement_pair_evidence_sha256"] = sha256_file(evidence_tmp)
    output_tmp = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    output_tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(evidence_tmp, evidence_path)
    os.replace(output_tmp, args.output)
    print(json.dumps({
        "status": report["status"], "effect_status": report["effect_status"],
        "descriptive_full50": report["descriptive_full50"],
        "gates": report["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
