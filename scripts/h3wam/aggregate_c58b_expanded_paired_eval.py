#!/usr/bin/env python3
"""Aggregate C58b vs frozen D0 over LIBERO trials 33..49."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TRIALS = tuple(range(33, 50))
C58_SHA256 = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
D0_SHA256 = "36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"
TRIAL33_SHA256 = "f7e9c8f65c177d33a3b168d0e0a47e79034d0054c99866a66ba09f82ee916ab3"
INITIAL_KEYS = (
    "step", "agentview_image", "wristview_image", "eef_pos", "eef_quat",
    "gripper_qpos", "previous_action", "sim_state",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in INITIAL_KEYS:
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def initial_state_digest(path: Path) -> str:
    with np.load(path) as archive:
        if any(name not in archive or len(archive[name]) == 0 for name in INITIAL_KEYS):
            raise ValueError(f"trajectory initial state contract mismatch: {path}")
        values = {name: np.array(archive[name][0], copy=True) for name in INITIAL_KEYS}
    return tensor_digest(values)


def wilson(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count <= 0:
        raise ValueError("Wilson interval requires positive count")
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def exact_mcnemar(candidate_wins: int, control_wins: int) -> dict[str, Any]:
    discordant = candidate_wins + control_wins
    if discordant == 0:
        return {
            "discordant_pairs": 0, "candidate_wins": 0, "control_wins": 0,
            "one_sided_p_candidate_better": 1.0, "two_sided_p": 1.0,
            "method": "exact_binomial_mcnemar_p0_5",
        }
    denominator = 2**discordant
    upper = sum(
        math.comb(discordant, index)
        for index in range(candidate_wins, discordant + 1)
    ) / denominator
    lower = sum(
        math.comb(discordant, index)
        for index in range(0, min(candidate_wins, control_wins) + 1)
    ) / denominator
    return {
        "discordant_pairs": discordant,
        "candidate_wins": candidate_wins,
        "control_wins": control_wins,
        "one_sided_p_candidate_better": upper,
        "two_sided_p": min(1.0, 2.0 * lower),
        "method": "exact_binomial_mcnemar_p0_5",
    }


def paired_delta_ci(candidate: list[bool], control: list[bool]) -> list[float]:
    values = np.asarray(candidate, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    mean = float(values.mean())
    if len(values) <= 1:
        return [mean, mean]
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    radius = 1.959963984540054 * standard_error
    return [max(-1.0, mean - radius), min(1.0, mean + radius)]


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = [bool(row["candidate"]) for row in rows]
    control = [bool(row["control"]) for row in rows]
    candidate_successes = sum(candidate)
    control_successes = sum(control)
    candidate_wins = sum(a and not b for a, b in zip(candidate, control))
    control_wins = sum(b and not a for a, b in zip(candidate, control))
    count = len(rows)
    return {
        "pairs": count,
        "candidate_successes": candidate_successes,
        "control_successes": control_successes,
        "candidate_rate": candidate_successes / count,
        "control_rate": control_successes / count,
        "candidate_rate_wilson95": wilson(candidate_successes, count),
        "control_rate_wilson95": wilson(control_successes, count),
        "success_rate_delta": (candidate_successes - control_successes) / count,
        "paired_delta_normal95": paired_delta_ci(candidate, control),
        "ties": count - candidate_wins - control_wins,
        **exact_mcnemar(candidate_wins, control_wins),
    }


def validate_episode(episode: dict[str, Any], task: int, trial: int) -> None:
    expected_seed = 42 + task * 100_000 + trial * 1_000
    replans = int(episode.get("replans", -1))
    if (
        episode.get("trial") != trial
        or episode.get("episode_seed") != expected_seed
        or episode.get("environment_seed") is not None
        or replans <= 0
        or replans > 50
        or episode.get("replan_noise_seeds")
        != list(range(expected_seed, expected_seed + replans))
    ):
        raise ValueError(f"episode seed contract mismatch: task{task}/trial{trial}")
    for name, shape in (
        ("first_environment_action", (7,)),
        ("first_environment_action_chunk", (32, 7)),
        ("replan_first_actions", (replans, 7)),
    ):
        value = np.asarray(episode.get(name), dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"episode action payload mismatch: task{task}/trial{trial}/{name}")


def validate_result_contract(
    payload: dict[str, Any], *, policy: str, checkpoint: Path,
    suite: str, tasks: list[int], trials: list[int], save_trajectories: bool,
) -> None:
    if (
        payload.get("policy") != policy
        or Path(payload.get("checkpoint", "")).resolve() != checkpoint.resolve()
        or payload.get("suite") != suite
        or payload.get("task_ids") != tasks
        or payload.get("trial_indices") != trials
        or payload.get("trials_per_task") != len(trials)
        or payload.get("max_steps") != 400
        or payload.get("wait_steps") != 30
        or payload.get("replan_steps") != 8
        or payload.get("action_horizon") != 32
        or payload.get("model_evaluations") != 10
        or payload.get("environment_seed") is not None
        or payload.get("policy_noise_seed_base") is not None
        or payload.get("normalized_action_pre_clamp") is not True
        or payload.get("sample_ensemble_size") != 1
        or payload.get("use_action_ensembler") is not False
        or payload.get("save_trajectories") is not save_trajectories
        or payload.get("binarize_gripper") is not True
        or payload.get("context_mode") != "cached"
    ):
        raise ValueError(f"rollout result contract mismatch: {suite}/{trials}")


def episode_map(payload: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    result = {}
    for task_row in payload.get("tasks", []):
        task = int(task_row["task_id"])
        for episode in task_row.get("episodes", []):
            trial = int(episode["trial"])
            validate_episode(episode, task, trial)
            key = (task, trial)
            if key in result:
                raise ValueError(f"duplicate episode {key}")
            result[key] = episode
    return result


def load_trial33(
    report_path: Path, candidate_checkpoint: Path, d0_checkpoint: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(report_path) != TRIAL33_SHA256:
        raise ValueError("trial33 report SHA256 mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "COMPLETE"
        or report.get("paired_episodes_per_arm") != 40
        or report.get("candidate_successes") != 18
        or report.get("control_successes") != 16
    ):
        raise ValueError("trial33 bridge summary mismatch")
    rows = []
    for suite in SUITES:
        arm_payloads = {}
        for arm, policy, checkpoint in (
            ("candidate_c58b", "h3_fastwam_online_int8", candidate_checkpoint),
            ("control_d0", "h3_dreamwam_kv_int8", d0_checkpoint),
        ):
            source = report["sources"][arm][suite]
            path = Path(source["path"]).resolve()
            if sha256_file(path) != source["sha256"]:
                raise ValueError(f"trial33 source changed: {arm}/{suite}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            validate_result_contract(
                payload, policy=policy, checkpoint=checkpoint, suite=suite,
                tasks=list(range(10)), trials=[33], save_trajectories=False,
            )
            arm_payloads[arm] = episode_map(payload)
        if set(arm_payloads["candidate_c58b"]) != set(arm_payloads["control_d0"]):
            raise ValueError(f"trial33 pair identity mismatch: {suite}")
        for (task, trial), candidate in arm_payloads["candidate_c58b"].items():
            control = arm_payloads["control_d0"][(task, trial)]
            for name in candidate["initial_object_joints"]:
                if not np.array_equal(
                    np.asarray(candidate["initial_object_joints"][name]),
                    np.asarray(control["initial_object_joints"][name]),
                ):
                    raise ValueError(f"trial33 initial state mismatch: {suite}/{task}")
            rows.append({
                "trial": trial, "suite": suite, "task": task,
                "candidate": bool(candidate["success"]),
                "control": bool(control["success"]),
                "mechanical_identity": "initial_object_joints_exact",
            })
    return rows, report


def verify_frozen_control(row: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(row["result"]).resolve()
    trajectory = Path(row["trajectory"]).resolve()
    if sha256_file(result_path) != row["result_sha256"]:
        raise ValueError(f"frozen D0 result changed: {result_path}")
    if sha256_file(trajectory) != row["trajectory_sha256"]:
        raise ValueError(f"frozen D0 trajectory changed: {trajectory}")
    if initial_state_digest(trajectory) != row["initial_state_sha256"]:
        raise ValueError(f"frozen D0 initial digest mismatch: {trajectory}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    suite, task, trial = row["suite"], int(row["task"]), int(row["trial"])
    validate_result_contract(
        payload, policy="h3_dreamwam_kv_int8",
        checkpoint=Path(payload["checkpoint"]), suite=suite,
        tasks=[task], trials=[trial], save_trajectories=True,
    )
    episode = episode_map(payload)[(task, trial)]
    if bool(episode["success"]) != bool(row["success"]):
        raise ValueError(f"frozen D0 outcome mismatch: {suite}/{task}/{trial}")
    return row


def load_frozen_controls(ready_path: Path, workers: int) -> tuple[dict, dict[tuple[str, int, int], dict]]:
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if (
        ready.get("status") != "PASS_D0_CONTROL_REUSE"
        or ready.get("permission") != "GO_C58B_CANDIDATE_ONLY_TRIALS34_49"
        or ready.get("checkpoint_sha256") != D0_SHA256
        or ready.get("controls") != 640
    ):
        raise ValueError("D0 control reuse gate mismatch")
    manifest = Path(ready["manifest"]).resolve()
    if sha256_file(manifest) != ready["manifest_sha256"]:
        raise ValueError("D0 control manifest changed")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(verify_frozen_control, rows))
    mapped = {(r["suite"], int(r["task"]), int(r["trial"])): r for r in rows}
    if len(mapped) != 640:
        raise ValueError("D0 frozen control identity count mismatch")
    return ready, mapped


def candidate_evidence(spec: tuple[dict, dict, Path]) -> list[dict[str, Any]]:
    job, payload, checkpoint = spec
    suite = job["suite"]
    validate_result_contract(
        payload, policy="h3_fastwam_online_int8", checkpoint=checkpoint,
        suite=suite, tasks=job["tasks"], trials=job["trials"],
        save_trajectories=True,
    )
    result_path = Path(job["output"]) / "results.json"
    result_hash = sha256_file(result_path)
    rows = []
    for (task, trial), episode in episode_map(payload).items():
        trajectory = Path(episode.get("trajectory", "")).resolve()
        if not trajectory.is_file():
            raise FileNotFoundError(trajectory)
        rows.append({
            "suite": suite, "task": task, "trial": trial,
            "success": bool(episode["success"]),
            "initial_object_joints": episode["initial_object_joints"],
            "initial_state_sha256": initial_state_digest(trajectory),
            "result": str(result_path.resolve()),
            "result_sha256": result_hash,
            "trajectory": str(trajectory),
            "trajectory_sha256": sha256_file(trajectory),
        })
    if len(rows) != int(job["episodes"]):
        raise ValueError(f"candidate job is incomplete: {result_path}")
    return rows


def load_candidates(root: Path, checkpoint: Path, workers: int) -> tuple[dict, dict[tuple[str, int, int], dict]]:
    prepared = json.loads((root / "PREPARED.json").read_text(encoding="utf-8"))
    manifest = root / "jobs.jsonl"
    if (
        prepared.get("permission")
        != "GO_8GPU_640_FRESH_PROCESSES_NO_INTERMEDIATE_STOP"
        or prepared.get("one_episode_per_process") is not True
        or prepared.get("candidate_checkpoint_sha256") != C58_SHA256
        or prepared.get("candidate_episodes") != 640
        or sha256_file(manifest) != prepared.get("manifest_sha256")
        or not (root / "COMPLETED.json").is_file()
    ):
        raise ValueError("expanded candidate preparation/completion mismatch")
    jobs = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    if (
        len(jobs) != 640
        or any(job.get("episodes") != 1 for job in jobs)
        or any(len(job.get("tasks", [])) != 1 for job in jobs)
        or any(len(job.get("trials", [])) != 1 for job in jobs)
    ):
        raise ValueError("expanded isolated candidate job count mismatch")
    specs = []
    for job in jobs:
        result_path = Path(job["output"]) / "results.json"
        specs.append((job, json.loads(result_path.read_text(encoding="utf-8")), checkpoint))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        groups = list(pool.map(candidate_evidence, specs))
    rows = [row for group in groups for row in group]
    mapped = {(r["suite"], int(r["task"]), int(r["trial"])): r for r in rows}
    if len(mapped) != 640:
        raise ValueError("expanded candidate identity count mismatch")
    return prepared, mapped


def same_object_joints(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        np.array_equal(np.asarray(a[name]), np.asarray(b[name])) for name in a
    )


def aggregate(
    root: Path, d0_ready_path: Path, trial33_path: Path,
    candidate_checkpoint: Path, d0_checkpoint: Path, workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve()
    candidate_checkpoint = candidate_checkpoint.resolve()
    d0_checkpoint = d0_checkpoint.resolve()
    if sha256_file(candidate_checkpoint) != C58_SHA256:
        raise ValueError("C58b checkpoint SHA256 mismatch")
    if sha256_file(d0_checkpoint) != D0_SHA256:
        raise ValueError("D0 checkpoint SHA256 mismatch")
    trial33_rows, trial33_report = load_trial33(
        trial33_path.resolve(), candidate_checkpoint, d0_checkpoint,
    )
    d0_ready, controls = load_frozen_controls(d0_ready_path.resolve(), workers)
    prepared, candidates = load_candidates(root, candidate_checkpoint, workers)
    if set(candidates) != set(controls):
        raise ValueError("expanded candidate/control pair identities differ")
    expanded = []
    evidence = []
    for key in sorted(candidates):
        candidate, control = candidates[key], controls[key]
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
    pairs = sorted(trial33_rows + expanded, key=lambda r: (r["trial"], r["suite"], r["task"]))
    if len(pairs) != 680 or len({(r["trial"], r["suite"], r["task"]) for r in pairs}) != 680:
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
    threshold = prepared["evaluation_gate"]
    gates = {
        "all_680_pairs_complete": len(pairs) == 680,
        "trial33_exact_control_bridge": (
            trial33_report["control_successes"] == 16
            and trial33_report["control_success_rate"] == 0.4
        ),
        "expanded_initial_states_exact": all(
            row["mechanical_identity"] == "full_trajectory_initial_state_exact"
            for row in expanded
        ),
        "candidate_one_episode_per_process": (
            prepared.get("one_episode_per_process") is True
            and prepared.get("jobs") == 640
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
            row["success_rate_delta"] >= threshold["no_suite_regression_below"]
            for row in per_suite.values()
        ),
    }
    passed = all(gates.values())
    report = {
        "format": "h3wam-c58b-vs-d0-expanded-paired-libero-trials33-49-v2",
        "status": "PASS_C58B_EXPANDED_PAIRED" if passed else "FAIL_C58B_EXPANDED_PAIRED",
        "permission": "GO_PROMOTE_C58B" if passed else "KEEP_D0_INCUMBENT",
        "effect_status": "EVIDENCE_READY" if passed else "NOT_EVIDENCE_READY",
        "candidate": "C58B_FASTWAM_FULL30_H3_LAYERWISE",
        "control": "D0_REPEAT_LAYER49_PARENT",
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": C58_SHA256,
        "control_checkpoint": str(d0_checkpoint),
        "control_checkpoint_sha256": D0_SHA256,
        "trial33_report": str(trial33_path.resolve()),
        "trial33_report_sha256": TRIAL33_SHA256,
        "d0_control_ready": str(d0_ready_path.resolve()),
        "d0_control_ready_sha256": sha256_file(d0_ready_path.resolve()),
        "d0_control_manifest_sha256": d0_ready["manifest_sha256"],
        "candidate_prepared_sha256": sha256_file(root / "PREPARED.json"),
        "candidate_job_manifest_sha256": prepared["manifest_sha256"],
        "pairs": 680,
        "trials": list(TRIALS),
        "overall": overall,
        "per_suite": per_suite,
        "per_trial": per_trial,
        "gates": gates,
        "pair_outcomes": pairs,
        "claim_boundary": (
            "Paired LIBERO simulator trials33..49 over all four suites and forty "
            "tasks under wait30/replan8/horizon32/eval10. Wilson intervals describe "
            "arm rates; paired_delta_normal95 uses the empirical paired outcome variance."
        ),
    }
    return report, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--d0-ready", type=Path, required=True)
    parser.add_argument("--trial33-results", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--d0-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    report, evidence = aggregate(
        args.root, args.d0_ready, args.trial33_results,
        args.candidate_checkpoint, args.d0_checkpoint, args.workers,
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
