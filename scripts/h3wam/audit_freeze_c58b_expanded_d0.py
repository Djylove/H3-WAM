#!/usr/bin/env python3
"""Freeze the historical D0 control for C58b trials 34..49."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TRIALS = tuple(range(34, 50))
D0_SHA256 = "36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"
C55_FINAL_SHA256 = "72a840eba03e0f79ff3a8568153adc9c5fe72165f4c8f2bec932aba390e4c799"
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


def source_root(mechanical: Path, fresh: Path, trial: int) -> Path:
    return mechanical if trial <= 36 else fresh


def audit_one(spec: tuple[Path, str, int, int, Path]) -> dict[str, Any]:
    root, suite, task, trial, checkpoint = spec
    directory = root / "runs" / "d0_parent" / suite / f"task{task:02d}_trial{trial:02d}"
    result_path = directory / "results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        payload.get("policy") != "h3_dreamwam_kv_int8"
        or Path(payload.get("checkpoint", "")).resolve() != checkpoint
        or payload.get("suite") != suite
        or payload.get("task_ids") != [task]
        or payload.get("trial_indices") != [trial]
        or payload.get("trials_per_task") != 1
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
        or payload.get("save_trajectories") is not True
        or payload.get("binarize_gripper") is not True
        or payload.get("context_mode") != "cached"
    ):
        raise ValueError(f"historical D0 rollout contract mismatch: {result_path}")
    tasks = payload.get("tasks", [])
    if len(tasks) != 1 or tasks[0].get("task_id") != task:
        raise ValueError(f"historical D0 task payload mismatch: {result_path}")
    episodes = tasks[0].get("episodes", [])
    if len(episodes) != 1:
        raise ValueError(f"historical D0 episode count mismatch: {result_path}")
    episode = episodes[0]
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
        raise ValueError(f"historical D0 seed contract mismatch: {result_path}")
    for name, shape in (
        ("first_environment_action", (7,)),
        ("first_environment_action_chunk", (32, 7)),
        ("replan_first_actions", (replans, 7)),
    ):
        value = np.asarray(episode.get(name), dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"historical D0 action payload mismatch: {result_path}/{name}")
    object_joints = episode.get("initial_object_joints")
    if not isinstance(object_joints, dict) or not object_joints:
        raise ValueError(f"historical D0 initial joints missing: {result_path}")
    if not all(np.isfinite(np.asarray(value, dtype=np.float64)).all() for value in object_joints.values()):
        raise ValueError(f"historical D0 initial joints non-finite: {result_path}")
    trajectory = Path(episode.get("trajectory", "")).resolve()
    if not trajectory.is_file():
        raise FileNotFoundError(trajectory)
    with np.load(trajectory) as archive:
        if any(name not in archive or len(archive[name]) == 0 for name in INITIAL_KEYS):
            raise ValueError(f"historical D0 trajectory contract mismatch: {trajectory}")
        initial = {name: np.array(archive[name][0], copy=True) for name in INITIAL_KEYS}
    return {
        "suite": suite,
        "task": task,
        "trial": trial,
        "success": bool(episode.get("success")),
        "steps": int(episode.get("steps", -1)),
        "replans": replans,
        "episode_seed": expected_seed,
        "initial_object_joints": object_joints,
        "initial_state_sha256": tensor_digest(initial),
        "result": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "trajectory": str(trajectory),
        "trajectory_sha256": sha256_file(trajectory),
    }


def freeze(
    mechanical_root: Path,
    fresh_root: Path,
    c55_final: Path,
    checkpoint: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    mechanical_root = mechanical_root.resolve()
    fresh_root = fresh_root.resolve()
    c55_final = c55_final.resolve()
    checkpoint = checkpoint.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if sha256_file(checkpoint) != D0_SHA256:
        raise ValueError("historical D0 checkpoint SHA256 mismatch")
    if sha256_file(c55_final) != C55_FINAL_SHA256:
        raise ValueError("C55 FINAL SHA256 mismatch")
    final = json.loads(c55_final.read_text(encoding="utf-8"))
    expected_outcomes = {
        (int(row["trial"]), str(row["suite"]), int(row["task"])): bool(row["d0_parent"])
        for row in final.get("pairs", [])
        if int(row["trial"]) in TRIALS
    }
    if len(expected_outcomes) != 640:
        raise ValueError("C55 FINAL does not contain the exact 640 D0 control pairs")
    specs = [
        (source_root(mechanical_root, fresh_root, trial), suite, task, trial, checkpoint)
        for trial in TRIALS for suite in SUITES for task in range(10)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(audit_one, specs))
    rows.sort(key=lambda row: (row["trial"], row["suite"], row["task"]))
    if len(rows) != 640 or len({(r["trial"], r["suite"], r["task"]) for r in rows}) != 640:
        raise ValueError("historical D0 frozen row identity mismatch")
    for row in rows:
        key = (row["trial"], row["suite"], row["task"])
        if row["success"] != expected_outcomes[key]:
            raise ValueError(f"historical D0 outcome differs from C55 FINAL: {key}")
    output_dir.mkdir(parents=True)
    manifest = output_dir / "controls.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    by_suite = {
        suite: sum(row["success"] for row in rows if row["suite"] == suite)
        for suite in SUITES
    }
    report = {
        "format": "h3wam-c58b-expanded-d0-control-freeze-v1",
        "status": "PASS_D0_CONTROL_REUSE",
        "permission": "GO_C58B_CANDIDATE_ONLY_TRIALS34_49",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": D0_SHA256,
        "c55_final": str(c55_final),
        "c55_final_sha256": C55_FINAL_SHA256,
        "trials": list(TRIALS),
        "suites": list(SUITES),
        "tasks_per_suite": 10,
        "controls": 640,
        "successes": sum(row["success"] for row in rows),
        "successes_by_suite": by_suite,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "contract": {
            "wait_steps": 30,
            "replan_steps": 8,
            "action_horizon": 32,
            "model_evaluations": 10,
            "seed": 42,
            "episode_seed": "42+task_id*100000+trial_index*1000",
            "normalized_action_pre_clamp": True,
            "save_trajectories": True,
        },
        "reuse_rationale": (
            "The same D0 checkpoint reproduced all 40 trial33 outcomes and ten "
            "mechanical fields exactly in the corrected C58b harness; every reused "
            "trial34..49 result and trajectory is independently content-hashed."
        ),
        "fallback": "Any audit failure requires paired D0 rerun; candidate-only launch is forbidden.",
    }
    temporary = output_dir / f".READY.json.{os.getpid()}.partial"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "READY.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanical-root", type=Path, required=True)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--c55-final", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    report = freeze(
        args.mechanical_root, args.fresh_root, args.c55_final,
        args.checkpoint, args.output_dir, args.workers,
    )
    print(json.dumps({
        "status": report["status"], "permission": report["permission"],
        "controls": report["controls"], "successes": report["successes"],
        "manifest_sha256": report["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
