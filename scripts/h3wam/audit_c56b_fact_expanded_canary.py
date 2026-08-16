#!/usr/bin/env python3
"""Audit mechanics of the C60 expanded canary without exposing its outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from prepare_c56b_fact_expanded_eval import C60_SHA256


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


def initial_state_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path) as archive:
        for name in INITIAL_KEYS:
            if name not in archive or len(archive[name]) == 0:
                raise ValueError(f"trajectory initial state contract mismatch: {path}")
            value = np.ascontiguousarray(archive[name][0])
            digest.update(name.encode())
            digest.update(value.dtype.str.encode())
            digest.update(json.dumps(value.shape).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def validate_result_contract(payload: dict, checkpoint: Path) -> None:
    if (
        payload.get("policy") != "h3_fact_online_int8"
        or Path(payload.get("checkpoint", "")).resolve() != checkpoint
        or payload.get("suite") != "libero_spatial"
        or payload.get("task_ids") != [0]
        or payload.get("trial_indices") != [34]
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
        raise ValueError("mechanical canary rollout contract mismatch")


def validate_episode(payload: dict) -> dict:
    tasks = payload.get("tasks", [])
    episodes = tasks[0].get("episodes", []) if len(tasks) == 1 else []
    if len(episodes) != 1 or tasks[0].get("task_id") != 0:
        raise ValueError("canary episode identity mismatch")
    episode = episodes[0]
    expected_seed = 34_042
    replans = int(episode.get("replans", -1))
    if (
        episode.get("trial") != 34
        or episode.get("episode_seed") != expected_seed
        or episode.get("environment_seed") is not None
        or replans <= 0 or replans > 50
        or episode.get("replan_noise_seeds")
        != list(range(expected_seed, expected_seed + replans))
    ):
        raise ValueError("canary seed/replan contract mismatch")
    for name, shape in (
        ("first_environment_action", (7,)),
        ("first_environment_action_chunk", (32, 7)),
        ("replan_first_actions", (replans, 7)),
    ):
        value = np.asarray(episode.get(name), dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"canary action payload mismatch: {name}")
    return episode


def audit(root: Path) -> dict:
    root = root.resolve()
    prepared_path = root / "PREPARED.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if (
        prepared.get("format") != "h3wam-c56b-fact-expanded-prepared-v1"
        or prepared.get("candidate_checkpoint_sha256") != C60_SHA256
        or prepared.get("jobs") != 640
    ):
        raise ValueError("C60 preparation mismatch")
    result_path = root / "mechanical-canary/results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = Path(prepared["candidate_checkpoint"]).resolve()
    validate_result_contract(payload, checkpoint)
    episode = validate_episode(payload)
    trajectory = Path(episode.get("trajectory", "")).resolve()
    initial_digest = initial_state_digest(trajectory)
    policy_log = root / "mechanical-canary/policy_server.log"
    text = policy_log.read_text(encoding="utf-8")
    if '"stage": "ready"' not in text or '"policy": "h3_fact_online_int8"' not in text:
        raise ValueError("real policy startup/restore did not reach ready")
    return {
        "format": "h3wam-c56b-fact-expanded-mechanical-canary-v1",
        "status": "PASS_REAL_POLICY_STARTUP_RESTORE_AND_ONE_EPISODE",
        "permission": "GO_8GPU_640_FRESH_PROCESSES_NO_INTERMEDIATE_STOP",
        "checkpoint_sha256": C60_SHA256,
        "prepared_sha256": sha256_file(prepared_path),
        "result_sha256": sha256_file(result_path),
        "trajectory_sha256": sha256_file(trajectory),
        "initial_state_sha256": initial_digest,
        "policy_log_sha256": sha256_file(policy_log),
        "success_redacted": True,
        "claim_boundary": "Mechanical infrastructure canary only; outcome was not used.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
