#!/usr/bin/env python3
"""Test deterministic paired branches from one canonical LIBERO saved state."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from scripts.h3wam.audit_libero_trajectory_restore import comparison


REQUIRED = {
    "step", "sim_state", "policy_actions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def observation_comparison(first: dict, second: dict) -> dict:
    return {
        "agentview_image": comparison(
            first["agentview_image"], second["agentview_image"]
        ),
        "wristview_image": comparison(
            first["robot0_eye_in_hand_image"],
            second["robot0_eye_in_hand_image"],
        ),
        "eef_pos": comparison(first["robot0_eef_pos"], second["robot0_eef_pos"]),
        "eef_quat": comparison(
            first["robot0_eef_quat"], second["robot0_eef_quat"]
        ),
        "gripper_qpos": comparison(
            first["robot0_gripper_qpos"], second["robot0_gripper_qpos"]
        ),
    }


def execute(env, state: np.ndarray, actions: np.ndarray, seed: int) -> dict:
    # LIBERO's seed mutates process-global RNG state. Re-seed immediately before
    # every reset so two environments receive the same randomized world layout.
    env.seed(seed)
    env.reset()
    start = env.regenerate_obs_from_state(state)
    obs = start
    done = bool(env.check_success())
    steps = 0
    for action in actions:
        if done:
            break
        obs, _, done, _ = env.step(action)
        steps += 1
    return {
        "start": start,
        "end": obs,
        "state": np.asarray(env.get_sim_state(), dtype=np.float64),
        "success": bool(done),
        "steps": steps,
    }


def main() -> None:
    args = parse_args()
    if args.execution_horizon <= 0:
        raise ValueError("execution-horizon must be positive")
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "experiments" / "libero"))
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))
    from libero.libero import benchmark
    from experiments.libero.libero_utils import get_libero_env

    with np.load(args.trajectory.resolve(), allow_pickle=False) as archive:
        missing = REQUIRED - set(archive.files)
        if missing:
            raise ValueError(f"trajectory misses branch fields: {sorted(missing)}")
        source = {key: archive[key] for key in REQUIRED}
    count = int(source["step"].shape[0])
    if source["sim_state"].shape[0] != count or source["policy_actions"].shape[0] != count:
        raise ValueError("trajectory branch fields have inconsistent row counts")
    indices = sorted({0, count // 2, count - 1})
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    first_env, _ = get_libero_env(task, args.resolution, args.seed)
    second_env, _ = get_libero_env(task, args.resolution, args.seed)
    rows = []
    try:
        for index in indices:
            state = np.asarray(source["sim_state"][index], dtype=np.float64)
            actions = np.asarray(
                source["policy_actions"][index, : args.execution_horizon],
                dtype=np.float32,
            )
            first = execute(first_env, state, actions, args.seed)
            second = execute(second_env, state, actions, args.seed)
            rows.append(
                {
                    "trajectory_index": index,
                    "environment_step": int(source["step"][index]),
                    "actions": comparison(actions, actions.copy()),
                    "start_observation": observation_comparison(
                        first["start"], second["start"]
                    ),
                    "end_observation": observation_comparison(
                        first["end"], second["end"]
                    ),
                    "end_state": comparison(first["state"], second["state"]),
                    "steps_equal": first["steps"] == second["steps"],
                    "success_equal": first["success"] == second["success"],
                }
            )
    finally:
        first_env.close()
        second_env.close()

    image_exact = all(
        row[phase][name]["exact_fraction"] == 1.0
        for row in rows
        for phase in ("start_observation", "end_observation")
        for name in ("agentview_image", "wristview_image")
    )
    numeric_close = all(
        row[phase][name]["max_abs"] <= 1e-10
        for row in rows
        for phase in ("start_observation", "end_observation")
        for name in ("eef_pos", "eef_quat", "gripper_qpos")
    ) and all(
        row["end_state"]["max_abs"] <= 1e-10
        and row["steps_equal"] and row["success_equal"]
        for row in rows
    )
    exact = image_exact and numeric_close
    report = {
        "format": "h3wam-libero-paired-branch-repeatability-v1",
        "trajectory": str(args.trajectory.resolve()),
        "suite": args.suite,
        "task_id": args.task_id,
        "indices": indices,
        "execution_horizon": args.execution_horizon,
        "numeric_tolerance": 1e-10,
        "image_contract": "pixel exact",
        "original_observation_fidelity": "NOT_CLAIMED_C19_FAILED",
        "status": (
            "PASS_PAIRED_BRANCH_REPEATABILITY_GATE"
            if exact else "FAIL_PAIRED_BRANCH_REPEATABILITY_GATE"
        ),
        "rows": rows,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
