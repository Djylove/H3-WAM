#!/usr/bin/env python3
"""Replay one LeRobot LIBERO demonstration to validate action conventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from experiments.libero.libero_utils import get_libero_env
from libero.libero import benchmark
from scripts.h3wam.rollout_libero import object_joint_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trial-index", type=int, default=25)
    parser.add_argument("--episode-index", type=int, default=25)
    parser.add_argument("--wait-steps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    initial_state = suite.get_task_init_states(args.task_id)[args.trial_index]
    parquet = (
        args.dataset_root
        / "data/chunk-000"
        / f"episode_{args.episode_index:06d}.parquet"
    )
    actions = np.asarray(
        pq.read_table(parquet, columns=["action"])["action"].to_pylist(),
        dtype=np.float32,
    )
    results = []
    for convention in ("raw", "dataset_to_libero"):
        env, description = get_libero_env(task, args.resolution, args.seed)
        try:
            env.reset()
            observation = env.set_init_state(initial_state)
            done = False
            for _ in range(args.wait_steps):
                observation, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            initial = {
                name: np.asarray(value, dtype=np.float64)
                for name, value in object_joint_state(env).items()
            }
            maximum_delta = {name: 0.0 for name in initial}
            steps = 0
            for source in actions:
                action = source.copy()
                if convention == "dataset_to_libero":
                    action[-1] = -(2.0 * action[-1] - 1.0)
                observation, _, done, _ = env.step(action)
                steps += 1
                current = object_joint_state(env)
                for name, value in current.items():
                    delta = np.max(
                        np.abs(np.asarray(value, dtype=np.float64) - initial[name])
                    )
                    maximum_delta[name] = max(maximum_delta[name], float(delta))
                if done:
                    break
            results.append(
                {
                    "convention": convention,
                    "success": bool(done),
                    "steps": steps,
                    "middle_drawer_delta": maximum_delta.get(
                        "wooden_cabinet_1_middle_level"
                    ),
                    "final_eef": np.asarray(
                        observation["robot0_eef_pos"], dtype=np.float64
                    ).tolist(),
                    "final_gripper": np.asarray(
                        observation["robot0_gripper_qpos"], dtype=np.float64
                    ).tolist(),
                }
            )
        finally:
            env.close()
    payload = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task": description,
        "trial_index": args.trial_index,
        "episode_index": args.episode_index,
        "wait_steps": args.wait_steps,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
