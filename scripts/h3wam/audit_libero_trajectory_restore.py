#!/usr/bin/env python3
"""Verify that saved LIBERO MuJoCo states reproduce policy observations exactly."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REQUIRED = {
    "step", "sim_state", "agentview_image", "wristview_image",
    "eef_pos", "eef_quat", "gripper_qpos",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def comparison(saved: np.ndarray, restored: np.ndarray) -> dict:
    saved = np.asarray(saved)
    restored = np.asarray(restored)
    if saved.shape != restored.shape:
        raise ValueError(f"restore shape mismatch: {saved.shape} != {restored.shape}")
    difference = np.abs(saved.astype(np.float64) - restored.astype(np.float64))
    return {
        "shape": list(saved.shape),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "exact_fraction": float(np.equal(saved, restored).mean()) if saved.size else 1.0,
    }


def main() -> None:
    args = parse_args()
    if args.task_id < 0 or args.seed < 0 or args.resolution <= 0:
        raise ValueError("task-id/seed must be non-negative and resolution positive")
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "experiments" / "libero"))
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

    from libero.libero import benchmark
    from experiments.libero.libero_utils import get_libero_env

    trajectory_path = args.trajectory.resolve()
    with np.load(trajectory_path, allow_pickle=False) as archive:
        missing = REQUIRED - set(archive.files)
        if missing:
            raise ValueError(f"trajectory misses restore fields: {sorted(missing)}")
        source = {key: archive[key] for key in REQUIRED}
    count = int(source["step"].shape[0])
    if count <= 0 or any(source[key].shape[0] != count for key in REQUIRED):
        raise ValueError("trajectory restore fields have inconsistent row counts")
    indices = sorted({0, count // 2, count - 1})

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_id < suite.get_num_tasks():
        raise ValueError("task-id is outside the selected LIBERO suite")
    env, _ = get_libero_env(suite.get_task(args.task_id), args.resolution, args.seed)
    rows = []
    try:
        env.reset()
        for index in indices:
            restored = env.regenerate_obs_from_state(
                np.asarray(source["sim_state"][index], dtype=np.float64)
            )
            row = {
                "trajectory_index": index,
                "environment_step": int(source["step"][index]),
                "sim_state": comparison(
                    source["sim_state"][index], env.get_sim_state()
                ),
                "agentview_image": comparison(
                    source["agentview_image"][index], restored["agentview_image"]
                ),
                "wristview_image": comparison(
                    source["wristview_image"][index],
                    restored["robot0_eye_in_hand_image"],
                ),
                "eef_pos": comparison(
                    source["eef_pos"][index], restored["robot0_eef_pos"]
                ),
                "eef_quat": comparison(
                    source["eef_quat"][index], restored["robot0_eef_quat"]
                ),
                "gripper_qpos": comparison(
                    source["gripper_qpos"][index], restored["robot0_gripper_qpos"]
                ),
                "success_predicate": bool(env.check_success()),
            }
            rows.append(row)
    finally:
        env.close()

    state_pass = all(row["sim_state"]["max_abs"] <= 1e-12 for row in rows)
    proprio_pass = all(
        row[name]["max_abs"] <= 1e-6
        for row in rows
        for name in ("eef_pos", "eef_quat", "gripper_qpos")
    )
    image_pass = all(
        row[name]["exact_fraction"] == 1.0
        for row in rows
        for name in ("agentview_image", "wristview_image")
    )
    report = {
        "format": "h3wam-libero-trajectory-restore-audit-v1",
        "trajectory": str(trajectory_path),
        "suite": args.suite,
        "task_id": args.task_id,
        "seed": args.seed,
        "indices": indices,
        "state_pass": state_pass,
        "proprio_pass": proprio_pass,
        "image_exact_pass": image_pass,
        "status": (
            "PASS_EXACT_RESTORE_GATE"
            if state_pass and proprio_pass and image_pass
            else "FAIL_EXACT_RESTORE_GATE"
        ),
        "rows": rows,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
