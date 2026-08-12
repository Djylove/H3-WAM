#!/usr/bin/env python3
"""Resume a saved H3 simulator state and let official FastWAM roll in.

H3 and official FastWAM do not fit together on a 32 GiB GPU.  This script is
the second half of a two-process intervention pipeline: ``rollout_libero.py
--save-trajectories`` stores full MuJoCo states, then this process restores one
state and records a continuous, on-policy FastWAM recovery trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
import site
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--intervention-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--libero-site-packages", type=Path, required=True)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--trial-index", type=int, required=True)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.intervention_index < 0:
        raise ValueError("intervention-index cannot be negative")
    if args.replan_steps <= 0 or args.max_steps <= 0 or args.resolution <= 0:
        raise ValueError("replan-steps, max-steps and resolution must be positive")

    repo_root = Path(__file__).resolve().parents[2]
    libero_root = args.libero_root.resolve()
    libero_site = args.libero_site_packages.resolve()
    sys.path.insert(0, str(repo_root / "experiments" / "libero"))
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(libero_root))
    site.addsitedir(str(libero_site))
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(args.model_base_path.resolve())

    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from libero.libero import benchmark

    from experiments.libero.eval_libero_single import (
        _load_model_checkpoint,
        _mixed_precision_to_model_dtype,
        _predict_action_chunk,
    )
    from experiments.libero.libero_utils import get_libero_env
    from fastwam.datasets.lerobot.processors.fastwam_processor import (
        FastWAMProcessor,
    )
    from fastwam.datasets.lerobot.utils.normalizer import (
        load_dataset_stats_from_json,
    )
    from fastwam.utils.pytorch_utils import set_global_seed

    trajectory_path = args.trajectory.resolve()
    with np.load(trajectory_path) as archive:
        source = {key: archive[key] for key in archive.files}
    required = {
        "step",
        "sim_state",
        "previous_action",
    }
    missing = sorted(required - set(source))
    if missing:
        raise ValueError(
            f"trajectory is missing {missing}; recapture it with the current "
            "rollout_libero.py --save-trajectories"
        )
    state_count = int(source["step"].shape[0])
    if args.intervention_index >= state_count:
        raise ValueError(
            f"intervention index {args.intervention_index} exceeds {state_count} states"
        )

    checkpoint_path = args.checkpoint.resolve()
    dataset_stats_path = args.dataset_stats.resolve()
    for path in (checkpoint_path, dataset_stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    set_global_seed(args.seed, get_worker_init_fn=False)
    overrides = [
        "task=libero_uncond_2cam224_1e-4",
        f"ckpt={checkpoint_path}",
        f"EVALUATION.dataset_stats_path={dataset_stats_path}",
        "EVALUATION.output_dir=.",
        f"seed={args.seed}",
    ]
    with initialize_config_dir(
        version_base="1.3", config_dir=str(repo_root / "configs")
    ):
        cfg = compose(config_name="sim_libero.yaml", overrides=overrides)

    model_device = str(cfg.EVALUATION.get("device", "cuda"))
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(checkpoint_path))
    model = model.to(model_device).eval()
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    action_horizon = int(cfg.data.train.num_frames) - 1
    if args.replan_steps > action_horizon:
        raise ValueError(
            f"replan steps {args.replan_steps} exceed teacher horizon {action_horizon}"
        )
    input_h, input_w = (int(value) for value in cfg.data.train.video_size)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_id < suite.get_num_tasks():
        raise ValueError(f"task-id {args.task_id} is outside suite {args.suite}")
    task = suite.get_task(args.task_id)
    task_description = task.language
    env, _ = get_libero_env(task, args.resolution, args.seed)
    records: dict[str, list] = {
        "step": [],
        "agentview_image": [],
        "wristview_image": [],
        "eef_pos": [],
        "eef_quat": [],
        "gripper_qpos": [],
        "previous_action": [],
        "sim_state": [],
        "teacher_actions": [],
    }
    start_step = int(source["step"][args.intervention_index])
    step = start_step
    replans = 0
    previous_action = np.asarray(
        source["previous_action"][args.intervention_index], dtype=np.float32
    )
    done = False
    started = time.perf_counter()
    try:
        env.reset()
        obs = env.regenerate_obs_from_state(
            np.asarray(source["sim_state"][args.intervention_index], dtype=np.float64)
        )
        done = bool(env.check_success())
        while step < args.max_steps and not done:
            records["step"].append(step)
            records["agentview_image"].append(
                np.ascontiguousarray(obs["agentview_image"], dtype=np.uint8)
            )
            records["wristview_image"].append(
                np.ascontiguousarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)
            )
            records["eef_pos"].append(
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
            )
            records["eef_quat"].append(
                np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
            )
            records["gripper_qpos"].append(
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
            )
            records["previous_action"].append(previous_action.copy())
            records["sim_state"].append(
                np.asarray(env.get_sim_state(), dtype=np.float64)
            )
            actions, _, _ = _predict_action_chunk(
                obs,
                task_description,
                model,
                processor,
                cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            records["teacher_actions"].append(
                actions.astype(np.float32, copy=False)
            )
            replans += 1
            for action in actions[: args.replan_steps]:
                obs, _, done, _ = env.step(action)
                previous_action = np.asarray(action, dtype=np.float32)
                step += 1
                if done or step >= args.max_steps:
                    break
    finally:
        env.close()

    if not records["step"]:
        raise RuntimeError("intervention produced no teacher states")
    packed = {key: np.asarray(values) for key, values in records.items()}
    packed["intervention_index"] = np.asarray(args.intervention_index)
    packed["trial_index"] = np.asarray(args.trial_index)
    packed["task_id"] = np.asarray(args.task_id)
    packed["success"] = np.asarray(done)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **packed)
    temporary.replace(output)
    report = {
        "output": str(output),
        "source_trajectory": str(trajectory_path),
        "suite": args.suite,
        "task_id": args.task_id,
        "trial_index": args.trial_index,
        "intervention_index": args.intervention_index,
        "start_step": start_step,
        "final_step": step,
        "replans": replans,
        "teacher_action_shape": list(packed["teacher_actions"].shape),
        "success": bool(done),
        "duration_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
