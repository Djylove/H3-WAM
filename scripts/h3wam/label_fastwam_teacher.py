#!/usr/bin/env python3
"""Relabel saved H3 rollout states with the official FastWAM policy.

This intentionally runs in a separate process from H3: both policies fit on a
5090 individually, but not at the same time.  Input trajectories are produced
by ``rollout_libero.py --save-trajectories``.
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--libero-site-packages", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride <= 0 or args.seed <= 0:
        raise ValueError("stride and seed must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive when provided")

    repo_root = Path(__file__).resolve().parents[2]
    libero_root = args.libero_root.resolve()
    libero_site = args.libero_site_packages.resolve()
    if not (libero_root / "libero").is_dir():
        raise FileNotFoundError(f"LIBERO package not found below {libero_root}")
    if not libero_site.is_dir():
        raise FileNotFoundError(f"LIBERO site-packages not found: {libero_site}")

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

    from experiments.libero.eval_libero_single import (
        _load_model_checkpoint,
        _mixed_precision_to_model_dtype,
        _predict_action_chunk,
    )
    from fastwam.datasets.lerobot.processors.fastwam_processor import (
        FastWAMProcessor,
    )
    from fastwam.datasets.lerobot.utils.normalizer import (
        load_dataset_stats_from_json,
    )
    from fastwam.utils.pytorch_utils import set_global_seed

    trajectory_path = args.trajectory.resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    checkpoint_path = args.checkpoint.resolve()
    dataset_stats_path = args.dataset_stats.resolve()
    for path in (checkpoint_path, dataset_stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(trajectory_path) as archive:
        source = {key: archive[key] for key in archive.files}
    required = {
        "step",
        "agentview_image",
        "wristview_image",
        "eef_pos",
        "eef_quat",
        "gripper_qpos",
        "previous_action",
        "policy_actions",
    }
    missing = sorted(required - set(source))
    if missing:
        raise ValueError(f"trajectory is missing fields: {missing}")
    count = int(source["step"].shape[0])
    if any(int(source[key].shape[0]) != count for key in required):
        raise ValueError("trajectory fields have inconsistent first dimensions")
    indices = np.arange(0, count, args.stride, dtype=np.int64)
    if args.limit is not None:
        indices = indices[: args.limit]
    if indices.size == 0:
        raise ValueError("trajectory selection is empty")

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
    input_h, input_w = (int(value) for value in cfg.data.train.video_size)

    teacher_actions = []
    started = time.perf_counter()
    for position, index in enumerate(indices.tolist(), start=1):
        obs = {
            "agentview_image": source["agentview_image"][index],
            "robot0_eye_in_hand_image": source["wristview_image"][index],
            "robot0_eef_pos": source["eef_pos"][index],
            "robot0_eef_quat": source["eef_quat"][index],
            "robot0_gripper_qpos": source["gripper_qpos"][index],
        }
        actions, _, _ = _predict_action_chunk(
            obs,
            args.task,
            model,
            processor,
            cfg,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        teacher_actions.append(actions.astype(np.float32, copy=False))
        if position % 10 == 0 or position == len(indices):
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "labeled": position,
                        "total": int(len(indices)),
                        "states_per_second": round(position / elapsed, 3),
                    }
                ),
                flush=True,
            )

    selected = {key: value[indices] for key, value in source.items()}
    selected["source_index"] = indices
    selected["teacher_actions"] = np.stack(teacher_actions, axis=0)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **selected)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "states": int(len(indices)),
                "teacher_action_shape": list(selected["teacher_actions"].shape),
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated() / 2**30, 3
                ),
                "duration_seconds": round(time.perf_counter() - started, 3),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
