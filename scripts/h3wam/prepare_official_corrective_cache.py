#!/usr/bin/env python3
"""Convert successful FastWAM roll-ins into official-H3 corrective windows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from precompute_libero_official_h3 import PIXEL_MEAN, PIXEL_STD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--episode-id", type=int, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--phase-length", type=int, default=151)
    parser.add_argument("--sample-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def save_atomic(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if len(args.input) != len(args.episode_id):
        raise ValueError("provide one --episode-id for each --input")
    if not 1 <= args.action_horizon <= 32:
        raise ValueError("action-horizon must be in [1,32]")
    if args.phase_length <= 0 or args.sample_weight <= 0:
        raise ValueError("phase-length and sample-weight must be positive")

    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from fastwam.models.h3wam import libero_observation_state, preprocess_libero_cameras

    device = torch.device(args.device)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.model.resolve(),
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    rows: list[dict] = []
    cache_root = args.cache_root.resolve()
    for source_path, episode_id in zip(args.input, args.episode_id):
        with np.load(source_path.resolve()) as archive:
            source = {key: archive[key] for key in archive.files}
        required = {
            "step", "agentview_image", "wristview_image", "eef_pos",
            "eef_quat", "gripper_qpos", "teacher_actions",
        }
        missing = sorted(required - set(source))
        if missing:
            raise ValueError(f"{source_path} misses {missing}")
        if not bool(source.get("success", False)):
            raise ValueError(f"refusing unsuccessful teacher roll-in {source_path}")
        for index, raw_step in enumerate(source["step"]):
            step = int(raw_step)
            item_id = f"recovery_t5_ep{episode_id:06d}_s{step:06d}"
            pixels = preprocess_libero_cameras(
                source["agentview_image"][index], source["wristview_image"][index]
            )
            video = (
                pixels.mul(255.0).round().to(torch.uint8)
                .permute(0, 3, 1, 2).unsqueeze(2).to(device)
            )
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16
            ):
                first = encode_vae_condition(
                    vae, video, PIXEL_MEAN, PIXEL_STD
                ).float().cpu()
            actions = torch.from_numpy(
                source["teacher_actions"][index, : args.action_horizon]
                .astype(np.float32, copy=True)
            )
            # FastWAM outputs LIBERO {-1=open,+1=close}; training windows use
            # {1=open,0=close} before environment conversion.
            actions[:, -1] = (1.0 - actions[:, -1]) * 0.5
            observation = {
                "eef_pos": source["eef_pos"][index],
                "eef_quat": source["eef_quat"][index],
                "gripper_qpos": source["gripper_qpos"][index],
            }
            save_atomic(
                {
                    "first_frame_latents": first,
                    "actions": actions,
                    "state": libero_observation_state(observation),
                    "task": args.task,
                    "episode": episode_id,
                    "start": step,
                    "length": args.phase_length,
                    "h3_frame_count": 39,
                    "vae_backend": "diffusers.AutoencoderKLMiniMaxH3",
                    "corrective": True,
                    "teacher": "official_fastwam_continuous_rollin",
                },
                cache_root / "windows" / f"{item_id}.pt",
            )
            rows.append(
                {
                    "id": item_id,
                    "episode": episode_id,
                    "start": step,
                    "length": args.phase_length,
                    "task": args.task,
                    "task_group": 0,
                    "sample_weight": args.sample_weight,
                    "corrective": True,
                }
            )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    print(json.dumps({"manifest": str(args.manifest.resolve()), "windows": len(rows)}))


if __name__ == "__main__":
    main()
