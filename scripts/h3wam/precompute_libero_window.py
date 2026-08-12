#!/usr/bin/env python3
"""Precompute one LIBERO window into H3 video latents for a smoke train."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import av
import pandas as pd
import torch
import torchvision.transforms.functional as tvf

from fastwam.models.h3wam import plan_h3_window, resample_video_nearest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--video-vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--camera-width", type=int, default=224)
    return parser.parse_args()


def decode_window(path: Path, *, start: int, count: int) -> torch.Tensor:
    frames = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index < start:
                continue
            if index >= start + count:
                break
            array = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
    if len(frames) != count:
        raise RuntimeError(f"decoded {len(frames)} frames from {path}, expected {count}")
    return torch.stack(frames).to(dtype=torch.float32).div_(255.0)


def load_tasks(root: Path) -> dict[int, str]:
    tasks = {}
    with (root / "meta/tasks.jsonl").open() as handle:
        for line in handle:
            item = json.loads(line)
            tasks[int(item["task_index"])] = item["task"]
    return tasks


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    comfy_root = args.comfy_root.resolve()
    vae_path = args.video_vae.resolve()
    output = args.output.resolve()
    sys.path.insert(0, str(comfy_root))

    import comfy.model_management as model_management
    import comfy.sd
    import comfy.utils

    info = json.loads((root / "meta/info.json").read_text())
    plan = plan_h3_window(
        action_horizon=args.action_horizon,
        source_fps=float(info["fps"]),
    )
    episode_name = f"episode_{args.episode:06d}"
    table = pd.read_parquet(root / "data/chunk-000" / f"{episode_name}.parquet")
    stop = args.start + plan.source_frame_count
    if args.start < 0 or stop > len(table):
        raise ValueError(f"window [{args.start}, {stop}) is outside episode length {len(table)}")

    camera_names = ["observation.images.image", "observation.images.wrist_image"]
    cameras = []
    for name in camera_names:
        path = root / "videos/chunk-000" / name / f"{episode_name}.mp4"
        frames = decode_window(path, start=args.start, count=plan.source_frame_count)
        frames = tvf.resize(
            frames,
            [args.camera_height, args.camera_width],
            antialias=True,
        )
        cameras.append(frames)
    video = torch.cat(cameras, dim=-1)
    video = resample_video_nearest(video, plan.h3_frame_count)
    video_nhwc = video.permute(0, 2, 3, 1).contiguous()

    load_started = time.perf_counter()
    vae_state = comfy.utils.load_torch_file(str(vae_path))
    vae = comfy.sd.VAE(sd=vae_state)
    del vae_state
    load_seconds = time.perf_counter() - load_started

    encode_started = time.perf_counter()
    with torch.inference_mode():
        video_latents = vae.encode(video_nhwc)
        first_frame_latents = vae.encode(video_nhwc[:1])
    if model_management.get_torch_device().type == "cuda":
        torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - encode_started

    expected_video_shape = (1, 24, plan.h3_latent_frames, 14, 28)
    expected_first_shape = (1, 24, 1, 14, 28)
    if tuple(video_latents.shape) != expected_video_shape:
        raise RuntimeError(
            f"unexpected video latent shape {tuple(video_latents.shape)}, expected {expected_video_shape}"
        )
    if tuple(first_frame_latents.shape) != expected_first_shape:
        raise RuntimeError(
            "unexpected first-frame latent shape "
            f"{tuple(first_frame_latents.shape)}, expected {expected_first_shape}"
        )

    action_slice = table["action"].iloc[args.start : args.start + args.action_horizon]
    actions = torch.stack([torch.from_numpy(value.copy()) for value in action_slice]).to(dtype=torch.float32)
    state = torch.from_numpy(table["observation.state"].iloc[args.start].copy()).to(dtype=torch.float32)
    task_index = int(table["task_index"].iloc[args.start])
    artifact = {
        "video_latents": video_latents.cpu(),
        "first_frame_latents": first_frame_latents.cpu(),
        # Clone so torch.save does not preserve the full 39-frame backing storage.
        "first_frame_pixels": video_nhwc[:1].clone().cpu(),
        "actions": actions,
        "state": state,
        "task": load_tasks(root)[task_index],
        "task_index": task_index,
        "episode": args.episode,
        "start": args.start,
        "source_fps": float(info["fps"]),
        "h3_frame_count": plan.h3_frame_count,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "video_latents": list(video_latents.shape),
                "first_frame_latents": list(first_frame_latents.shape),
                "actions": list(actions.shape),
                "state": list(state.shape),
                "task": artifact["task"],
                "vae_load_seconds": load_seconds,
                "vae_encode_seconds": encode_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
