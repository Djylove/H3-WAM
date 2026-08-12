#!/usr/bin/env python3
"""Validate one LeRobot LIBERO episode for the H3-WAM window contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import pandas as pd

from fastwam.models.h3wam import plan_h3_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    info = json.loads((root / "meta/info.json").read_text())
    episode_name = f"episode_{args.episode:06d}"
    parquet = root / "data/chunk-000" / f"{episode_name}.parquet"
    table = pd.read_parquet(parquet)
    plan = plan_h3_window(action_horizon=args.action_horizon, source_fps=float(info["fps"]))

    cameras = {}
    for video in sorted((root / "videos").rglob(f"{episode_name}.mp4")):
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            cameras[video.parent.name] = {
                "frames": stream.frames,
                "width": stream.width,
                "height": stream.height,
                "fps": float(stream.average_rate),
            }

    if len(table) < plan.source_frame_count:
        raise RuntimeError(
            f"episode has {len(table)} frames, fewer than required {plan.source_frame_count}"
        )
    if not cameras or any(camera["frames"] != len(table) for camera in cameras.values()):
        raise RuntimeError("parquet/video frame counts do not match")

    result = {
        "dataset": str(root),
        "episode": args.episode,
        "episode_frames": len(table),
        "source_fps": info["fps"],
        "action_shape": list(table["action"].iloc[0].shape),
        "state_shape": list(table["observation.state"].iloc[0].shape),
        "cameras": cameras,
        "window": {
            "source_frames": plan.source_frame_count,
            "action_horizon": plan.action_horizon,
            "duration_seconds": plan.action_duration_seconds,
            "h3_frames": plan.h3_frame_count,
            "h3_latent_frames": plan.h3_latent_frames,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
