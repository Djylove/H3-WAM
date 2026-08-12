#!/usr/bin/env python3
"""Precompute DreamWAM-style RAFT motion latents with the MiniMax-H3 VAE."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "h3wam"))

from precompute_libero_official_h3 import (  # noqa: E402
    CAMERAS,
    PIXEL_MEAN,
    PIXEL_STD,
    decode_episode,
    resize_uint8,
)
from fastwam.models.h3wam import plan_h3_window, resample_video_nearest  # noqa: E402


class _RAFTArgs(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def load_raft(
    source_root: Path,
    checkpoint: Path,
    *,
    device: torch.device,
) -> torch.nn.Module:
    core = source_root / "core"
    if not (core / "raft.py").is_file():
        raise FileNotFoundError(f"missing RAFT source: {core / 'raft.py'}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing RAFT checkpoint: {checkpoint}")
    sys.path.insert(0, str(core))
    try:
        from raft import RAFT
    finally:
        sys.path.pop(0)
    args = _RAFTArgs(small=False, mixed_precision=False, alternate_corr=False)
    wrapped = torch.nn.DataParallel(RAFT(args))
    wrapped.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    return wrapped.module.to(device).eval()


class InputPadder:
    def __init__(self, shape: torch.Size):
        height, width = shape[-2:]
        pad_height = (((height // 8) + 1) * 8 - height) % 8
        pad_width = (((width // 8) + 1) * 8 - width) % 8
        self.padding = [
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
        ]

    def pad(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        return [F.pad(value, self.padding, mode="replicate") for value in inputs]

    def unpad(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        return value[
            ...,
            self.padding[2] : height - self.padding[3],
            self.padding[0] : width - self.padding[1],
        ]


def _flow_colorwheel() -> np.ndarray:
    segments = (15, 6, 4, 11, 13, 6)
    wheel = np.zeros((sum(segments), 3))
    column = 0
    transitions = ((0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2))
    for length, (fixed, changing) in zip(segments, transitions, strict=True):
        wheel[column : column + length, fixed] = 255
        values = np.floor(255 * np.arange(length) / length)
        if (fixed, changing) in {(1, 0), (2, 1), (0, 2)}:
            values = 255 - values
        wheel[column : column + length, changing] = values
        column += length
    return wheel


def flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError("flow must be [N,H,W,2]")
    wheel = _flow_colorwheel()
    columns = wheel.shape[0]
    images = []
    for field in flow.detach().float().cpu().numpy():
        horizontal, vertical = field[..., 0], field[..., 1]
        radius = np.sqrt(horizontal**2 + vertical**2)
        maximum = np.max(radius)
        horizontal /= maximum + 1.0e-5
        vertical /= maximum + 1.0e-5
        radius = np.sqrt(horizontal**2 + vertical**2)
        position = (np.arctan2(-vertical, -horizontal) / np.pi + 1.0) * 0.5
        position *= columns - 1
        lower = np.floor(position).astype(np.int32)
        upper = (lower + 1) % columns
        fraction = position - lower
        image = np.zeros((*horizontal.shape, 3), dtype=np.uint8)
        for channel in range(3):
            color = (1.0 - fraction) * (wheel[lower, channel] / 255.0)
            color += fraction * (wheel[upper, channel] / 255.0)
            within = radius <= 1
            color[within] = 1.0 - radius[within] * (1.0 - color[within])
            color[~within] *= 0.75
            image[..., channel] = np.floor(255.0 * color)
        images.append(torch.from_numpy(image))
    return torch.stack(images).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raft-source", type=Path, required=True)
    parser.add_argument("--raft-checkpoint", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--raft-iterations", type=int, default=20)
    parser.add_argument("--raft-batch-size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task", help="Optional exact task-string manifest filter.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def read_manifest(path: Path, limit: int | None) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows if limit is None else rows[:limit]


@torch.inference_mode()
def raft_motion_video(
    raft: torch.nn.Module,
    video: torch.Tensor,
    *,
    device: torch.device,
    iterations: int,
    batch_size: int,
) -> torch.Tensor:
    """Return the paper's color-coded flow video as uint8 ``[T,3,H,W]``."""

    if video.ndim != 4 or video.shape[1] != 3 or video.shape[0] < 2:
        raise ValueError("video must be uint8 [T,3,H,W] with T >= 2")
    first = video[:-1].float()
    second = video[1:].float()
    flow_fields = []
    for start in range(0, first.shape[0], batch_size):
        image1 = first[start : start + batch_size].to(device)
        image2 = second[start : start + batch_size].to(device)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        _, flow = raft(image1, image2, iters=iterations, test_mode=True)
        flow_fields.append(padder.unpad(flow).cpu().permute(0, 2, 3, 1))
    flow_rgb = flow_to_rgb(torch.cat(flow_fields)).permute(0, 3, 1, 2)
    return torch.cat((flow_rgb[:1].clone(), flow_rgb), dim=0).contiguous()


def progress(complete: int, total: int, started: float, rank: int) -> None:
    elapsed = time.perf_counter() - started
    per_window = elapsed / max(complete, 1)
    print(
        json.dumps(
            {
                "event": "h3_motion_progress",
                "rank": rank,
                "complete": complete,
                "total": total,
                "seconds": round(elapsed, 2),
                "seconds_per_window": round(per_window, 3),
                "eta_seconds": round(per_window * (total - complete), 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world-size)")
    if min(args.action_horizon, args.raft_iterations, args.raft_batch_size) <= 0:
        raise ValueError("horizon, RAFT iterations, and batch size must be positive")
    rows = read_manifest(args.manifest.resolve(), None)
    if args.task is not None:
        rows = [row for row in rows if row.get("task") == args.task]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("manifest/task selection produced no windows")
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset_root"]), int(row["episode"]))].append(row)
    assigned = [
        item
        for index, item in enumerate(sorted(groups.items()))
        if index % args.world_size == args.rank
    ]
    total = sum(len(group_rows) for _, group_rows in assigned)
    device = torch.device(args.device, args.rank if args.device == "cuda" else None)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition

    raft = load_raft(
        args.raft_source.resolve(),
        args.raft_checkpoint.resolve(),
        device=device,
    )
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.model.resolve(),
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    started = time.perf_counter()
    complete = 0
    for (raw_root, episode), group_rows in assigned:
        root = Path(raw_root)
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        plan = plan_h3_window(
            action_horizon=args.action_horizon,
            source_fps=float(info["fps"]),
        )
        episode_name = f"episode_{episode:06d}"
        decoded = {
            camera: decode_episode(
                root / "videos/chunk-000" / camera / f"{episode_name}.mp4"
            )
            for camera in CAMERAS
        }
        for row in group_rows:
            output = args.motion_root / f"{row['id']}.pt"
            if output.exists() and not args.overwrite:
                complete += 1
                continue
            clips = []
            start = int(row["start"])
            for camera in CAMERAS:
                clip = decoded[camera][start : start + plan.source_frame_count]
                if clip.shape[0] != plan.source_frame_count:
                    raise RuntimeError(f"short video window for {row['id']}")
                clips.append(resize_uint8(clip, args.camera_height, args.camera_width))
            video = resample_video_nearest(torch.cat(clips, dim=-1), plan.h3_frame_count)
            motion_video = raft_motion_video(
                raft,
                video,
                device=device,
                iterations=args.raft_iterations,
                batch_size=args.raft_batch_size,
            )
            # A 39-frame H3 VAE encode peaks close to an 80 GiB device by
            # itself. RAFT has already returned a CPU color-flow video, so do
            # not keep its parameters/activations resident during VAE encode.
            if device.type == "cuda":
                raft.to("cpu")
                torch.cuda.empty_cache()
            try:
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    motion_latents = encode_vae_condition(
                        vae,
                        motion_video.permute(1, 0, 2, 3).unsqueeze(0).to(device),
                        PIXEL_MEAN,
                        PIXEL_STD,
                    ).detach().cpu()
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    raft.to(device)
            rgb_artifact = torch.load(
                args.cache_root / "windows" / f"{row['id']}.pt",
                map_location="cpu",
                weights_only=False,
            )
            if motion_latents.shape != rgb_artifact["video_latents"].shape:
                raise ValueError(
                    f"motion/RGB latent mismatch for {row['id']}: "
                    f"{tuple(motion_latents.shape)} vs "
                    f"{tuple(rgb_artifact['video_latents'].shape)}"
                )
            artifact = {
                "flow_latents": motion_latents.contiguous(),
                "sample_id": row["id"],
                "representation": "RAFT color-wheel video encoded by MiniMax-H3 VAE",
                "raft_iterations": args.raft_iterations,
                "h3_frame_count": plan.h3_frame_count,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
            torch.save(artifact, temporary)
            os.replace(temporary, output)
            complete += 1
            if complete % args.progress_every == 0:
                progress(complete, total, started, args.rank)
        del decoded
    progress(complete, total, started, args.rank)


if __name__ == "__main__":
    main()
