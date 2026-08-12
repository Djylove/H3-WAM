#!/usr/bin/env python3
"""Precompute full LIBERO windows with the official Diffusers MiniMax-H3 VAE."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from fastwam.models.h3wam import (
    h3_latent_is_pad,
    plan_h3_window,
    resample_video_nearest,
)


CAMERAS = ("observation.images.image", "observation.images.wrist_image")
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("vae", "stats"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument(
        "--rank-offset",
        type=int,
        default=0,
        help=(
            "Offset LOCAL_RANK only for dataset assignment. This lets multiple "
            "nodes cooperatively encode one shared cache without distributed "
            "collectives (for example offsets 0 and 8 with world-size 16)."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--vae-batch-size",
        type=int,
        default=1,
        help="Encode this many equal-shaped H3 windows per VAE call.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def read_manifest(path: Path, limit: int | None) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows if limit is None else rows[:limit]


def decode_episode(path: Path) -> torch.Tensor:
    import av

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return torch.stack(frames)


def resize_uint8(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(frames.shape[-2:]) == (height, width):
        return frames
    resized = F.interpolate(
        frames.float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.round().clamp_(0, 255).to(torch.uint8)


def encode_vae_condition_batch_exact(
    vae,
    pixels: torch.Tensor,
    pixel_mean: tuple,
    pixel_std: tuple,
    encode_seed: int = 42,
) -> torch.Tensor:
    """Batch H3 VAE encoding while preserving per-window posterior sampling.

    The official helper resets a CPU generator to ``encode_seed`` for every
    conditioning clip.  Sampling one batched posterior with one generator
    would therefore change every item after index zero.  Encode the batch once
    but sample each posterior slice with its own freshly seeded generator so
    cached latents remain numerically equivalent to the original batch-1 path.
    """

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    mean = torch.tensor(pixel_mean, device=pixels.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(pixel_std, device=pixels.device).view(1, -1, 1, 1, 1)
    normalized = (pixels.to(torch.float32).div(255.0) - mean) / std
    posterior = vae.encode(normalized, return_dict=False)[0]
    sampled = []
    for index in range(pixels.shape[0]):
        generator = torch.Generator(device="cpu").manual_seed(encode_seed)
        noise = torch.randn(
            posterior.mean[index : index + 1].shape,
            generator=generator,
            dtype=posterior.mean.dtype,
            device="cpu",
        ).to(posterior.mean.device)
        sampled.append(
            posterior.mean[index : index + 1]
            + posterior.std[index : index + 1] * noise
        )
    latents = torch.cat(sampled).to(torch.float16).float().cpu()
    return (latents - latents_mean) / latents_std


def progress(stage: str, complete: int, total: int, started: float, rank: int) -> None:
    elapsed = time.perf_counter() - started
    seconds_per_window = elapsed / max(complete, 1)
    print(
        json.dumps(
            {
                "event": "progress",
                "stage": stage,
                "rank": rank,
                "complete": complete,
                "total": total,
                "seconds": round(elapsed, 2),
                "seconds_per_window": round(seconds_per_window, 3),
                "eta_seconds": round(seconds_per_window * (total - complete), 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def round_robin_episode_groups(
    groups: dict[tuple[str, int], list[dict]],
) -> list[tuple[tuple[str, int], list[dict]]]:
    """Interleave episodes across tasks before assigning them to GPU ranks.

    Dense manifests are ordered by suite, task, episode and start.  Assigning
    the sorted episode keys directly therefore fills a few tasks first and
    makes a partially completed cache unusable for a multi-task canary.  A
    task-round-robin order gives every task one episode before taking a second
    episode from any task while preserving deterministic output.
    """

    episodes_by_task: dict[tuple[str, str], list[tuple[tuple[str, int], list[dict]]]] = (
        defaultdict(list)
    )
    for key, episode_rows in groups.items():
        task_keys = {
            (str(row.get("suite", "")), str(row["task"])) for row in episode_rows
        }
        if len(task_keys) != 1:
            raise ValueError(f"episode group {key} contains multiple tasks: {task_keys}")
        episodes_by_task[task_keys.pop()].append((key, episode_rows))
    for task_groups in episodes_by_task.values():
        task_groups.sort(key=lambda item: item[0])

    ordered: list[tuple[tuple[str, int], list[dict]]] = []
    task_keys = sorted(episodes_by_task)
    for episode_offset in range(max(map(len, episodes_by_task.values()))):
        for task_key in task_keys:
            task_groups = episodes_by_task[task_key]
            if episode_offset < len(task_groups):
                ordered.append(task_groups[episode_offset])
    if len(ordered) != len(groups):
        raise RuntimeError("task-round-robin ordering lost an episode group")
    return ordered


def precompute_vae(args: argparse.Namespace, rows: list[dict]) -> None:
    if args.model is None:
        raise ValueError("--model is required for the VAE stage")
    assignment_rank = args.rank + args.rank_offset
    if not 0 <= assignment_rank < args.world_size:
        raise ValueError("rank + rank-offset must be in [0, world-size)")
    import pandas as pd
    from diffusers import AutoencoderKLMiniMaxH3

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset_root"]), int(row["episode"]))].append(row)
    ordered_groups = round_robin_episode_groups(groups)
    assigned_groups = [
        item
        for index, item in enumerate(ordered_groups)
        if index % args.world_size == assignment_rank
    ]
    assigned_total = sum(len(items) for _, items in assigned_groups)
    device = torch.device(args.device, args.rank if args.device == "cuda" else None)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.model.resolve(),
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    vae.eval()
    started = time.perf_counter()
    complete = 0
    for (raw_root, episode_index), episode_rows in assigned_groups:
        root = Path(raw_root)
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        plan = plan_h3_window(
            action_horizon=args.action_horizon,
            source_fps=float(info["fps"]),
        )
        episode_name = f"episode_{episode_index:06d}"
        table = pd.read_parquet(root / "data/chunk-000" / f"{episode_name}.parquet")
        decoded = {
            camera: decode_episode(
                root / "videos/chunk-000" / camera / f"{episode_name}.mp4"
            )
            for camera in CAMERAS
        }
        if any(len(frames) != len(table) for frames in decoded.values()):
            raise RuntimeError(f"video/parquet mismatch in {root.name}/{episode_name}")
        pending = [
            row
            for row in episode_rows
            if args.overwrite
            or not (args.cache_root / "windows" / f"{row['id']}.pt").exists()
        ]
        complete += len(episode_rows) - len(pending)
        for batch_start in range(0, len(pending), args.vae_batch_size):
            batch_rows = pending[batch_start : batch_start + args.vae_batch_size]
            videos = []
            for row in batch_rows:
                start = int(row["start"])
                camera_clips = []
                source_is_pad = torch.arange(plan.source_frame_count) + start >= len(table)
                source_indices = (
                    torch.arange(plan.source_frame_count) + start
                ).clamp_max(len(table) - 1)
                for camera in CAMERAS:
                    clip = decoded[camera].index_select(0, source_indices)
                    camera_clips.append(
                        resize_uint8(clip, args.camera_height, args.camera_width)
                    )
                video = torch.cat(camera_clips, dim=-1)
                videos.append(resample_video_nearest(video, plan.h3_frame_count))
            video_bcfhw = torch.stack(videos).permute(0, 2, 1, 3, 4).to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                video_latents = encode_vae_condition_batch_exact(
                    vae, video_bcfhw, PIXEL_MEAN, PIXEL_STD
                )
                first_frame_latents = encode_vae_condition_batch_exact(
                    vae, video_bcfhw[:, :, :1], PIXEL_MEAN, PIXEL_STD
                )
            if video_latents.shape[0] != len(batch_rows):
                raise RuntimeError("batched H3 VAE changed the batch dimension")
            for index, row in enumerate(batch_rows):
                start = int(row["start"])
                action_indices = (
                    torch.arange(args.action_horizon) + start
                ).clamp_max(len(table) - 1)
                action_is_pad = torch.arange(args.action_horizon) + start >= len(table)
                action_values = table["action"].iloc[action_indices.tolist()]
                actions = torch.stack(
                    [torch.from_numpy(value.copy()) for value in action_values]
                ).float()
                actions[action_is_pad, :6] = 0.0
                state = torch.from_numpy(
                    table["observation.state"].iloc[start].copy()
                ).float()
                if tuple(actions.shape) != (args.action_horizon, 7) or tuple(state.shape) != (8,):
                    raise ValueError(
                        f"unexpected action/state shapes for {row['id']}: "
                        f"{tuple(actions.shape)}, {tuple(state.shape)}"
                    )
                artifact = {
                    "video_latents": video_latents[index : index + 1],
                    "first_frame_latents": first_frame_latents[index : index + 1],
                    "actions": actions,
                    "state": state,
                    "task": row["task"],
                    "suite": row["suite"],
                    "episode": episode_index,
                    "start": start,
                    "source_fps": float(info["fps"]),
                    "h3_frame_count": plan.h3_frame_count,
                    "action_is_pad": action_is_pad,
                    "image_is_pad": resample_video_nearest(
                        source_is_pad, plan.h3_frame_count
                    ),
                    "vae_backend": "diffusers.AutoencoderKLMiniMaxH3",
                    "vae_pixel_mean": PIXEL_MEAN,
                    "vae_pixel_std": PIXEL_STD,
                }
                artifact["latent_is_pad"] = h3_latent_is_pad(
                    artifact["image_is_pad"]
                )
                output = args.cache_root / "windows" / f"{row['id']}.pt"
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
                torch.save(artifact, temporary)
                os.replace(temporary, output)
                complete += 1
            if complete % args.progress_every < len(batch_rows):
                progress("vae", complete, assigned_total, started, assignment_rank)
        del table, decoded
    progress("vae", complete, assigned_total, started, assignment_rank)


def compute_stats(args: argparse.Namespace, rows: list[dict]) -> None:
    if args.world_size != 1 or args.rank != 0:
        raise ValueError("stats stage runs once with rank=0, world-size=1")
    actions, states = [], []
    for row in rows:
        path = args.cache_root / "windows" / f"{row['id']}.pt"
        window = torch.load(path, map_location="cpu", weights_only=False)
        action = window["actions"][: args.action_horizon].float()
        action_is_pad = window.get(
            "action_is_pad", torch.zeros(len(action), dtype=torch.bool)
        )[: len(action)].bool()
        actions.append(action[~action_is_pad])
        states.append(window["state"].float())
    action = torch.cat(actions)
    state = torch.stack(states)
    stats = {
        "num_windows": len(rows),
        "action_min": action.amin(dim=0),
        "action_max": action.amax(dim=0),
        "action_mean": action.mean(dim=0),
        "action_std": action.std(dim=0).clamp_min(1e-6),
        "state_min": state.amin(dim=0),
        "state_max": state.amax(dim=0),
        "state_mean": state.mean(dim=0),
        "state_std": state.std(dim=0).clamp_min(1e-6),
    }
    args.cache_root.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.cache_root / "stats.pt")
    print(json.dumps({"event": "stats", "windows": len(rows)}, sort_keys=True))


def main() -> None:
    args = parse_args()
    if (
        args.action_horizon <= 0
        or args.world_size <= 0
        or args.progress_every <= 0
        or args.vae_batch_size <= 0
    ):
        raise ValueError("invalid positive argument")
    args.cache_root = args.cache_root.resolve()
    rows = read_manifest(args.manifest.resolve(), args.limit)
    if not rows:
        raise ValueError("manifest contains no rows")
    if args.stage == "vae":
        precompute_vae(args, rows)
    else:
        compute_stats(args, rows)


if __name__ == "__main__":
    main()
