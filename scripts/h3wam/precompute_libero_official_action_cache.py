#!/usr/bin/env python3
"""Build a dense LIBERO action/first-frame cache with the official H3 VAE."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

from precompute_libero_official_h3 import (
    CAMERAS,
    PIXEL_MEAN,
    PIXEL_STD,
    decode_episode,
    resize_uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("vae", "stats"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help=(
            "Fallback LeRobot root for manifests without dataset_root. Combined "
            "multi-suite manifests should carry dataset_root on every row."
        ),
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_atomic(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def run_vae(args: argparse.Namespace, rows: list[dict]) -> None:
    if args.model is None:
        raise ValueError("--model is required for the vae stage")
    import pandas as pd
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        raw_root = row.get("dataset_root")
        if raw_root is None:
            if args.dataset_root is None:
                raise ValueError("manifest row has no dataset_root and no fallback was set")
            raw_root = str(args.dataset_root.resolve())
        groups[(str(Path(raw_root).resolve()), int(row["episode"]))].append(row)
    assigned = [item for index, item in enumerate(sorted(groups.items())) if index % args.world_size == args.rank]
    total = sum(len(items) for _, items in assigned)
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
    completed = 0
    started = time.perf_counter()
    for (raw_root, episode), episode_rows in assigned:
        root = Path(raw_root)
        episode_rows = sorted(episode_rows, key=lambda item: int(item["start"]))
        episode_name = f"episode_{episode:06d}"
        table = pd.read_parquet(root / "data/chunk-000" / f"{episode_name}.parquet")
        decoded = {
            camera: decode_episode(
                root / "videos/chunk-000" / camera / f"{episode_name}.mp4"
            )
            for camera in CAMERAS
        }
        if any(len(frames) != len(table) for frames in decoded.values()):
            raise RuntimeError(f"video/parquet mismatch for {episode_name}")
        for offset in range(0, len(episode_rows), args.batch_size):
            batch = episode_rows[offset : offset + args.batch_size]
            pending = [
                item
                for item in batch
                if args.overwrite
                or not (args.cache_root / "windows" / f"{item['id']}.pt").exists()
            ]
            if pending:
                starts = torch.tensor([int(item["start"]) for item in pending], dtype=torch.long)
                cameras = [
                    resize_uint8(
                        decoded[camera].index_select(0, starts),
                        args.camera_height,
                        args.camera_width,
                    )
                    for camera in CAMERAS
                ]
                frames = torch.cat(cameras, dim=-1).unsqueeze(2).to(device)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    first_latents = encode_vae_condition(
                        vae, frames, PIXEL_MEAN, PIXEL_STD
                    ).cpu()
                if first_latents.shape[0] != len(pending):
                    raise RuntimeError("official H3 VAE changed the cache batch dimension")
                for index, item in enumerate(pending):
                    start = int(item["start"])
                    action_values = table["action"].iloc[start : start + args.action_horizon]
                    actions = torch.stack(
                        [torch.from_numpy(value.copy()) for value in action_values]
                    ).float()
                    state = torch.from_numpy(
                        table["observation.state"].iloc[start].copy()
                    ).float()
                    if tuple(actions.shape) != (args.action_horizon, 7) or tuple(state.shape) != (8,):
                        raise ValueError(f"bad action/state shape for {item['id']}")
                    save_atomic(
                        {
                            # A view retains the storage for the entire VAE batch,
                            # making every per-window torch.save tens of times too
                            # large. Materialize the single-sample storage first.
                            "first_frame_latents": first_latents[
                                index : index + 1
                            ].clone(),
                            "actions": actions,
                            "state": state,
                            "task": item["task"],
                            "episode": episode,
                            "start": start,
                            "length": int(item["length"]),
                            "h3_frame_count": 39,
                            "vae_backend": "diffusers.AutoencoderKLMiniMaxH3",
                        },
                        args.cache_root / "windows" / f"{item['id']}.pt",
                    )
            completed += len(batch)
            if completed % args.progress_every < len(batch) or completed == total:
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "rank": args.rank,
                            "complete": completed,
                            "total": total,
                            "elapsed_seconds": round(elapsed, 2),
                        }
                    ),
                    flush=True,
                )
        del table, decoded


def run_stats(args: argparse.Namespace, rows: list[dict]) -> None:
    if args.world_size != 1 or args.rank != 0:
        raise ValueError("stats must run once")
    actions, states = [], []
    for row in rows:
        item = torch.load(
            args.cache_root / "windows" / f"{row['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        actions.append(item["actions"])
        states.append(item["state"])
    action = torch.cat(actions)
    state = torch.stack(states)
    stats = {
        "num_windows": len(rows),
        "action_min": action.amin(0),
        "action_max": action.amax(0),
        "action_mean": action.mean(0),
        "action_std": action.std(0).clamp_min(1e-6),
        "state_min": state.amin(0),
        "state_max": state.amax(0),
        "state_mean": state.mean(0),
        "state_std": state.std(0).clamp_min(1e-6),
    }
    save_atomic(stats, args.cache_root / "stats.pt")
    print(json.dumps({"stage": "stats", "windows": len(rows)}), flush=True)


def main() -> None:
    args = parse_args()
    if min(args.action_horizon, args.batch_size, args.world_size, args.progress_every) <= 0:
        raise ValueError("positive cache arguments are required")
    rows = read_rows(args.manifest.resolve())
    if not rows:
        raise ValueError("manifest is empty")
    args.cache_root = args.cache_root.resolve()
    if args.stage == "vae":
        run_vae(args, rows)
    else:
        run_stats(args, rows)


if __name__ == "__main__":
    main()
