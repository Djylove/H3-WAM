#!/usr/bin/env python3
"""Batch-cache LIBERO windows for H3-WAM without co-loading large models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import av
import pandas as pd
import torch
import torchvision.transforms.functional as tvf

from fastwam.models.h3wam import plan_h3_window, resample_video_nearest


CAMERAS = ("observation.images.image", "observation.images.wrist_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("vae", "context", "actions"))
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--video-vae", type=Path)
    parser.add_argument("--text-encoder", type=Path)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--first-frame-only",
        action="store_true",
        help="Cache only the current-frame VAE latent for H3 feature-action training.",
    )
    return parser.parse_args()


def read_manifest(path: Path, limit: int | None = None) -> list[dict]:
    with path.open() as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    return items if limit is None else items[:limit]


def decode_episode(path: Path) -> torch.Tensor:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            array = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return torch.stack(frames)


def load_tasks(root: Path) -> dict[int, str]:
    tasks = {}
    with (root / "meta/tasks.jsonl").open() as handle:
        for line in handle:
            item = json.loads(line)
            tasks[int(item["task_index"])] = item["task"]
    return tasks


def print_progress(stage: str, completed: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    rate = elapsed / max(completed, 1)
    print(
        json.dumps(
            {
                "stage": stage,
                "completed": completed,
                "total": total,
                "elapsed_seconds": round(elapsed, 2),
                "seconds_per_window": round(rate, 3),
                "eta_seconds": round(rate * (total - completed), 1),
            }
        ),
        flush=True,
    )


def cache_vae(args: argparse.Namespace, items: list[dict]) -> None:
    if args.video_vae is None:
        raise ValueError("--video-vae is required for the vae stage")
    import comfy.model_management as model_management
    import comfy.sd
    import comfy.utils

    root = args.dataset_root.resolve()
    info = json.loads((root / "meta/info.json").read_text())
    plan = plan_h3_window(
        action_horizon=args.action_horizon,
        source_fps=float(info["fps"]),
    )
    tasks = load_tasks(root)
    vae_state = comfy.utils.load_torch_file(str(args.video_vae.resolve()))
    vae = comfy.sd.VAE(sd=vae_state)
    del vae_state

    grouped: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        grouped[int(item["episode"])].append(item)
    started = time.perf_counter()
    completed = 0
    for episode_index, episode_items in grouped.items():
        episode_name = f"episode_{episode_index:06d}"
        table = pd.read_parquet(root / "data/chunk-000" / f"{episode_name}.parquet")
        decoded = {
            camera: decode_episode(
                root / "videos/chunk-000" / camera / f"{episode_name}.mp4"
            )
            for camera in CAMERAS
        }
        if any(frames.shape[0] != len(table) for frames in decoded.values()):
            raise RuntimeError(f"video/parquet frame mismatch in episode {episode_index}")

        for item in episode_items:
            output = args.cache_root / "windows" / f"{item['id']}.pt"
            if output.exists() and not args.overwrite:
                completed += 1
                continue
            start = int(item["start"])
            cameras = []
            for camera in CAMERAS:
                stop = start + (1 if args.first_frame_only else plan.source_frame_count)
                frames = decoded[camera][start:stop].to(dtype=torch.float32).div_(255.0)
                frames = tvf.resize(
                    frames,
                    [args.camera_height, args.camera_width],
                    antialias=True,
                )
                cameras.append(frames)
            video = torch.cat(cameras, dim=-1)
            if not args.first_frame_only:
                video = resample_video_nearest(video, plan.h3_frame_count)
            video_nhwc = video.permute(0, 2, 3, 1).contiguous()
            with torch.inference_mode():
                first_frame_latents = vae.encode(video_nhwc[:1])
                video_latents = (
                    None if args.first_frame_only else vae.encode(video_nhwc)
                )

            action_values = table["action"].iloc[start : start + args.action_horizon]
            actions = torch.stack(
                [torch.from_numpy(value.copy()) for value in action_values]
            ).float()
            state = torch.from_numpy(table["observation.state"].iloc[start].copy()).float()
            task_index = int(table["task_index"].iloc[start])
            artifact = {
                "first_frame_latents": first_frame_latents.cpu(),
                "actions": actions,
                "state": state,
                "task": tasks[task_index],
                "task_index": task_index,
                "episode": episode_index,
                "start": start,
                "source_fps": float(info["fps"]),
                "h3_frame_count": plan.h3_frame_count,
            }
            if video_latents is not None:
                artifact["video_latents"] = video_latents.cpu()
                artifact["first_frame_pixels"] = video_nhwc[:1].clone().cpu()
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(artifact, output)
            completed += 1
            if completed % args.progress_every == 0:
                print_progress("vae", completed, len(items), started)
        del decoded, table
    if model_management.get_torch_device().type == "cuda":
        torch.cuda.synchronize()
    write_stats(args.cache_root, items, args.action_horizon)
    print_progress("vae", completed, len(items), started)


def write_stats(cache_root: Path, items: list[dict], action_horizon: int) -> None:
    actions, states = [], []
    for item in items:
        path = cache_root / "windows" / f"{item['id']}.pt"
        if not path.exists():
            continue
        window = torch.load(path, map_location="cpu", weights_only=False)
        actions.append(window["actions"][:action_horizon])
        states.append(window["state"])
    if not actions:
        return
    action = torch.cat(actions, dim=0).float()
    state = torch.stack(states).float()
    stats = {
        "num_windows": len(states),
        "action_min": action.amin(dim=0),
        "action_max": action.amax(dim=0),
        "action_mean": action.mean(dim=0),
        "action_std": action.std(dim=0).clamp_min(1e-6),
        "state_min": state.amin(dim=0),
        "state_max": state.amax(dim=0),
        "state_mean": state.mean(dim=0),
        "state_std": state.std(dim=0).clamp_min(1e-6),
    }
    torch.save(stats, cache_root / "stats.pt")


def cache_actions(args: argparse.Namespace, items: list[dict]) -> None:
    """Extend cached action chunks without recomputing VAE or text features.

    Dense Horizon1 manifests can add tail-state windows that share IDs with a
    later Horizon8 manifest.  Keep longer existing chunks intact and only
    repair artifacts that are shorter than the requested horizon.
    """

    root = args.dataset_root.resolve()
    grouped: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        grouped[int(item["episode"])].append(item)
    started = time.perf_counter()
    completed = 0
    repaired = 0
    for episode_index, episode_items in grouped.items():
        episode_name = f"episode_{episode_index:06d}"
        table = pd.read_parquet(root / "data/chunk-000" / f"{episode_name}.parquet")
        for item in episode_items:
            output = args.cache_root / "windows" / f"{item['id']}.pt"
            if not output.exists():
                raise FileNotFoundError(output)
            artifact = torch.load(output, map_location="cpu", weights_only=False)
            if len(artifact["actions"]) < args.action_horizon:
                start = int(item["start"])
                values = table["action"].iloc[start : start + args.action_horizon]
                actions = torch.stack(
                    [torch.from_numpy(value.copy()) for value in values]
                ).float()
                if len(actions) != args.action_horizon:
                    raise RuntimeError(
                        f"{item['id']} has only {len(actions)} source actions; "
                        f"expected {args.action_horizon}"
                    )
                artifact["actions"] = actions
                temporary = output.with_suffix(output.suffix + ".tmp")
                torch.save(artifact, temporary)
                temporary.replace(output)
                repaired += 1
            completed += 1
            if completed % args.progress_every == 0:
                print_progress("actions", completed, len(items), started)
    write_stats(args.cache_root, items, args.action_horizon)
    print(
        json.dumps(
            {
                "stage": "actions",
                "completed": completed,
                "total": len(items),
                "repaired": repaired,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
            }
        ),
        flush=True,
    )


def cache_context(args: argparse.Namespace, items: list[dict]) -> None:
    if args.text_encoder is None:
        raise ValueError("--text-encoder is required for the context stage")
    import comfy.model_management as model_management
    import comfy.sd

    clip = comfy.sd.load_clip(
        ckpt_paths=[str(args.text_encoder.resolve())],
        clip_type=comfy.sd.CLIPType.MINIMAX,
    )
    started = time.perf_counter()
    completed = 0
    for item in items:
        output = args.cache_root / "contexts" / f"{item['id']}.pt"
        if output.exists() and not args.overwrite:
            completed += 1
            continue
        window_path = args.cache_root / "windows" / f"{item['id']}.pt"
        if not window_path.exists():
            raise FileNotFoundError(window_path)
        window = torch.load(window_path, map_location="cpu", weights_only=False)
        tokens = clip.tokenize(window["task"], images=[window["first_frame_pixels"]])
        conditioning = clip.encode_from_tokens_scheduled(tokens, show_pbar=False)
        context, metadata = conditioning[0]
        token_tags = metadata.get("minimax_token_tags")
        if token_tags is None:
            raise RuntimeError("MiniMax text encoder did not return minimax_token_tags")
        artifact = {
            "context": context.cpu(),
            "token_tags": token_tags.cpu(),
            "task": window["task"],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, output)
        completed += 1
        if completed % args.progress_every == 0:
            print_progress("context", completed, len(items), started)
    if model_management.get_torch_device().type == "cuda":
        torch.cuda.synchronize()
    print_progress("context", completed, len(items), started)


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.manifest = args.manifest.resolve()
    args.cache_root = args.cache_root.resolve()
    sys.path.insert(0, str(args.comfy_root.resolve()))
    items = read_manifest(args.manifest, args.limit)
    if not items:
        raise RuntimeError("manifest contains no windows")
    if args.stage == "vae":
        cache_vae(args, items)
    elif args.stage == "context":
        cache_context(args, items)
    else:
        cache_actions(args, items)


if __name__ == "__main__":
    main()
