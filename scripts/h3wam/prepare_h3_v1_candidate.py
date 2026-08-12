#!/usr/bin/env python3
"""Audit cached LIBERO/H3 pairs and make an episode-disjoint V1 split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-episodes-per-task", type=int, default=1)
    parser.add_argument("--split-salt", default="h3wam-libero-goal-v1-2026-08-06")
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def validate_pair(cache_root: Path, row: dict) -> dict:
    sample_id = str(row["id"])
    window_path = cache_root / "windows" / f"{sample_id}.pt"
    context_path = cache_root / "contexts" / f"{sample_id}.pt"
    if not window_path.is_file() or not context_path.is_file():
        raise FileNotFoundError(f"missing window/context pair for {sample_id}")
    window = torch.load(window_path, map_location="cpu", weights_only=False)
    conditioning = torch.load(context_path, map_location="cpu", weights_only=False)

    video = window["video_latents"]
    first = window["first_frame_latents"]
    context = conditioning["context"]
    tags = conditioning["token_tags"]
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError(f"{sample_id}: invalid video shape {tuple(video.shape)}")
    if tuple(first.shape) != (1, 24, 1, video.shape[3], video.shape[4]):
        raise ValueError(f"{sample_id}: invalid first-frame shape {tuple(first.shape)}")
    if context.ndim != 3 or context.shape[0] != 1 or context.shape[-1] != 5120:
        raise ValueError(f"{sample_id}: invalid raw H3 context shape {tuple(context.shape)}")
    if tags.ndim != 1 or tags.numel() != context.shape[1]:
        raise ValueError(f"{sample_id}: token tags do not match context")
    if not set(tags.unique().tolist()).issubset({0, 1}):
        raise ValueError(f"{sample_id}: invalid H3 token tags")
    for name, tensor in {
        "video_latents": video,
        "first_frame_latents": first,
        "context": context,
    }.items():
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{sample_id}: {name} is non-floating or non-finite")
    task = str(row["task"])
    if str(window.get("task")) != task or str(conditioning.get("task")) != task:
        raise ValueError(f"{sample_id}: task metadata mismatch")
    return {
        "video_shape": tuple(video.shape),
        "context_shape": tuple(context.shape),
        "video_dtype": str(video.dtype),
        "context_dtype": str(context.dtype),
    }


def main() -> None:
    args = parse_args()
    if args.val_episodes_per_task < 1:
        raise ValueError("val-episodes-per-task must be positive")
    cache_root = args.cache_root.resolve()
    source_manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    rows = [
        json.loads(line)
        for line in source_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("manifest contains duplicate sample IDs")

    schemas = Counter()
    episodes_by_task: dict[int, set[int]] = defaultdict(set)
    task_names: dict[int, str] = {}
    for index, row in enumerate(rows, 1):
        task_group = int(row["task_group"])
        episode = int(row["episode"])
        task = str(row["task"])
        if task_group in task_names and task_names[task_group] != task:
            raise ValueError(f"task group {task_group} has inconsistent names")
        task_names[task_group] = task
        episodes_by_task[task_group].add(episode)
        schema = validate_pair(cache_root, row)
        schemas[json.dumps(schema, sort_keys=True)] += 1
        if index % 50 == 0:
            print(json.dumps({"event": "audit", "checked": index, "total": len(rows)}), flush=True)

    validation_episodes: dict[int, list[int]] = {}
    for task_group, episodes in sorted(episodes_by_task.items()):
        if len(episodes) <= args.val_episodes_per_task:
            raise ValueError(f"task {task_group} does not have enough episodes")
        ranked = sorted(
            episodes,
            key=lambda episode: hashlib.sha256(
                f"{args.split_salt}:{task_group}:{episode}".encode()
            ).hexdigest(),
        )
        validation_episodes[task_group] = sorted(ranked[: args.val_episodes_per_task])

    validation_keys = {
        (task_group, episode)
        for task_group, episodes in validation_episodes.items()
        for episode in episodes
    }
    train = [
        row
        for row in rows
        if (int(row["task_group"]), int(row["episode"])) not in validation_keys
    ]
    validation = [
        row
        for row in rows
        if (int(row["task_group"]), int(row["episode"])) in validation_keys
    ]
    train_groups = {(int(row["task_group"]), int(row["episode"])) for row in train}
    validation_groups = {
        (int(row["task_group"]), int(row["episode"])) for row in validation
    }
    if train_groups & validation_groups:
        raise RuntimeError("episode leakage between train and validation")

    output_dir.mkdir(parents=True, exist_ok=False)
    train_manifest = output_dir / "manifest_train.jsonl"
    validation_manifest = output_dir / "manifest_val.jsonl"
    write_jsonl(train_manifest, train)
    write_jsonl(validation_manifest, validation)
    report = {
        "candidate_type": "fast_bring_up",
        "source": "LIBERO Goal cached H3 VAE latents and per-window raw Qwen context",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": digest(source_manifest),
        "cache_root": str(cache_root),
        "model_contract": {
            "model": "MiniMaxAI/MiniMax-H3 transformer FL2VA",
            "video_latents": "[1,24,T,H,W] float finite",
            "first_frame_latents": "[1,24,1,H,W] float finite",
            "context": "[1,L,5120] raw Qwen context, not refined 5376",
            "token_tags": "[L] values in {0,1}",
            "normalization": "none; consume H3 VAE latent domain as cached",
        },
        "split": {
            "method": "salted_sha256_episode_group",
            "salt": args.split_salt,
            "validation_episodes_by_task": validation_episodes,
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": digest(train_manifest),
            "validation_manifest": str(validation_manifest),
            "validation_manifest_sha256": digest(validation_manifest),
        },
        "source_demo_count": len({int(row["episode"]) for row in rows}),
        "segment_count": len(rows),
        "train_segments": len(train),
        "validation_segments": len(validation),
        "tasks": task_names,
        "schema_counts": dict(schemas),
        "quarantine": [],
        "loader_smoke": "pending cloud multi-sample canary",
    }
    report_path = output_dir / "training_candidate.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "report": str(report_path), **report}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
