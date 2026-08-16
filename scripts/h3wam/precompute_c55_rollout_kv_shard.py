#!/usr/bin/env python3
"""Precompute exact five-layer H3 K/V for FACT rollout observations.

The extractor is intentionally schema-generic: C48 observational rollouts and
C60 state-aligned counterfactual branches share one audited image/KV path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam import (  # noqa: E402
    H3Int8FeatureBackbone,
    H3Int8OnlineKVContract,
    H3Int8OnlineKVProvider,
    encode_h3_vae_condition_standalone,
    preprocess_libero_cameras,
)
from fastwam.models.h3wam.dreamwam_kv_carrier import (  # noqa: E402
    DEFAULT_H3_CARRIER_LAYERS,
    DREAMWAM_COMMIT,
)


FORMAT = "h3wam-c55-rollout-kv-shard-v1"
KV_SCHEMA = "h3_dreamwam_kv_v1"
KV_STRATEGY = "adaptive_avg_pool1d_sequence_v1"
EXPECTED_H3_SHA256 = (
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
)
EXPECTED_SOURCE_SHA256 = (
    "cab8876f067114dce41d16ca52cb0bafddf17da33c92d0adde5f11d7ac9555b9"
)
SUPPORTED_DATASET_FORMATS = {
    "h3wam-c48-fact-dense-value-dataset-v1",
    "h3wam-c60-counterfactual-failure-dataset-v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-observations-sha256", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--h3-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "validation"))
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def task_context_ids(source_manifest: Path, tasks: set[str]) -> dict[str, str]:
    matches: dict[str, set[str]] = defaultdict(set)
    with source_manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            task = str(row["task"])
            if task in tasks:
                matches[task].add(str(row["context_id"]))
    missing = sorted(tasks - set(matches))
    ambiguous = {task: sorted(ids) for task, ids in matches.items() if len(ids) != 1}
    if missing or ambiguous:
        raise ValueError(
            f"task/context mapping invalid: missing={missing}, ambiguous={ambiguous}"
        )
    return {task: next(iter(ids)) for task, ids in matches.items()}


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError("shard must be in [0,num_shards)")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")
    if args.limit < 0:
        raise ValueError("limit cannot be negative")
    splits = tuple(str(value) for value in args.splits)
    if not splits or not set(splits) <= {"train", "validation"}:
        raise ValueError("C55 K/V extraction permits train/validation only")

    dataset_path = args.dataset.resolve()
    observations_path = args.observations.resolve()
    source_manifest = args.source_manifest.resolve()
    h3_checkpoint = args.h3_checkpoint.resolve()
    output_root = args.output_root.resolve()
    marker = output_root / "markers" / f"shard{args.shard}.json"
    if marker.exists():
        raise FileExistsError(f"refusing to overwrite completed shard: {marker}")
    if sha256_file(h3_checkpoint) != EXPECTED_H3_SHA256:
        raise ValueError("H3 checkpoint identity mismatch")
    if sha256_file(source_manifest) != EXPECTED_SOURCE_SHA256:
        raise ValueError("source manifest identity mismatch")
    dataset_sha256 = sha256_file(dataset_path)
    observations_sha256 = sha256_file(observations_path)
    if dataset_sha256 != args.expected_dataset_sha256:
        raise ValueError("FACT rollout dataset identity mismatch")
    if observations_sha256 != args.expected_observations_sha256:
        raise ValueError("FACT rollout observations identity mismatch")

    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if dataset.get("format") not in SUPPORTED_DATASET_FORMATS:
        raise ValueError(f"unsupported FACT rollout dataset: {dataset.get('format')!r}")
    required_ids = {
        int(row["current_observation_id"])
        for row in dataset["samples"]
        if row["split"] in splits
    }
    observation_rows = [
        json.loads(line)
        for line in observations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {int(row["observation_id"]): row for row in observation_rows}
    if len(by_id) != len(observation_rows) or not required_ids <= set(by_id):
        raise ValueError("observation provenance is incomplete or duplicated")
    rows = [
        by_id[observation_id]
        for observation_id in sorted(required_ids)
        if observation_id % args.num_shards == args.shard
    ]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("empty C55 K/V shard")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("INT8 H3 K/V extraction requires CUDA")
    torch.cuda.set_device(device)
    context_map = task_context_ids(
        source_manifest, {str(row["task_language"]) for row in rows}
    )
    contexts = {}
    cache_root = args.cache_root.resolve()
    for task, context_id in context_map.items():
        payload = torch.load(
            cache_root / "contexts" / f"{context_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        tags = payload.get("token_tags")
        if payload.get("text_only") is not True or tags is None or torch.any(tags != 1):
            raise ValueError(f"context is not audited text-only: {context_id}")
        contexts[task] = {
            "id": context_id,
            "context": payload["context"].to(device=device, dtype=torch.bfloat16),
            "tags": tags.to(device),
        }

    from diffusers import AutoencoderKLMiniMaxH3

    backbone = H3Int8FeatureBackbone.from_checkpoint(h3_checkpoint).to(device).eval()
    backbone.requires_grad_(False)
    provider = H3Int8OnlineKVProvider(
        backbone,
        H3Int8OnlineKVContract(
            layers=DEFAULT_H3_CARRIER_LAYERS,
            action_horizon=32,
            target_latent_frames=12,
            video_timestep=1.0,
            condition_video_timestep=1.0,
            capture_token_count=32,
            pool_strategy=KV_STRATEGY,
        ),
    )
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.h3_model.resolve(),
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    vae.requires_grad_(False)

    item_root = output_root / "items"
    item_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    archive = None
    last_trajectory = None
    written = []
    try:
        for position, row in enumerate(rows):
            trajectory = str(row["trajectory"])
            if trajectory != last_trajectory:
                if archive is not None:
                    archive.close()
                archive = np.load(trajectory, allow_pickle=False)
                last_trajectory = trajectory
            if row["kind"] != "row":
                raise ValueError("C55 current observations must be trajectory rows")
            index = int(row["row_index"])
            pixels = preprocess_libero_cameras(
                archive["agentview_image"][index],
                archive["wristview_image"][index],
            )
            video = (
                pixels.mul(255).round().to(torch.uint8).permute(0, 3, 1, 2)
                .unsqueeze(2).to(device)
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                first_frame = encode_h3_vae_condition_standalone(
                    vae,
                    video,
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ).to(device=device, dtype=torch.float32)
            context = contexts[str(row["task_language"])]
            with torch.inference_mode():
                live = provider(first_frame, context["context"], context["tags"])
            video_kv_cache = {
                layer: {
                    name: live[layer][name][0].to(torch.bfloat16).cpu().clone()
                    for name in ("k", "v")
                }
                for layer in DEFAULT_H3_CARRIER_LAYERS
            }
            observation_id = int(row["observation_id"])
            output = item_root / f"obs_{observation_id:06d}.pt"
            if output.exists():
                raise FileExistsError(f"refusing to overwrite K/V item: {output}")
            payload = {
                "schema": KV_SCHEMA,
                "format": FORMAT,
                "observation_id": observation_id,
                "episode_id": int(row["episode_id"]),
                "split": str(row["split"]),
                "context_id": context["id"],
                "layers": DEFAULT_H3_CARRIER_LAYERS,
                "capture_token_count": 32,
                "num_heads": 56,
                "attn_head_dim": 128,
                "capture_token_strategy": KV_STRATEGY,
                "dreamwam_commit": DREAMWAM_COMMIT,
                "h3_checkpoint_sha256": EXPECTED_H3_SHA256,
                "dataset_format": dataset["format"],
                "dataset_sha256": dataset_sha256,
                "observations_sha256": observations_sha256,
                "video_kv_cache": video_kv_cache,
            }
            temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
            torch.save(payload, temporary)
            os.replace(temporary, output)
            written.append(observation_id)
            if (position + 1) % args.progress_every == 0 or position + 1 == len(rows):
                print(
                    json.dumps(
                        {
                            "shard": args.shard,
                            "completed": position + 1,
                            "total": len(rows),
                            "seconds": round(time.perf_counter() - started, 2),
                        }
                    ),
                    flush=True,
                )
    finally:
        if archive is not None:
            archive.close()

    marker.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "format": FORMAT,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "splits": list(splits),
        "items": len(written),
        "first_observation_id": min(written),
        "last_observation_id": max(written),
        "dataset_format": dataset["format"],
        "dataset_sha256": dataset_sha256,
        "observations_sha256": observations_sha256,
        "h3_checkpoint_sha256": EXPECTED_H3_SHA256,
        "elapsed_seconds": time.perf_counter() - started,
    }
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
