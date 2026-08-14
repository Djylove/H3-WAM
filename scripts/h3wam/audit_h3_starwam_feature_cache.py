#!/usr/bin/env python3
"""Audit a completed H3 last-layer StarWAM feature cache without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch


LAYERS = (49,)
TOKEN_STRATEGY = "starwam_adaptive_avg_pool1d_v1"
BACKBONE = "H3Int8FeatureBackbone"
QUANTIZATION = "int8_tensorwise_convrot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--feature-subdir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 audits the full manifest")
    parser.add_argument("--producer-num-shards", type=int, default=32)
    parser.add_argument("--capture-token-count", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=5376)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--condition-video-timestep", type=float, default=1.0)
    parser.add_argument("--expected-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--max-error-examples", type=int, default=50)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_payload(
    payload: dict[str, Any],
    row: dict[str, Any],
    *,
    row_index: int,
    manifest_items: int,
    args: argparse.Namespace,
) -> list[str]:
    sample_id = str(row["id"])
    expected_metadata = {
        "layers": LAYERS,
        "capture_token_count": int(args.capture_token_count),
        "capture_token_strategy": TOKEN_STRATEGY,
        "capture_compatibility": "none",
        "episode": int(row["episode"]),
        "start": int(row["start"]),
        "suite": str(row["suite"]),
        "context_id": str(row["context_id"]),
        "timestep": float(args.timestep),
        "condition_video_timestep": float(args.condition_video_timestep),
        "action_horizon": int(args.action_horizon),
        "backbone": BACKBONE,
        "quantization": QUANTIZATION,
        "checkpoint": str(args.expected_checkpoint.resolve()),
        "manifest_items": int(manifest_items),
        "num_shards": int(args.producer_num_shards),
        "shard_index": row_index % int(args.producer_num_shards),
    }
    errors: list[str] = []
    for key, expected in expected_metadata.items():
        actual = payload.get(key)
        if key == "layers" and actual is not None:
            actual = tuple(actual)
        if actual != expected:
            errors.append(f"{sample_id}:metadata:{key}:{actual!r}!={expected!r}")

    context_width = payload.get("context_width")
    context_mode = payload.get("context_mode")
    if (context_width, context_mode) not in ((5120, "raw_qwen"), (5376, "refined_h3")):
        errors.append(
            f"{sample_id}:metadata:context_contract:{context_width!r},{context_mode!r}"
        )

    features = payload.get("features")
    expected_shape = (len(LAYERS), args.capture_token_count, args.feature_dim)
    if not isinstance(features, torch.Tensor):
        errors.append(f"{sample_id}:features:not_tensor")
        return errors
    if tuple(features.shape) != expected_shape:
        errors.append(f"{sample_id}:features:shape:{tuple(features.shape)}")
    if features.dtype != torch.bfloat16:
        errors.append(f"{sample_id}:features:dtype:{features.dtype}")
    if not bool(torch.isfinite(features.float()).all()):
        errors.append(f"{sample_id}:features:nonfinite")
    return errors


def main() -> None:
    args = parse_args()
    if min(
        args.producer_num_shards,
        args.capture_token_count,
        args.feature_dim,
        args.action_horizon,
        args.max_error_examples,
    ) <= 0:
        raise ValueError("positive audit dimensions are required")
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    started = time.perf_counter()
    manifest = args.manifest.resolve()
    feature_root = args.cache_root.resolve() / args.feature_subdir
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("manifest is empty")

    ids = [str(row["id"]) for row in rows]
    expected_ids = set(ids)
    duplicate_ids = len(ids) - len(expected_ids)
    paths = list(feature_root.iterdir()) if feature_root.is_dir() else []
    completed_paths = {
        path.stem: path
        for path in paths
        if path.is_file() and path.suffix == ".pt" and not path.name.startswith(".")
    }
    actual_ids = set(completed_paths)
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    temporary_files = sorted(
        path.name
        for path in paths
        if path.is_file() and (path.name.startswith(".") or path.suffix != ".pt")
    )

    aggregate_hash = hashlib.sha256()
    file_sizes: list[int] = []
    error_count = 0
    error_examples: list[str] = []
    producer_counts = {index: 0 for index in range(args.producer_num_shards)}
    split_counts: dict[str, int] = {}
    for row_index, row in enumerate(rows):
        split = str(row.get("split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
        sample_id = str(row["id"])
        path = completed_paths.get(sample_id)
        if path is None:
            continue
        producer_counts[row_index % args.producer_num_shards] += 1
        try:
            data = path.read_bytes()
            file_sizes.append(len(data))
            digest = sha256_bytes(data)
            aggregate_hash.update(sample_id.encode("utf-8"))
            aggregate_hash.update(b"\0")
            aggregate_hash.update(digest.encode("ascii"))
            aggregate_hash.update(b"\n")
            payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
            payload_errors = audit_payload(
                payload,
                row,
                row_index=row_index,
                manifest_items=len(rows),
                args=args,
            )
        except Exception as error:  # keep auditing independent files
            payload_errors = [f"{sample_id}:load:{type(error).__name__}:{error}"]
        error_count += len(payload_errors)
        remaining = args.max_error_examples - len(error_examples)
        if remaining > 0:
            error_examples.extend(payload_errors[:remaining])

    expected_per_producer = {
        index: len(rows[index :: args.producer_num_shards])
        for index in range(args.producer_num_shards)
    }
    checkpoint = args.expected_checkpoint.resolve()
    checkpoint_sha256 = sha256_file(checkpoint)
    checkpoint_identity_match = checkpoint_sha256 == args.expected_checkpoint_sha256
    valid = not any(
        (
            duplicate_ids,
            missing_ids,
            extra_ids,
            temporary_files,
            error_count,
            producer_counts != expected_per_producer,
            not checkpoint_identity_match,
        )
    )
    result = {
        "valid": valid,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "manifest_rows": len(rows),
        "split_counts": split_counts,
        "duplicate_ids": duplicate_ids,
        "cache_root": str(feature_root),
        "completed_files": len(completed_paths),
        "missing_count": len(missing_ids),
        "missing_examples": missing_ids[: args.max_error_examples],
        "extra_count": len(extra_ids),
        "extra_examples": extra_ids[: args.max_error_examples],
        "temporary_count": len(temporary_files),
        "temporary_examples": temporary_files[: args.max_error_examples],
        "tensor_metadata_error_count": error_count,
        "error_examples": error_examples,
        "expected_per_producer_shard": expected_per_producer,
        "completed_per_producer_shard": producer_counts,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
        "checkpoint_identity_match": checkpoint_identity_match,
        "aggregate_cache_sha256": aggregate_hash.hexdigest(),
        "total_bytes": sum(file_sizes),
        "file_size_min": min(file_sizes) if file_sizes else 0,
        "file_size_median": statistics.median(file_sizes) if file_sizes else 0,
        "file_size_max": max(file_sizes) if file_sizes else 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
