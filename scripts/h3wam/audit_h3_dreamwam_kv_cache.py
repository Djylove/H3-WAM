#!/usr/bin/env python3
"""Audit a completed Candidate D projected K/V cache without modifying it."""

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


DREAMWAM_COMMIT = "6e989facc0c452fd3488d75f60bc36411005558c"
SCHEMA = "h3_dreamwam_kv_v1"
LAYERS = (9, 19, 29, 39, 49)
TOKEN_STRATEGY = "adaptive_avg_pool1d_sequence_v1"
BACKBONE = "H3Int8FeatureBackbone"
QUANTIZATION = "int8_tensorwise_convrot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--kv-subdir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8560)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--capture-token-count", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=56)
    parser.add_argument("--attn-head-dim", type=int, default=128)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--condition-video-timestep", type=float, default=1.0)
    parser.add_argument("--expected-checkpoint", type=Path, required=True)
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
    args: argparse.Namespace,
) -> list[str]:
    sample_id = str(row["id"])
    expected_metadata = {
        "schema": SCHEMA,
        "layers": LAYERS,
        "episode": int(row["episode"]),
        "start": int(row["start"]),
        "suite": str(row["suite"]),
        "context_id": str(row["context_id"]),
        "timestep": float(args.timestep),
        "condition_video_timestep": float(args.condition_video_timestep),
        "action_horizon": int(args.action_horizon),
        "capture_token_count": int(args.capture_token_count),
        "num_heads": int(args.num_heads),
        "attn_head_dim": int(args.attn_head_dim),
        "capture_token_strategy": TOKEN_STRATEGY,
        "dreamwam_commit": DREAMWAM_COMMIT,
        "backbone": BACKBONE,
        "quantization": QUANTIZATION,
        "checkpoint": str(args.expected_checkpoint.resolve()),
        "manifest_items": int(args.limit),
        "num_shards": int(args.num_shards),
        "shard_index": row_index % args.num_shards,
    }
    errors = []
    for key, expected in expected_metadata.items():
        actual = payload.get(key)
        if key == "layers" and actual is not None:
            actual = tuple(actual)
        if actual != expected:
            errors.append(
                f"{sample_id}:metadata:{key}:{actual!r}!={expected!r}"
            )
    cache = payload.get("video_kv_cache")
    if not isinstance(cache, dict) or set(cache) != set(LAYERS):
        errors.append(f"{sample_id}:video_kv_cache:layer_set")
        return errors
    expected_shape = (
        args.capture_token_count,
        args.num_heads,
        args.attn_head_dim,
    )
    storage_pointers: set[int] = set()
    for layer in LAYERS:
        item = cache[layer]
        if not isinstance(item, dict) or set(item) != {"k", "v"}:
            errors.append(f"{sample_id}:layer:{layer}:keys")
            continue
        for name in ("k", "v"):
            tensor = item[name]
            if not isinstance(tensor, torch.Tensor):
                errors.append(f"{sample_id}:layer:{layer}:{name}:not_tensor")
                continue
            if tuple(tensor.shape) != expected_shape:
                errors.append(
                    f"{sample_id}:layer:{layer}:{name}:shape:{tuple(tensor.shape)}"
                )
            if tensor.dtype != torch.bfloat16:
                errors.append(
                    f"{sample_id}:layer:{layer}:{name}:dtype:{tensor.dtype}"
                )
            if not bool(torch.isfinite(tensor.float()).all()):
                errors.append(f"{sample_id}:layer:{layer}:{name}:nonfinite")
            pointer = tensor.untyped_storage().data_ptr()
            if pointer in storage_pointers:
                errors.append(f"{sample_id}:layer:{layer}:{name}:storage_alias")
            storage_pointers.add(pointer)
    return errors


def main() -> None:
    args = parse_args()
    if min(
        args.limit,
        args.num_shards,
        args.capture_token_count,
        args.num_heads,
        args.attn_head_dim,
        args.action_horizon,
        args.max_error_examples,
    ) <= 0:
        raise ValueError("positive audit dimensions are required")
    started = time.perf_counter()
    manifest = args.manifest.resolve()
    cache_root = args.cache_root.resolve()
    kv_root = cache_root / args.kv_subdir
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    if len(rows) != args.limit:
        raise ValueError(f"manifest contains {len(rows)} rows, expected {args.limit}")
    ids = [str(row["id"]) for row in rows]
    expected_ids = set(ids)
    duplicate_ids = len(ids) - len(expected_ids)
    paths = list(kv_root.iterdir()) if kv_root.is_dir() else []
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
        if path.is_file()
        and (path.name.startswith(".") or path.suffix != ".pt")
    )
    aggregate_hash = hashlib.sha256()
    file_sizes = []
    error_count = 0
    error_examples: list[str] = []
    shard_counts = {index: 0 for index in range(args.num_shards)}
    split_counts: dict[str, int] = {}
    for row_index, row in enumerate(rows):
        split = str(row.get("split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
        sample_id = str(row["id"])
        path = completed_paths.get(sample_id)
        if path is None:
            continue
        shard_counts[row_index % args.num_shards] += 1
        try:
            data = path.read_bytes()
            file_sizes.append(len(data))
            digest = sha256_bytes(data)
            aggregate_hash.update(sample_id.encode("utf-8"))
            aggregate_hash.update(b"\0")
            aggregate_hash.update(digest.encode("ascii"))
            aggregate_hash.update(b"\n")
            payload = torch.load(
                io.BytesIO(data), map_location="cpu", weights_only=False
            )
            payload_errors = audit_payload(
                payload, row, row_index=row_index, args=args
            )
        except Exception as error:  # keep auditing the remaining independent files
            payload_errors = [f"{sample_id}:load:{type(error).__name__}:{error}"]
        error_count += len(payload_errors)
        remaining = args.max_error_examples - len(error_examples)
        if remaining > 0:
            error_examples.extend(payload_errors[:remaining])
    expected_per_shard = {
        index: len(rows[index :: args.num_shards])
        for index in range(args.num_shards)
    }
    checkpoint_sha256 = sha256_file(args.expected_checkpoint.resolve())
    valid = not any(
        (
            duplicate_ids,
            missing_ids,
            extra_ids,
            temporary_files,
            error_count,
            shard_counts != expected_per_shard,
        )
    )
    result = {
        "valid": valid,
        "schema": SCHEMA,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "manifest_rows": len(rows),
        "split_counts": split_counts,
        "duplicate_ids": duplicate_ids,
        "cache_root": str(kv_root),
        "completed_files": len(completed_paths),
        "missing_count": len(missing_ids),
        "missing_examples": missing_ids[: args.max_error_examples],
        "extra_count": len(extra_ids),
        "extra_examples": extra_ids[: args.max_error_examples],
        "temporary_count": len(temporary_files),
        "temporary_examples": temporary_files[: args.max_error_examples],
        "tensor_metadata_error_count": error_count,
        "error_examples": error_examples,
        "expected_per_shard": expected_per_shard,
        "completed_per_shard": shard_counts,
        "checkpoint": str(args.expected_checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
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
