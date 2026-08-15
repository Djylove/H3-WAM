#!/usr/bin/env python3
"""Audit one deterministic shard of the dense DreamWAM or StarWAM cache."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


# One process owns one independent manifest shard.  Letting every process
# create a large intra-op pool oversubscribes the host and makes small tensor
# validation slower than single-threaded execution.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


SCRIPT_ROOT = Path(__file__).resolve().parent


def load_module(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPT_ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


KV_AUDIT = load_module("_h3_kv_audit", "audit_h3_dreamwam_kv_cache.py")
STAR_AUDIT = load_module("_h3_star_audit", "audit_h3_starwam_feature_cache.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("kv", "star"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--subdir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-num-shards", type=int, default=32)
    parser.add_argument("--audit-shard-index", type=int, required=True)
    parser.add_argument("--producer-num-shards", type=int, default=32)
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


def audit_args(args: argparse.Namespace, manifest_items: int) -> argparse.Namespace:
    common = {
        "capture_token_count": 32,
        "action_horizon": 32,
        "timestep": 1.0,
        "condition_video_timestep": 1.0,
        "expected_checkpoint": args.expected_checkpoint,
        "max_error_examples": args.max_error_examples,
    }
    if args.mode == "kv":
        return argparse.Namespace(
            **common,
            limit=manifest_items,
            num_shards=args.producer_num_shards,
            num_heads=56,
            attn_head_dim=128,
        )
    return argparse.Namespace(
        **common,
        producer_num_shards=args.producer_num_shards,
        feature_dim=5376,
    )


def main() -> None:
    args = parse_args()
    if args.audit_num_shards <= 0 or not 0 <= args.audit_shard_index < args.audit_num_shards:
        raise ValueError("audit-shard-index must be in [0,audit-num-shards)")
    if args.producer_num_shards <= 0 or args.max_error_examples <= 0:
        raise ValueError("producer shards and max error examples must be positive")
    if len(args.expected_checkpoint_sha256) != 64:
        raise ValueError("expected checkpoint SHA256 must contain 64 hex characters")

    started = time.perf_counter()
    manifest = args.manifest.resolve()
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("manifest is empty")
    all_ids = [str(row["id"]) for row in rows]
    duplicate_ids = len(all_ids) - len(set(all_ids))
    selected = list(enumerate(rows))[args.audit_shard_index :: args.audit_num_shards]
    expected_selected = len(rows[args.audit_shard_index :: args.audit_num_shards])
    if len(selected) != expected_selected:
        raise RuntimeError("audit shard selection is inconsistent")

    cache_dir = args.cache_root.resolve() / args.subdir
    aggregate_hash = hashlib.sha256()
    file_sizes: list[int] = []
    missing: list[str] = []
    error_count = 0
    error_examples: list[str] = []
    resolved_audit_args = audit_args(args, len(rows))
    for row_index, row in selected:
        sample_id = str(row["id"])
        path = cache_dir / f"{sample_id}.pt"
        if not path.is_file():
            missing.append(sample_id)
            continue
        try:
            data = path.read_bytes()
            file_sizes.append(len(data))
            digest = sha256_bytes(data)
            aggregate_hash.update(sample_id.encode("utf-8"))
            aggregate_hash.update(b"\0")
            aggregate_hash.update(digest.encode("ascii"))
            aggregate_hash.update(b"\n")
            payload: dict[str, Any] = torch.load(
                io.BytesIO(data), map_location="cpu", weights_only=False
            )
            if args.mode == "kv":
                errors = KV_AUDIT.audit_payload(
                    payload,
                    row,
                    row_index=row_index,
                    args=resolved_audit_args,
                )
            else:
                errors = STAR_AUDIT.audit_payload(
                    payload,
                    row,
                    row_index=row_index,
                    manifest_items=len(rows),
                    args=resolved_audit_args,
                )
        except Exception as error:  # continue independent files in this shard
            errors = [f"{sample_id}:load:{type(error).__name__}:{error}"]
        error_count += len(errors)
        remaining = args.max_error_examples - len(error_examples)
        if remaining > 0:
            error_examples.extend(errors[:remaining])

    valid = duplicate_ids == 0 and not missing and error_count == 0
    result = {
        "valid": valid,
        "mode": args.mode,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "manifest_rows": len(rows),
        "duplicate_manifest_ids": duplicate_ids,
        "cache_root": str(cache_dir),
        "audit_num_shards": args.audit_num_shards,
        "audit_shard_index": args.audit_shard_index,
        "producer_num_shards": args.producer_num_shards,
        "expected_rows": expected_selected,
        "audited_rows": len(file_sizes),
        "missing_count": len(missing),
        "missing_examples": missing[: args.max_error_examples],
        "tensor_metadata_error_count": error_count,
        "error_examples": error_examples,
        "expected_checkpoint": str(args.expected_checkpoint.resolve()),
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
        "aggregate_shard_sha256": aggregate_hash.hexdigest(),
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
