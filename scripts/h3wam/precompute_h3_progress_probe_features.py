#!/usr/bin/env python3
"""Compact frozen H3 K/V into a small, episode-disjoint progress probe set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from fastwam.models.h3wam import compact_h3_kv_progress_feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--kv-subdir", required=True)
    parser.add_argument("--per-suite", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_rows(rows: list[dict], per_suite: int, bins: int = 5) -> list[dict]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        index = min(bins - 1, int(float(row["value_raw"]) * bins))
        grouped[row["suite"]][index].append(row)
    selected = []
    for suite in sorted(grouped):
        candidates = []
        quota = per_suite // bins
        for index in range(bins):
            ordered = sorted(
                grouped[suite][index],
                key=lambda row: hashlib.sha256(row["id"].encode()).digest(),
            )
            take = ordered[:quota]
            selected.extend(take)
            candidates.extend(ordered[quota:])
        remaining = per_suite - sum(row["suite"] == suite for row in selected)
        selected.extend(sorted(candidates, key=lambda row: hashlib.sha256(row["id"].encode()).digest())[:remaining])
    return sorted(selected, key=lambda row: (row["suite"], row["id"]))


def main() -> None:
    args = parse_args()
    if args.per_suite <= 0 or args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid selection or shard arguments")
    rows = [json.loads(line) for line in args.target_manifest.read_text().splitlines() if line.strip()]
    selected = select_rows(rows, args.per_suite)
    selection_hash = hashlib.sha256("\n".join(row["id"] for row in selected).encode()).hexdigest()
    shard = selected[args.shard_index :: args.num_shards]
    features = []
    cache = args.cache_root / args.kv_subdir
    for row in shard:
        payload = torch.load(cache / f"{row['id']}.pt", map_location="cpu", weights_only=False)
        features.append(
            compact_h3_kv_progress_feature(payload["video_kv_cache"][49]).to(
                torch.bfloat16
            )
        )
    result = {
        "format": "h3wam-progress-probe-features-v1",
        "selection_sha256": selection_hash,
        "selected_total": len(selected), "num_shards": args.num_shards, "shard_index": args.shard_index,
        "ids": [row["id"] for row in shard],
        "suite": [row["suite"] for row in shard],
        "context_id": [row["context_id"] for row in shard],
        "start": torch.tensor([row["start"] for row in shard], dtype=torch.float32),
        "target": torch.tensor([row["value_raw"] for row in shard], dtype=torch.float32),
        "features": torch.stack(features),
        "feature_contract": "layer49 concat(mean_k,std_k,mean_v,std_v) over token/head -> 512",
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    torch.save(result, temporary); os.replace(temporary, output)
    print(json.dumps({key: result[key] for key in ("selection_sha256", "selected_total", "num_shards", "shard_index")}, sort_keys=True))


if __name__ == "__main__":
    main()
