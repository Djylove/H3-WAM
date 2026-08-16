#!/usr/bin/env python3
"""Audit C56b's no-K/V-cache datasets before online H3 integration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam.fact_online_data import (  # noqa: E402
    OnlineH3FACTDemoDataset,
    OnlineH3FACTRolloutDataset,
    collate_online_fact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-manifest", type=Path, required=True)
    parser.add_argument("--demo-source-manifest", type=Path, required=True)
    parser.add_argument("--demo-cache-root", type=Path, required=True)
    parser.add_argument("--c48-dataset", type=Path, required=True)
    parser.add_argument("--c48-observations", type=Path, required=True)
    parser.add_argument("--c59-overlay-root", type=Path, required=True)
    parser.add_argument("--c60-dataset", type=Path, required=True)
    parser.add_argument("--c60-observations", type=Path, required=True)
    parser.add_argument("--expected-c60-sha256", required=True)
    parser.add_argument("--expected-c60-observations-sha256", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summary(dataset, *, read_item: bool = True) -> dict:
    result = {
        "stream": dataset.stream_name,
        "items": len(dataset),
        "episodes": len(dataset.episode_to_indices),
    }
    if read_item:
        item = dataset[0]
        batch = collate_online_fact([item])
        forbidden = sorted(
            {"video_kv_cache", "h3_features", "future_h3_target"} & set(batch)
        )
        if forbidden:
            raise RuntimeError(f"cached H3 fields crossed online boundary: {forbidden}")
        result.update(
            {
                "sample_id": item["sample_id"],
                "input_mode": item["input_mode"],
                "current_h3_input_shape": list(batch["current_h3_input"].shape),
                "future_h3_input_shape": list(batch["future_h3_input"].shape),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    demo = OnlineH3FACTDemoDataset(
        args.demo_manifest,
        args.demo_source_manifest,
        args.demo_cache_root,
        split=args.split,
    )
    c48 = OnlineH3FACTRolloutDataset(
        args.c48_dataset,
        args.c48_observations,
        args.demo_source_manifest,
        args.demo_cache_root,
        split=args.split,
        c59_overlay_root=args.c59_overlay_root,
    )
    c60 = OnlineH3FACTRolloutDataset(
        args.c60_dataset,
        args.c60_observations,
        args.demo_source_manifest,
        args.demo_cache_root,
        split=args.split,
        expected_dataset_sha256=args.expected_c60_sha256,
        expected_observations_sha256=args.expected_c60_observations_sha256,
    )
    report = {
        "format": "h3wam-c56b-online-data-audit-v1",
        "status": "PASS_NO_KV_CACHE_DATA_BOUNDARY",
        "effect_status": "NOT_EVIDENCE_READY",
        "split": args.split,
        "streams": [summary(dataset) for dataset in (demo, c48, c60)],
        "c60_dataset_sha256": c60.dataset_sha256,
        "c60_observations_sha256": c60.observations_sha256,
        "claim_boundary": (
            "CPU data/label/input-boundary audit only; frozen H3 online forward, "
            "shared30 gradients, memory and throughput remain blocked on C58 interface."
        ),
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite audit: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
