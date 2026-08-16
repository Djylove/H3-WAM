#!/usr/bin/env python3
"""Freeze the exact C58/full30 versus D0/five-layer paired training order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DistributedSampler


EXPECTED_D0_SHA256 = "36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"
BASE_OFFSET = 112_000
STAGE_STEPS = 1_000
GLOBAL_BATCH = 8
STAGES = 10
STAGE_ROWS = STAGE_STEPS * GLOBAL_BATCH
SEED = 42


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def stage_rank_orders(ids: list[str]) -> list[list[str]]:
    if len(ids) != STAGE_ROWS:
        raise ValueError(f"paired stage must contain {STAGE_ROWS} rows")
    orders = []
    for rank in range(GLOBAL_BATCH):
        sampler = DistributedSampler(
            ids,
            num_replicas=GLOBAL_BATCH,
            rank=rank,
            shuffle=True,
            seed=SEED,
            drop_last=False,
        )
        orders.append([ids[index] for index in sampler])
    if any(len(order) != STAGE_STEPS for order in orders):
        raise RuntimeError("rank-local paired sample count differs from 1000")
    flattened = [sample_id for order in orders for sample_id in order]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("paired DistributedSampler order duplicated sample IDs")
    return orders


def build_pair_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 200_779:
        raise ValueError(f"C58 manifest must contain 200779 rows, got {len(rows)}")
    all_ids = [str(row["id"]) for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("C58 manifest IDs are not unique")
    stages = []
    combined = []
    for stage_index in range(STAGES):
        offset = BASE_OFFSET + stage_index * STAGE_ROWS
        selected = all_ids[offset : offset + STAGE_ROWS]
        orders = stage_rank_orders(selected)
        rank_major = [sample_id for order in orders for sample_id in order]
        step_major = [
            orders[rank][step]
            for step in range(STAGE_STEPS)
            for rank in range(GLOBAL_BATCH)
        ]
        combined.extend(selected)
        stages.append(
            {
                "milestone": (stage_index + 1) * STAGE_STEPS,
                "sample_offset": offset,
                "sample_limit": STAGE_ROWS,
                "selected_manifest_order_sha256": sha256_strings(selected),
                "rank_order_sha256": [sha256_strings(order) for order in orders],
                "rank_major_consumed_sha256": sha256_strings(rank_major),
                "step_major_global_batch_sha256": sha256_strings(step_major),
                "first_step_sample_ids_by_rank": [order[0] for order in orders],
                "last_step_sample_ids_by_rank": [order[-1] for order in orders],
            }
        )
    if len(combined) != 80_000 or len(set(combined)) != 80_000:
        raise RuntimeError("C58 pair does not select exactly 80000 unique rows")
    return {
        "format": "h3wam-c58-matched-depth-pair-contract-v1",
        "status": "PREREGISTERED_NOT_EFFECT_EVIDENCE",
        "arms": {
            "candidate": "C58_FASTWAM_FULL30_H3_LAYER49",
            "control": "C58_MATCHED_D0_FRESH_OPTIMIZER",
        },
        "single_variable": "official ActionDiT depth 30 versus D0 depth 5",
        "common_contract": {
            "base_offset": BASE_OFFSET,
            "optimizer_steps": STAGES * STAGE_STEPS,
            "global_batch": GLOBAL_BATCH,
            "training_samples": 80_000,
            "seed": SEED,
            "distributed_sampler": "shuffle=True,epoch=0,drop_last=False per explicit 8000-row stage",
            "flow_seed": "base42+completed_step*1000003+accumulation_index*10007+rank*10000019",
            "optimizer": "AdamW(lr=1e-4,weight_decay=0.01,betas=(0.9,0.95))",
            "schedule": "linear_warmup1000_then_cosine_horizon10000_min_lr1e-6",
            "action": "horizon32,shift5,7D,minmax,pad-masked velocity MSE",
            "carrier": "frozen H3 repeat_layer49 K/V, 32 tokens, 56x128",
            "checkpoint_every": 1000,
        },
        "combined_selected_manifest_order_sha256": sha256_strings(combined),
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--d0-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    plan = build_pair_plan(rows)
    parent_path = args.d0_parent.resolve()
    parent_sha256 = sha256_file(parent_path)
    if parent_sha256 != EXPECTED_D0_SHA256:
        raise ValueError(f"D0 parent SHA256 mismatch: {parent_sha256}")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    contract = parent.get("contract", {})
    if (
        contract.get("candidate") != "D0"
        or contract.get("carrier_source_mode") != "repeat_layer49"
        or parent.get("completed_steps") != 14_000
    ):
        raise ValueError("D0 parent identity/contract mismatch")
    selected = {
        str(row["id"])
        for row in rows[BASE_OFFSET : BASE_OFFSET + STAGES * STAGE_ROWS]
    }
    consumed = set(parent.get("data_state", {}).get("sample_ids", []))
    overlap = selected & consumed
    if overlap:
        raise ValueError(f"paired 80k rows overlap D0 parent by {len(overlap)}")
    plan.update(
        {
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest.resolve()),
            "d0_parent_path": str(parent_path),
            "d0_parent_sha256": parent_sha256,
            "d0_parent_completed_steps": 14_000,
            "d0_parent_optimizer_policy": "weights_only_fresh_optimizer_for_both_arms",
            "d0_parent_consumed_overlap": 0,
        }
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(plan, indent=2) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"existing pair contract differs: {output}")
        return
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
