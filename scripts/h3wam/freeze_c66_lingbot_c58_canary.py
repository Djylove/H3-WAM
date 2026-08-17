#!/usr/bin/env python3
"""Freeze the balanced, episode-disjoint C66 100-step canary split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


SCHEMA = "h3wam-c66-lingbot-c58-canary-plan-v1"
SEQUENCE_SCHEMA = "c57_lingbot_replan8_v1"
C58_SHA256 = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_manifest", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=66017)
    args = parser.parse_args()

    source = args.sequence_manifest.resolve()
    parent = args.parent_checkpoint.resolve()
    if sha256_file(parent) != C58_SHA256:
        raise ValueError("C66 split must be frozen against the promoted C58 checkpoint")
    parent_payload = torch.load(parent, map_location="cpu", weights_only=False)
    consumed = set(parent_payload["data_state"]["sample_ids"])
    del parent_payload
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("sequence_schema") != SEQUENCE_SCHEMA for row in rows):
        raise ValueError("C66 source sequence schema mismatch")

    representatives: dict[tuple[str, int], dict] = {}
    for row in rows:
        if row["current_id"] in consumed or int(row["history_chunks"]) != 7:
            continue
        key = (str(row["suite"]), int(row["episode"]))
        previous = representatives.get(key)
        if previous is None or rank(args.seed, row["id"]) < rank(args.seed, previous["id"]):
            representatives[key] = row

    train: list[dict] = []
    heldout: list[dict] = []
    for suite in SUITES:
        candidates = [row for (name, _), row in representatives.items() if name == suite]
        candidates.sort(key=lambda row: rank(args.seed + 1, f"{suite}:{row['episode']}"))
        if len(candidates) < 216:
            raise ValueError(f"C66 needs 216 unseen full-history episodes for {suite}")
        train.extend(candidates[:200])
        heldout.extend(candidates[200:216])
    train.sort(key=lambda row: rank(args.seed + 2, row["id"]))
    heldout.sort(key=lambda row: rank(args.seed + 3, row["id"]))
    train_keys = {(row["suite"], row["episode"]) for row in train}
    heldout_keys = {(row["suite"], row["episode"]) for row in heldout}
    if len(train) != 800 or len(heldout) != 64 or train_keys & heldout_keys:
        raise RuntimeError("C66 balanced episode split construction failed")

    output = args.output_dir.resolve()
    train_path = output / "manifest_train800.jsonl"
    heldout_path = output / "manifest_heldout64.jsonl"
    atomic_text(train_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in train))
    atomic_text(heldout_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in heldout))
    plan = {
        "schema": SCHEMA,
        "seed": args.seed,
        "source_sequence_manifest": str(source),
        "source_sequence_manifest_sha256": sha256_file(source),
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": C58_SHA256,
        "selection": "one full-seven-chunk row per episode; current target unseen by C58 training; balanced 200/16 per suite",
        "train_manifest": str(train_path),
        "train_manifest_sha256": sha256_file(train_path),
        "heldout_manifest": str(heldout_path),
        "heldout_manifest_sha256": sha256_file(heldout_path),
        "train_rows": 800,
        "heldout_rows": 64,
        "train_episodes": 800,
        "heldout_episodes": 64,
        "episode_intersection": 0,
        "suite_train_rows": {suite: sum(row["suite"] == suite for row in train) for suite in SUITES},
        "suite_heldout_rows": {suite: sum(row["suite"] == suite for row in heldout) for suite in SUITES},
        "history_chunks": 7,
        "history_observation_frames": 15,
        "history_executed_actions": 56,
        "budget": {"world_size": 8, "steps": 100, "microbatch": 1, "training_samples": 800},
        "arms": ["clean_context", "history_action_shuffle", "context_off"],
    }
    atomic_text(output / "PLAN.json", json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
