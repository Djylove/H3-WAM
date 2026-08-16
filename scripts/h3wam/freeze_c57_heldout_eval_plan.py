#!/usr/bin/env python3
"""Freeze episode-disjoint C57 samples, flow seeds and checkpoint schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def episode_identity(row: dict) -> tuple[str, str]:
    return str(row["suite"]), str(row["episode"])


def stable_rank(seed: int, row: dict) -> str:
    return hashlib.sha256(f"{seed}:{row['current_id']}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source-manifest", type=Path, required=True)
    parser.add_argument("--heldout-source-manifest", type=Path, required=True)
    parser.add_argument("--heldout-sequence-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--per-suite", type=int, default=20)
    parser.add_argument("--seed", type=int, default=570042)
    args = parser.parse_args()
    if args.per_suite <= 0:
        raise ValueError("per-suite must be positive")
    train = rows(args.train_source_manifest)
    heldout = rows(args.heldout_source_manifest)
    sequence = rows(args.heldout_sequence_manifest)
    train_episodes = {episode_identity(row) for row in train}
    heldout_episodes = {episode_identity(row) for row in heldout}
    overlap = train_episodes & heldout_episodes
    if overlap:
        raise ValueError(f"train/heldout episode leakage: {sorted(overlap)[:5]}")
    source_by_id = {str(row["id"]): row for row in heldout}
    if len(source_by_id) != len(heldout):
        raise ValueError("heldout source IDs are not unique")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sequence:
        current = source_by_id.get(str(row["current_id"]))
        if current is None:
            raise ValueError(f"sequence current ID absent from heldout: {row['current_id']}")
        if not row.get("history"):
            continue
        enriched = dict(row)
        enriched["suite"] = str(current["suite"])
        enriched["episode"] = str(current["episode"])
        grouped[enriched["suite"]].append(enriched)
    selected: list[dict] = []
    for suite in sorted(grouped):
        best_per_episode: dict[str, dict] = {}
        for row in grouped[suite]:
            episode = row["episode"]
            old = best_per_episode.get(episode)
            score = (len(row["history"]), stable_rank(args.seed, row))
            if old is None or score > (len(old["history"]), stable_rank(args.seed, old)):
                best_per_episode[episode] = row
        candidates = sorted(
            best_per_episode.values(), key=lambda row: stable_rank(args.seed, row)
        )
        if len(candidates) < args.per_suite:
            raise ValueError(
                f"suite {suite} has only {len(candidates)} episode-disjoint history samples"
            )
        selected.extend(candidates[: args.per_suite])
    if not selected:
        raise ValueError("no heldout history sample was selected")
    selected.sort(key=lambda row: (row["suite"], row["episode"], row["current_id"]))
    for row in selected:
        row["eval_flow_seed"] = int.from_bytes(
            hashlib.sha256(f"{args.seed}:{row['current_id']}:flow".encode()).digest()[:8],
            "big",
        ) % (2**31)
    args.selected_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.selected_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    )
    milestones = list(range(200, 5001, 200))
    plan = {
        "schema": "c57_heldout_eval_plan_v1",
        "candidate": "C57",
        "control": "D0_frozen_source_checkpoint",
        "selection_seed": args.seed,
        "samples": len(selected),
        "suites": sorted(grouped),
        "per_suite": args.per_suite,
        "train_source_manifest": str(args.train_source_manifest.resolve()),
        "train_source_sha256": sha256(args.train_source_manifest),
        "heldout_source_manifest": str(args.heldout_source_manifest.resolve()),
        "heldout_source_sha256": sha256(args.heldout_source_manifest),
        "heldout_sequence_manifest": str(args.heldout_sequence_manifest.resolve()),
        "heldout_sequence_sha256": sha256(args.heldout_sequence_manifest),
        "selected_manifest": str(args.selected_manifest.resolve()),
        "selected_manifest_sha256": sha256(args.selected_manifest),
        "checkpoint_milestones": milestones,
        "selection_policy": "max-history_one-sample-per-episode_then_seeded-hash",
        "promotion_checkpoint": 5000,
        "offline_gate": {
            "paired_mean_loss_relative_improvement_min": 0.03,
            "paired_sample_win_fraction_min": 0.55,
            "nonfinite_allowed": 0,
            "decision": "GO_CLOSED_LOOP_CANARY_ONLY",
        },
        "closed_loop_gate": {
            "replan": 8,
            "observe_every": 4,
            "same_task_trials_environment_seed_as_D0": True,
            "requires_real_persistent_trace": True,
        },
    }
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
