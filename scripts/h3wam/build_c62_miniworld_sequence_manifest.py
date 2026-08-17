#!/usr/bin/env python3
"""Freeze episode-disjoint C62 real-history sequences and canary gates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "c62_miniworld_replan8_real_history_v1"
PLAN_SCHEMA = "h3wam-c62-causal-canary-plan-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def episode_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["suite"]), int(row["episode"])


def episode_score(key: tuple[str, int], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).hexdigest()


def build_sequence_rows(
    source_rows: list[dict[str, Any]], *, history_chunks: int
) -> list[dict[str, Any]]:
    if history_chunks < 2:
        raise ValueError("C62 requires at least two history chunks for shuffle audit")
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in source_rows:
        key = (*episode_key(row), int(row["start"]))
        if key in by_key:
            raise ValueError(f"duplicate suite/episode/start: {key}")
        by_key[key] = row
    result = []
    for current in source_rows:
        suite, episode = episode_key(current)
        start = int(current["start"])
        observation_starts = [
            start - 8 * distance for distance in range(history_chunks, 0, -1)
        ]
        if observation_starts[0] < 0:
            continue
        observations = [by_key.get((suite, episode, value)) for value in observation_starts]
        current_action_source = by_key.get((suite, episode, start - 8))
        if any(item is None for item in observations) or current_action_source is None:
            continue
        history = []
        for index, (observation_start, observation) in enumerate(
            zip(observation_starts, observations, strict=True)
        ):
            assert observation is not None
            # The first real observation is the persistent sink and therefore
            # has null action conditioning.  Every later observation is paired
            # with exactly the preceding eight executed actions.
            action_source = None
            if index > 0:
                action_source = by_key.get((suite, episode, observation_start - 8))
                if action_source is None:
                    break
            history.append(
                {
                    "observation_id": str(observation["id"]),
                    "observation_start": observation_start,
                    "actions_before_observation_id": (
                        None if action_source is None else str(action_source["id"])
                    ),
                    "action_indices": (
                        []
                        if action_source is None
                        else list(range(observation_start - 8, observation_start))
                    ),
                }
            )
        if len(history) != history_chunks:
            continue
        if any(max(item["action_indices"], default=-1) >= item["observation_start"] for item in history):
            raise RuntimeError("future action leakage in C62 history")
        enriched = dict(current)
        enriched.update(
            {
                "sequence_schema": SCHEMA,
                "current_id": str(current["id"]),
                "history": history,
                "actions_before_current_id": str(current_action_source["id"]),
                "actions_before_current_indices": list(range(start - 8, start)),
            }
        )
        result.append(enriched)
    return result


def select_episode_disjoint(
    rows: list[dict[str, Any]],
    *,
    train_per_suite: int,
    heldout_per_suite: int,
    heldout_episodes_per_suite: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_suite_episode: dict[str, dict[tuple[str, int], list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_suite_episode[str(row["suite"])][episode_key(row)].append(row)
    train, heldout = [], []
    for suite in sorted(by_suite_episode):
        episodes = sorted(
            by_suite_episode[suite], key=lambda key: episode_score(key, seed)
        )
        heldout_keys = set(episodes[:heldout_episodes_per_suite])
        heldout_pool = [
            row
            for key in episodes
            if key in heldout_keys
            for row in sorted(by_suite_episode[suite][key], key=lambda value: int(value["start"]))
        ]
        train_pool = [
            row
            for key in episodes
            if key not in heldout_keys
            for row in sorted(by_suite_episode[suite][key], key=lambda value: int(value["start"]))
        ]
        if len(train_pool) < train_per_suite or len(heldout_pool) < heldout_per_suite:
            raise ValueError(f"insufficient eligible C62 rows for {suite}")
        train.extend(train_pool[:train_per_suite])
        heldout.extend(heldout_pool[:heldout_per_suite])
    return train, heldout


def jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--history-chunks", type=int, default=3)
    parser.add_argument("--train-per-suite", type=int, default=200)
    parser.add_argument("--heldout-per-suite", type=int, default=16)
    parser.add_argument("--heldout-episodes-per-suite", type=int, default=8)
    parser.add_argument("--seed", type=int, default=62017)
    args = parser.parse_args()
    source_rows = [
        json.loads(line)
        for line in args.source_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eligible = build_sequence_rows(source_rows, history_chunks=args.history_chunks)
    train, heldout = select_episode_disjoint(
        eligible,
        train_per_suite=args.train_per_suite,
        heldout_per_suite=args.heldout_per_suite,
        heldout_episodes_per_suite=args.heldout_episodes_per_suite,
        seed=args.seed,
    )
    train_episodes = {episode_key(row) for row in train}
    heldout_episodes = {episode_key(row) for row in heldout}
    if train_episodes & heldout_episodes:
        raise RuntimeError("C62 train/heldout episode leakage")
    train_payload, heldout_payload = jsonl(train), jsonl(heldout)
    root = args.output_root.resolve()
    train_path, heldout_path = root / "train.jsonl", root / "heldout.jsonl"
    plan = {
        "schema": PLAN_SCHEMA,
        "sequence_schema": SCHEMA,
        "seed": args.seed,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "train_manifest": str(train_path),
        "train_manifest_sha256": hashlib.sha256(train_payload).hexdigest(),
        "heldout_manifest": str(heldout_path),
        "heldout_manifest_sha256": hashlib.sha256(heldout_payload).hexdigest(),
        "eligible_rows": len(eligible),
        "train_rows": len(train),
        "heldout_rows": len(heldout),
        "train_episode_count": len(train_episodes),
        "heldout_episode_count": len(heldout_episodes),
        "episode_intersection": 0,
        "suite_train_rows": dict(Counter(str(row["suite"]) for row in train)),
        "suite_heldout_rows": dict(Counter(str(row["suite"]) for row in heldout)),
        "history_chunks": args.history_chunks,
        "replan": 8,
        "miniworld_action_group": 4,
        "canary_budget": {
            "world_size": 8,
            "steps": 100,
            "global_batch": 8,
            "training_samples": 800,
            "effective_epochs": 1.0,
        },
        "gates": {
            "parent_default_off_max_abs": 0.0,
            "parent_parameters_updated": False,
            "h3_source_requires_grad": False,
            "bridge_all_30_refiners_positive_gradient": True,
            "bridge_checkpoint_restore_max_abs": 0.0,
            "heldout_clean_mse_vs_shuffled_relative_improvement_min": 0.01,
            "heldout_clean_mse_vs_context_off_regression_max": 0.05,
            "heldout_shuffle_prediction_max_abs_min": 1e-5,
        },
        "permission_on_pass": "GO_BOUNDED_C62_ABLATION",
        "permission_on_fail": "NO_GO_C62_TRAINING",
        "claim_boundary": "Causal/optimizer canary only; never LIBERO effectiveness evidence.",
    }
    atomic_write(train_path, train_payload)
    atomic_write(heldout_path, heldout_payload)
    atomic_write(root / "PLAN.json", canonical_json(plan))
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
