#!/usr/bin/env python3
"""Freeze the causal replan-8 sequence contract used by C57.

The source manifest contains one window per episode start.  For each current
decision this builder references only preceding replan-8 decisions, their
eight executed actions, and observations available at four-action intervals.
No tensors are copied: every reference resolves to an already-audited window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA = "c57_lingbot_replan8_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def episode_key(row: dict) -> tuple[str, int]:
    return str(row["suite"]), int(row["episode"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--replan", type=int, default=8)
    parser.add_argument("--observe-every", type=int, default=4)
    parser.add_argument("--max-history-chunks", type=int, default=7)
    args = parser.parse_args()
    if args.replan <= 0 or args.observe_every <= 0:
        raise ValueError("replan and observe-every must be positive")
    if args.replan % args.observe_every:
        raise ValueError("replan must be divisible by observe-every")

    raw_lines = [
        line for line in args.source_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [json.loads(line) for line in raw_lines]
    by_episode_start: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
    ids: set[str] = set()
    for row in rows:
        key, start, row_id = episode_key(row), int(row["start"]), str(row["id"])
        if row_id in ids or start in by_episode_start[key]:
            raise ValueError(f"duplicate source row: {row_id}")
        ids.add(row_id)
        by_episode_start[key][start] = row

    output_rows = []
    missing_references = 0
    leakage = 0
    history_chunks = Counter()
    observation_frames = Counter()
    executed_actions = Counter()
    suites = Counter()
    max_tokens = 0
    for row in rows:  # preserve the frozen source ordering
        key = episode_key(row)
        start = int(row["start"])
        # Anchor the grid to the *current* decision.  Episode windows may start
        # at any integer, so a grid anchored at zero can cross the current
        # boundary (for example 0,8,...,40 for current start 46).
        decisions = [
            start - args.replan * distance
            for distance in range(args.max_history_chunks, 0, -1)
            if start - args.replan * distance >= 0
        ]
        history = []
        for history_index, decision_start in enumerate(decisions):
            action_row = by_episode_start[key].get(decision_start)
            if action_row is None:
                missing_references += 1
                continue
            observation_starts = list(
                range(decision_start + args.observe_every, decision_start + args.replan + 1, args.observe_every)
            )
            if history_index == 0:
                observation_starts.insert(0, decision_start)
            observation_rows = [by_episode_start[key].get(value) for value in observation_starts]
            if any(item is None for item in observation_rows):
                missing_references += 1
                continue
            action_indices = list(range(decision_start, decision_start + args.replan))
            if max(action_indices) >= start or max(observation_starts) > start:
                leakage += 1
            history.append(
                {
                    "decision_start": decision_start,
                    "action_source_id": str(action_row["id"]),
                    "action_indices": action_indices,
                    "observation_source_ids": [str(item["id"]) for item in observation_rows],
                    "observation_starts": observation_starts,
                }
            )
        if len(history) != len(decisions):
            # A contiguous source manifest is a hard C57 input contract.
            raise ValueError(f"missing causal source references for {row['id']}")
        num_observations = sum(len(item["observation_source_ids"]) for item in history)
        num_actions = len(history) * args.replan
        max_tokens = max(max_tokens, num_observations * 32 + num_actions)
        history_chunks[len(history)] += 1
        observation_frames[num_observations] += 1
        executed_actions[num_actions] += 1
        suites[str(row["suite"])] += 1
        enriched = dict(row)
        enriched.update(
            {
                "sequence_schema": SCHEMA,
                "current_id": str(row["id"]),
                "history": history,
                "history_chunks": len(history),
                "history_observation_frames": num_observations,
                "history_executed_actions": num_actions,
            }
        )
        output_rows.append(enriched)

    if missing_references or leakage:
        raise RuntimeError(
            f"sequence gate failed: missing={missing_references}, leakage={leakage}"
        )
    manifest_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in output_rows
    ).encode()
    audit = {
        "schema": SCHEMA,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "rows": len(output_rows),
        "episodes": len(by_episode_start),
        "replan": args.replan,
        "observe_every": args.observe_every,
        "max_history_chunks": args.max_history_chunks,
        "max_observation_frames": max(observation_frames, default=0),
        "max_executed_actions": max(executed_actions, default=0),
        "max_persistent_tokens_per_layer": max_tokens,
        "token_capacity": 15 * (32 + 4),
        "missing_references": missing_references,
        "future_leakage": leakage,
        "history_chunk_histogram": dict(sorted(history_chunks.items())),
        "observation_frame_histogram": dict(sorted(observation_frames.items())),
        "executed_action_histogram": dict(sorted(executed_actions.items())),
        "suite_rows": dict(sorted(suites.items())),
        "gate": "PASS" if max_tokens <= 15 * (32 + 4) else "FAIL",
    }
    if audit["gate"] != "PASS":
        raise RuntimeError(f"persistent token capacity exceeded: {max_tokens}")
    # Fail closed: a capacity-invalid candidate must never replace a prior
    # audited manifest merely because validation happens after serialization.
    atomic_write(args.output_manifest, manifest_payload)
    atomic_write(args.audit, (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
