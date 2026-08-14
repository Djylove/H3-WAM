#!/usr/bin/env python3
"""Build deterministic task-stratified episode-disjoint train/val manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def episode_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["dataset_root"]), str(row["suite"]), int(row["episode"])


def _stable_score(salt: str, key: tuple[str, str, int]) -> str:
    return hashlib.sha256(
        f"{salt}|{key[0]}|{key[1]}|{key[2]}".encode("utf-8")
    ).hexdigest()


def split_rows(
    rows: list[dict[str, Any]], *, val_fraction: float, salt: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    if not rows:
        raise ValueError("source manifest is empty")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("source manifest contains duplicate window ids")

    episode_rows: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episode_rows[episode_key(row)].append(row)
    task_episodes: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    episode_task: dict[tuple[str, str, int], str] = {}
    for key, group in episode_rows.items():
        tasks = {str(row["task"]) for row in group}
        if len(tasks) != 1:
            raise ValueError(f"episode {key} maps to multiple tasks: {sorted(tasks)}")
        task = next(iter(tasks))
        episode_task[key] = task
        task_episodes[task].append(key)

    val_episodes: set[tuple[str, str, int]] = set()
    for task, keys in sorted(task_episodes.items()):
        if len(keys) < 2:
            raise ValueError(f"task {task!r} has fewer than two episodes")
        count = max(1, int(round(len(keys) * val_fraction)))
        count = min(count, len(keys) - 1)
        ranked = sorted(keys, key=lambda key: (_stable_score(salt, key), key))
        val_episodes.update(ranked[:count])

    train_rows = [row for row in rows if episode_key(row) not in val_episodes]
    val_rows = [row for row in rows if episode_key(row) in val_episodes]
    train_episodes = {episode_key(row) for row in train_rows}
    observed_val_episodes = {episode_key(row) for row in val_rows}
    overlap = train_episodes & observed_val_episodes
    if overlap:
        raise RuntimeError(f"episode leakage detected: {sorted(overlap)[:5]}")
    source_ids = set(ids)
    if {str(row["id"]) for row in train_rows} | {
        str(row["id"]) for row in val_rows
    } != source_ids:
        raise RuntimeError("split does not partition all source window ids")

    def coverage(selected: list[dict[str, Any]]) -> dict[str, Any]:
        episodes = {episode_key(row) for row in selected}
        return {
            "windows": len(selected),
            "episodes": len(episodes),
            "tasks": len({str(row["task"]) for row in selected}),
            "suites": dict(sorted(Counter(str(row["suite"]) for row in selected).items())),
            "windows_by_task": dict(sorted(Counter(str(row["task"]) for row in selected).items())),
            "episodes_by_task": dict(
                sorted(Counter(episode_task[key] for key in episodes).items())
            ),
        }

    audit = {
        "salt": salt,
        "val_fraction_requested": val_fraction,
        "episode_key": ["dataset_root", "suite", "episode"],
        "source": coverage(rows),
        "train": coverage(train_rows),
        "validation": coverage(val_rows),
        "episode_overlap": 0,
        "window_id_overlap": len(
            {str(row["id"]) for row in train_rows}
            & {str(row["id"]) for row in val_rows}
        ),
        "all_source_windows_partitioned_once": True,
        "all_tasks_present_in_both_splits": (
            coverage(train_rows)["tasks"]
            == coverage(val_rows)["tasks"]
            == coverage(rows)["tasks"]
        ),
    }
    return train_rows, val_rows, audit


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--salt", default="h3-int8-starwam-episode-val-v1-2026-08-14")
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.source_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_rows, val_rows, audit = split_rows(
        rows, val_fraction=args.val_fraction, salt=args.salt
    )
    train_path = args.output_dir / "manifest_train_episode_disjoint.jsonl"
    val_path = args.output_dir / "manifest_val_episode_disjoint.jsonl"
    _atomic_write_jsonl(train_path, train_rows)
    _atomic_write_jsonl(val_path, val_rows)
    audit.update(
        {
            "source_manifest": str(args.source_manifest.resolve()),
            "source_manifest_sha256": sha256_file(args.source_manifest),
            "source_manifest_items": len(rows),
            "train_manifest": str(train_path.resolve()),
            "train_manifest_sha256": sha256_file(train_path),
            "validation_manifest": str(val_path.resolve()),
            "validation_manifest_sha256": sha256_file(val_path),
        }
    )
    audit_path = args.output_dir / "episode_disjoint_split_audit.json"
    temporary = audit_path.with_name(f".{audit_path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, audit_path)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
