#!/usr/bin/env python3
"""Build a task-balanced canary from currently completed dense cache files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--sparse-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=128)
    parser.add_argument(
        "--allow-sparse-fill",
        action="store_true",
        help=(
            "Fill a task's quota with old sparse windows only after using every "
            "currently cached new-dense window for that task."
        ),
    )
    parser.add_argument("--salt", default="h3wam-dense-canary-v1-2026-08-12")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def digest(salt: str, row: dict) -> str:
    return hashlib.sha256(f"{salt}\x1f{row['id']}".encode()).hexdigest()


def task_key(row: dict) -> tuple[str, str]:
    return str(row["suite"]), str(row["task"])


def cached_ids(cache: Path) -> set[str]:
    return {path.stem for path in (cache / "windows").glob("*.pt")}


def interleave(grouped: dict[tuple[str, str], list[dict]]) -> list[dict]:
    keys = sorted(grouped)
    return [grouped[key][offset] for offset in range(len(grouped[keys[0]])) for key in keys]


def main() -> None:
    args = parse_args()
    if args.per_task <= 0:
        raise ValueError("--per-task must be positive")
    base = args.base_root.resolve()
    cache = args.cache_root.resolve()
    sparse = args.sparse_cache_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")

    available_ids = cached_ids(cache)
    sparse_ids = cached_ids(sparse)
    train = [
        row
        for row in read_jsonl(base / "manifest_train.jsonl")
        if str(row["id"]) in available_ids
    ]
    validation = [
        row
        for row in read_jsonl(base / "manifest_val.jsonl")
        if str(row["id"]) in available_ids
    ]
    dense_grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    sparse_grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in train:
        # The canary must measure added temporal coverage, not replay the old
        # five-window-per-episode sparse population.
        if str(row["id"]) in sparse_ids:
            sparse_grouped[task_key(row)].append(row)
        else:
            dense_grouped[task_key(row)].append(row)
    all_tasks = sorted({task_key(row) for row in train + validation})
    available_counts = {
        key: len(dense_grouped[key])
        + (len(sparse_grouped[key]) if args.allow_sparse_fill else 0)
        for key in all_tasks
    }
    missing = [key for key in all_tasks if available_counts[key] < args.per_task]
    if missing:
        counts = {"/".join(key): available_counts[key] for key in missing}
        raise RuntimeError(f"dense cache is not canary-ready: {counts}")
    selected = {}
    for key in all_tasks:
        dense_ranked = sorted(
            dense_grouped[key], key=lambda row: digest(args.salt, row)
        )
        selected[key] = dense_ranked[: args.per_task]
        if len(selected[key]) < args.per_task and args.allow_sparse_fill:
            sparse_ranked = sorted(
                sparse_grouped[key], key=lambda row: digest(args.salt + "-sparse", row)
            )
            selected[key].extend(
                sparse_ranked[: args.per_task - len(selected[key])]
            )
    train_rows = interleave(selected)

    validation_grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in validation:
        validation_grouped[task_key(row)].append(row)
    if any(not validation_grouped[key] for key in all_tasks):
        raise RuntimeError("dense cache lacks a validation window for at least one task")
    val_rows = [
        min(validation_grouped[key], key=lambda row: digest(args.salt + "-val", row))
        for key in all_tasks
    ]

    output.mkdir(parents=True)
    write_jsonl(output / "manifest_train_uniform.jsonl", train_rows)
    write_jsonl(output / "manifest_train.jsonl", train_rows)
    write_jsonl(output / "manifest_val_stratified40.jsonl", val_rows)
    shutil.copyfile(base / "task_contexts.json", output / "task_contexts.json")
    report = {
        "schema_version": 1,
        "status": "canary_not_frozen",
        "tasks": len(all_tasks),
        "train_windows": len(train_rows),
        "train_windows_per_task": args.per_task,
        "validation_windows": len(val_rows),
        "sampling": "task_round_robin_new_dense_only",
        "new_dense_train_windows": sum(
            str(row["id"]) not in sparse_ids
            for row in train_rows
        ),
        "suite_counts": dict(sorted(Counter(row["suite"] for row in train_rows).items())),
        "normalization": "sparse full40 train min/max (provisional; same control and treatment)",
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    (output / "candidate_report.json").write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
