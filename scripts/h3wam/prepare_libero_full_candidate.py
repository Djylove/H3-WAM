#!/usr/bin/env python3
"""Build deterministic train/validation windows over all four LIBERO suites."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from build_libero_manifest import evenly_spaced_starts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="SUITE=PATH",
        help="LIBERO suite name and extracted LeRobot root; repeat for all suites.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows-per-episode", type=int, default=5)
    parser.add_argument(
        "--dense",
        action="store_true",
        help=(
            "Include every valid action-window start, matching FastWAM's "
            "frame-indexed RobotVideoDataset instead of subsampling a fixed "
            "number of windows per episode."
        ),
    )
    parser.add_argument(
        "--frame-indexed",
        action="store_true",
        help=(
            "Use every raw frame as a window start and mark tail windows for "
            "FastWAM-compatible padding/masked loss."
        ),
    )
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--all-train",
        action="store_true",
        help="Match official FastWAM LIBERO: use every episode for training.",
    )
    parser.add_argument("--split-salt", default="h3wam-libero-full-v1-2026-08-06")
    return parser.parse_args()


def parse_datasets(values: list[str]) -> dict[str, Path]:
    datasets = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"dataset must be SUITE=PATH, got {value!r}")
        suite, raw_path = value.split("=", 1)
        if not suite or not raw_path or suite in datasets:
            raise ValueError(f"invalid or duplicate dataset: {value!r}")
        path = Path(raw_path).resolve()
        if not (path / "meta/episodes.jsonl").is_file():
            raise FileNotFoundError(f"invalid LeRobot dataset root: {path}")
        datasets[suite] = path
    return datasets


def stable_digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.dense and args.frame_indexed:
        raise ValueError("--dense and --frame-indexed are mutually exclusive")
    if args.windows_per_episode <= 0 or args.action_horizon <= 0:
        raise ValueError("window count and action horizon must be positive")
    if not args.all_train and not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be in (0, 0.5)")
    datasets = parse_datasets(args.dataset)
    episodes_by_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for suite, root in sorted(datasets.items()):
        for line in (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines():
            episode = json.loads(line)
            task = str(episode["tasks"][0])
            episodes_by_task[(suite, task)].append(episode)

    validation_episode_keys: set[tuple[str, int]] = set()
    if not args.all_train:
        for (suite, task), episodes in sorted(episodes_by_task.items()):
            ranked = sorted(
                episodes,
                key=lambda item: stable_digest(
                    args.split_salt, suite, task, int(item["episode_index"])
                ),
            )
            validation_count = max(1, round(len(ranked) * args.validation_fraction))
            validation_episode_keys.update(
                (suite, int(item["episode_index"]))
                for item in ranked[:validation_count]
            )

    all_rows: list[dict] = []
    contexts: dict[str, str] = {}
    for (suite, task), episodes in sorted(episodes_by_task.items()):
        context_id = f"task_{stable_digest(task)[:16]}"
        contexts[context_id] = task
        root = datasets[suite]
        for episode in sorted(episodes, key=lambda item: int(item["episode_index"])):
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            max_start = length - (args.action_horizon + 1)
            if args.frame_indexed:
                starts = range(length)
            elif args.dense:
                starts = range(max_start + 1)
            else:
                starts = evenly_spaced_starts(max_start, args.windows_per_episode)
            for start in starts:
                all_rows.append(
                    {
                        "id": f"{suite}_ep{episode_index:06d}_s{start:06d}",
                        "suite": suite,
                        "dataset_root": str(root),
                        "episode": episode_index,
                        "start": start,
                        "length": length,
                        "task": task,
                        "context_id": context_id,
                        "split": (
                            "validation"
                            if (suite, episode_index) in validation_episode_keys
                            else "train"
                        ),
                        "padded_tail": start > max_start,
                    }
                )
    train_rows = [row for row in all_rows if row["split"] == "train"]
    validation_rows = [row for row in all_rows if row["split"] == "validation"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "manifest_all.jsonl", all_rows)
    write_jsonl(output_dir / "manifest_train.jsonl", train_rows)
    write_jsonl(output_dir / "manifest_val.jsonl", validation_rows)
    (output_dir / "task_contexts.json").write_text(
        json.dumps(contexts, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "datasets": {suite: str(root) for suite, root in sorted(datasets.items())},
        "tasks": len(episodes_by_task),
        "episodes": sum(len(items) for items in episodes_by_task.values()),
        "validation_episodes": len(validation_episode_keys),
        "windows": len(all_rows),
        "train_windows": len(train_rows),
        "validation_windows": len(validation_rows),
        "windows_per_episode": args.windows_per_episode,
        "window_sampling": (
            "frame_indexed_padded"
            if args.frame_indexed
            else ("dense" if args.dense else "evenly_spaced")
        ),
        "action_horizon": args.action_horizon,
        "validation_fraction": args.validation_fraction,
        "all_train": args.all_train,
        "split_salt": args.split_salt,
    }
    (output_dir / "candidate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
