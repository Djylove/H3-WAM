#!/usr/bin/env python3
"""Build a task-balanced LIBERO window manifest for H3-WAM experiments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--windows-per-episode", type=int, default=5)
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Include every valid start instead of evenly spaced starts.",
    )
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument(
        "--task",
        action="append",
        help="Exact task name to include; may be repeated. Default includes all tasks.",
    )
    return parser.parse_args()


def evenly_spaced_starts(max_start: int, count: int) -> list[int]:
    if max_start < 0:
        return []
    if count <= 0:
        raise ValueError("windows-per-episode must be positive")
    if count == 1 or max_start == 0:
        return [0]
    return sorted({round(index * max_start / (count - 1)) for index in range(count)})


def main() -> None:
    args = parse_args()
    if args.episodes_per_task <= 0:
        raise ValueError("episodes-per-task must be positive")
    root = args.dataset_root.resolve()
    grouped: dict[str, list[dict]] = defaultdict(list)
    selected_tasks = None if args.task is None else set(args.task)
    with (root / "meta/episodes.jsonl").open() as handle:
        for line in handle:
            episode = json.loads(line)
            task = episode["tasks"][0]
            if selected_tasks is not None and task not in selected_tasks:
                continue
            grouped[task].append(episode)

    if not grouped:
        raise ValueError("no dataset tasks matched --task filters")

    windows = []
    for task_index, task in enumerate(sorted(grouped)):
        episodes = sorted(grouped[task], key=lambda item: item["episode_index"])
        for episode in episodes[: args.episodes_per_task]:
            max_start = int(episode["length"]) - (args.action_horizon + 1)
            starts = (
                range(max_start + 1)
                if args.dense
                else evenly_spaced_starts(max_start, args.windows_per_episode)
            )
            for start in starts:
                episode_index = int(episode["episode_index"])
                windows.append(
                    {
                        "id": f"ep{episode_index:06d}_s{start:06d}",
                        "episode": episode_index,
                        "start": start,
                        "length": int(episode["length"]),
                        "task": task,
                        "task_group": task_index,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for window in windows:
            handle.write(json.dumps(window, ensure_ascii=False) + "\n")
    summary = {
        "output": str(args.output.resolve()),
        "tasks": len(grouped),
        "selected_episodes": len({item["episode"] for item in windows}),
        "windows": len(windows),
        "action_horizon": args.action_horizon,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
