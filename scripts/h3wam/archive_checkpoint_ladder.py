#!/usr/bin/env python3
"""Archive completed checkpoints before a training run rotates them away."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument(
        "--idle-polls-after-complete",
        type=int,
        default=2,
        help="Exit after this many polls once a complete event has been observed.",
    )
    return parser.parse_args()


def checkpoint_events(train_log: Path) -> tuple[list[Path], bool]:
    checkpoints: list[Path] = []
    complete = False
    if not train_log.is_file():
        return checkpoints, complete
    for line in train_log.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "checkpoint" and event.get("path"):
            checkpoints.append(Path(event["path"]))
        elif event.get("event") == "complete":
            complete = True
    return checkpoints, complete


def archive_checkpoint(source: Path, archive_dir: Path) -> bool:
    required = (source / "manifest.json", source / "action_head.pt")
    if not all(path.is_file() for path in required):
        return False
    destination = archive_dir / source.name
    if destination.is_dir():
        return False
    temporary = archive_dir / f".{source.name}.partial"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    temporary.rename(destination)
    return True


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0 or args.idle_polls_after_complete <= 0:
        raise ValueError("poll arguments must be positive")
    train_log = args.train_log.resolve()
    archive_dir = args.archive_dir.resolve()
    archive_dir.mkdir(parents=True, exist_ok=True)
    complete_polls = 0
    while complete_polls < args.idle_polls_after_complete:
        sources, complete = checkpoint_events(train_log)
        for source in sources:
            if archive_checkpoint(source, archive_dir):
                print(f"archived {source.name}", flush=True)
        complete_polls = complete_polls + 1 if complete else 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
