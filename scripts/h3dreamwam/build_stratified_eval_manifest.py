#!/usr/bin/env python3
"""Select a deterministic, balanced evaluation manifest by task."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def digest(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-task", type=int, default=1)
    parser.add_argument("--salt", default="h3dreamwam-uniform-val-v1-2026-08-08")
    args = parser.parse_args()
    if args.per_task <= 0:
        raise ValueError("--per-task must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation manifest: {output}")
    rows = [
        json.loads(line)
        for line in args.input.resolve().read_text(encoding="utf-8").splitlines()
        if line
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)
    selected: list[dict] = []
    for task, task_rows in sorted(grouped.items()):
        ranked = sorted(task_rows, key=lambda row: digest(args.salt, task, row["id"]))
        if len(ranked) < args.per_task:
            raise ValueError(f"task has too few validation rows: {task!r}")
        selected.extend(ranked[: args.per_task])
    selected.sort(key=lambda row: digest(args.salt, "schedule", row["id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    suites = Counter(str(row["suite"]) for row in selected)
    episodes = {(str(row["suite"]), int(row["episode"])) for row in selected}
    print(
        json.dumps(
            {
                "event": "stratified_eval_manifest_complete",
                "rows": len(selected),
                "tasks": len(grouped),
                "per_task": args.per_task,
                "episodes": len(episodes),
                "suites": dict(sorted(suites.items())),
                "salt": args.salt,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
