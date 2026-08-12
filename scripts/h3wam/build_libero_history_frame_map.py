#!/usr/bin/env python3
"""Map each cached LIBERO window to an earlier cached observation window."""

from __future__ import annotations

import argparse
import bisect
import json
import os
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--history-offset", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for path in paths:
        for line in path.resolve().read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["id"])
            previous = by_id.setdefault(sample_id, row)
            if previous != row:
                raise ValueError(f"conflicting duplicate manifest row: {sample_id}")
    return list(by_id.values())


def build_history_sources(rows: list[dict], history_offset: int) -> dict[str, dict]:
    if history_offset <= 0:
        raise ValueError("history-offset must be positive")
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("dataset_root"):
            raise ValueError(f"manifest row {row['id']} has no dataset_root")
        key = (str(Path(row["dataset_root"]).resolve()), int(row["episode"]))
        groups[key].append(row)

    result: dict[str, dict] = {}
    for episode_rows in groups.values():
        ordered = sorted(episode_rows, key=lambda row: (int(row["start"]), str(row["id"])))
        starts = [int(row["start"]) for row in ordered]
        for row in ordered:
            current_start = int(row["start"])
            target_start = max(0, current_start - history_offset)
            source_index = max(0, bisect.bisect_right(starts, target_start) - 1)
            source = ordered[source_index]
            source_start = int(source["start"])
            if source_start > current_start:
                raise ValueError(f"history source is in the future for {row['id']}")
            result[str(row["id"])] = {
                "source_id": str(source["id"]),
                "current_start": current_start,
                "source_start": source_start,
            }
    return result


def main() -> None:
    args = parse_args()
    rows = read_rows(args.manifests)
    if not rows:
        raise ValueError("manifests are empty")
    mapping = build_history_sources(rows, args.history_offset)
    ids = sorted(mapping)
    payload = {
        "format": "h3wam-libero-history-frame-map-v1",
        "ids": ids,
        "source_ids": [mapping[sample_id]["source_id"] for sample_id in ids],
        "current_starts": torch.tensor(
            [mapping[sample_id]["current_start"] for sample_id in ids],
            dtype=torch.int32,
        ),
        "source_starts": torch.tensor(
            [mapping[sample_id]["source_start"] for sample_id in ids],
            dtype=torch.int32,
        ),
        "history_offset": int(args.history_offset),
        "manifests": [str(path.resolve()) for path in args.manifests],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite history-frame map: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    actual_lags = payload["current_starts"] - payload["source_starts"]
    print(
        json.dumps(
            {
                "output": str(output),
                "windows": len(ids),
                "episodes": len(
                    {
                        (str(row["dataset_root"]), int(row["episode"]))
                        for row in rows
                    }
                ),
                "history_offset": args.history_offset,
                "nonzero_history_windows": int((actual_lags > 0).sum()),
                "mean_actual_lag": float(actual_lags.float().mean()),
                "bytes": output.stat().st_size,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
