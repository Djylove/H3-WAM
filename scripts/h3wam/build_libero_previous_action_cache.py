#!/usr/bin/env python3
"""Build a compact previous-action sidecar for sparse/dense LIBERO windows."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", type=Path, nargs="+")
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


def main() -> None:
    args = parse_args()
    import pandas as pd

    rows = read_rows(args.manifests)
    if not rows:
        raise ValueError("manifests are empty")
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("dataset_root"):
            raise ValueError(f"manifest row {row['id']} has no dataset_root")
        groups[(str(Path(row["dataset_root"]).resolve()), int(row["episode"]))].append(row)

    values: dict[str, torch.Tensor] = {}
    initial = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    for (dataset_root, episode), episode_rows in sorted(groups.items()):
        parquet = (
            Path(dataset_root)
            / "data/chunk-000"
            / f"episode_{episode:06d}.parquet"
        )
        table = pd.read_parquet(parquet, columns=["action"])
        for row in episode_rows:
            start = int(row["start"])
            if start < 0 or start >= len(table):
                raise ValueError(f"invalid start {start} for {row['id']}")
            value = (
                initial.clone()
                if start == 0
                else torch.from_numpy(table["action"].iloc[start - 1].copy()).float()
            )
            if tuple(value.shape) != (7,) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"bad previous action for {row['id']}")
            values[str(row["id"])] = value

    ids = sorted(values)
    payload = {
        "format": "h3wam-libero-previous-action-v1",
        "ids": ids,
        "values": torch.stack([values[sample_id] for sample_id in ids]),
        "manifests": [str(path.resolve()) for path in args.manifests],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite previous-action cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "windows": len(ids),
                "episodes": len(groups),
                "bytes": output.stat().st_size,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
