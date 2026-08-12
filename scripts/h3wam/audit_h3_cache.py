#!/usr/bin/env python3
"""Audit that every manifest row has one finite, schema-compatible H3 cache artifact."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cache_root = args.cache_root.resolve()
    files = {path.stem for path in (cache_root / "windows").glob("*.pt")}
    identifiers = {str(row["id"]) for row in rows}
    if files != identifiers:
        raise ValueError(
            f"cache/manifest mismatch: extra={len(files - identifiers)}, "
            f"missing={len(identifiers - files)}"
        )
    shapes: collections.Counter = collections.Counter()
    suites: collections.Counter = collections.Counter()
    nonfinite = []
    total_bytes = 0
    for row in rows:
        path = cache_root / "windows" / f"{row['id']}.pt"
        total_bytes += path.stat().st_size
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        shapes[
            (
                tuple(artifact["video_latents"].shape),
                tuple(artifact["first_frame_latents"].shape),
                tuple(artifact["actions"].shape),
                tuple(artifact["state"].shape),
            )
        ] += 1
        suites[str(artifact["suite"])] += 1
        if not all(
            bool(torch.isfinite(artifact[key]).all())
            for key in ("video_latents", "first_frame_latents", "actions", "state")
        ):
            nonfinite.append(str(row["id"]))
    stats = torch.load(cache_root / "stats.pt", map_location="cpu", weights_only=False)
    if int(stats["num_windows"]) != len(rows):
        raise ValueError("normalization stats window count does not match manifest")
    if nonfinite:
        raise ValueError(f"non-finite cache artifacts: {nonfinite[:10]}")
    report = {
        "event": "complete",
        "windows": len(rows),
        "bytes": total_bytes,
        "nonfinite": len(nonfinite),
        "shapes": [
            {"schema": [list(shape) for shape in schema], "windows": count}
            for schema, count in sorted(shapes.items())
        ],
        "suites": dict(sorted(suites.items())),
        "stats_windows": int(stats["num_windows"]),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
