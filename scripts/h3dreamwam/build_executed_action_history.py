#!/usr/bin/env python3
"""Build compact per-episode action sidecars for rolling-history training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episodes = {
        (str(row["suite"]), int(row["episode"])): Path(row["dataset_root"])
        for row in rows
    }
    output = args.output_root.resolve()
    (output / "actions").mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total_actions = 0
    import pandas as pd

    for (suite, episode), root in sorted(episodes.items()):
        episode_name = f"episode_{episode:06d}"
        table = pd.read_parquet(
            root / "data/chunk-000" / f"{episode_name}.parquet",
            columns=["action"],
        )
        actions = torch.stack(
            [torch.from_numpy(value.copy()) for value in table["action"]]
        ).float()
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"unexpected actions for {suite}/{episode_name}: {actions.shape}")
        payload = {
            "schema_version": 1,
            "suite": suite,
            "episode": episode,
            "dataset_root": str(root),
            "actions": actions,
        }
        path = output / "actions" / f"{suite}_ep{episode:06d}.pt"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        digest.update(f"{suite}:{episode}:{len(actions)}\n".encode())
        total_actions += len(actions)

    report = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "windows": len(rows),
        "episodes": len(episodes),
        "actions": total_actions,
        "episode_length_sha256": digest.hexdigest(),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
