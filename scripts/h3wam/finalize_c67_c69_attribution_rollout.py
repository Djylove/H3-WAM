#!/usr/bin/env python3
"""Seal all distributed C67/C69 rollout shards without inspecting success."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "COMPLETED.json"
    if args.num_shards <= 0 or output.exists() or (root / "INVALID.json").exists():
        raise ValueError("invalid or already finalized C67/C69 rollout root")
    authorization_path = root / "AUTHORIZATION.json"
    manifest_path = root / "jobs.jsonl"
    authorization = json.loads(authorization_path.read_text())
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    authorization_sha = sha256_file(authorization_path)
    if (
        authorization.get("format")
        != "h3wam-c67-c69-paired-rollout-authorization-v1"
        or authorization.get("status")
        != "AUTHORIZED_C67_C69_FIXED_S20_PAIRED_680"
        or len(rows) != 1_360
        or sha256_file(manifest_path) != authorization.get("manifest_sha256")
    ):
        raise ValueError("C67/C69 authorization or manifest mismatch")
    markers = []
    for shard in range(args.num_shards):
        path = root / f"SHARD_{shard:02d}_COMPLETE.json"
        marker = json.loads(path.read_text())
        expected_jobs = sum(
            row["pair_id"] % args.num_shards == shard for row in rows
        )
        if (
            marker.get("format")
            != "h3wam-c67-c69-paired680-shard-complete-v1"
            or marker.get("status") != "COMPLETE"
            or marker.get("shard_index") != shard
            or marker.get("num_shards") != args.num_shards
            or marker.get("jobs") != expected_jobs
            or marker.get("pairs") != expected_jobs // 2
            or marker.get("authorization_sha256") != authorization_sha
            or marker.get("manifest_sha256") != authorization["manifest_sha256"]
        ):
            raise ValueError(f"C67/C69 shard marker mismatch: {path}")
        markers.append({"path": str(path), "sha256": sha256_file(path)})
    seen = set()
    for row in rows:
        result_path = Path(row["output"]) / "results.json"
        result = json.loads(result_path.read_text())
        episodes = [
            episode
            for task in result.get("tasks", [])
            for episode in task.get("episodes", [])
        ]
        key = (row["arm"], row["suite"], row["tasks"][0], row["trials"][0])
        if (
            key in seen
            or len(episodes) != 1
            or result.get("task_ids") != row["tasks"]
            or result.get("trial_indices") != row["trials"]
            or result.get("c67_c69_attribution_authorization_sha256")
            != authorization_sha
        ):
            raise ValueError(f"invalid C67/C69 isolated result: {result_path}")
        seen.add(key)
    if len(seen) != 1_360:
        raise ValueError("C67/C69 completed result grid mismatch")
    report = {
        "format": "h3wam-c67-c69-paired680-isolated-complete-v1",
        "status": "COMPLETE",
        "jobs": 1_360,
        "pairs": 680,
        "episodes_per_arm": 680,
        "num_shards": args.num_shards,
        "authorization_sha256": authorization_sha,
        "manifest_sha256": authorization["manifest_sha256"],
        "shard_markers": markers,
        "success_counts_inspected": False,
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({key: report[key] for key in ("status", "jobs", "pairs")}))


if __name__ == "__main__":
    main()
