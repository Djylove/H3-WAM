#!/usr/bin/env python3
"""Audit all fixed-parent LIBERO trial12..15 source episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")
TRIALS = (12, 13, 14, 15)
TASKS = tuple(range(10))
CHECKPOINT_NAME = "d0_h32_s14000"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/mnt/h3-wam"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    rows = []
    for trial in TRIALS:
        marker = workspace / "eval/c41-fresh-parent-sources-v1" / f"trial{trial}.COMPLETED"
        if not marker.is_file():
            raise FileNotFoundError(marker)
        for suite in SUITES:
            for task in TASKS:
                slug = suite.removeprefix("libero_")
                path = workspace / "outputs/eval-dense-d0-long" / (
                    f"{CHECKPOINT_NAME}_{slug}_task{task}_trial{trial}_replan8/results.json"
                )
                payload = json.loads(path.read_text())
                episode = payload["tasks"][0]["episodes"][0]
                if int(episode["trial"]) != trial:
                    raise ValueError(f"trial mismatch: {path}")
                rows.append({
                    "suite": suite, "task": task, "trial": trial,
                    "success": bool(episode["success"]), "steps": int(episode["steps"]),
                    "path": str(path), "sha256": sha256_file(path),
                    "role": "ranker_train_source" if trial < 15 else "future_final_source",
                })
    successes = Counter(row["suite"] for row in rows if row["success"])
    gate = {
        "all_160_results_present": len(rows) == 160,
        "at_least_32_success_sources": sum(successes.values()) >= 32,
        "goal_at_least_4": successes["libero_goal"] >= 4,
        "object_at_least_10": successes["libero_object"] >= 10,
        "spatial_at_least_6": successes["libero_spatial"] >= 6,
        "trial15_reserved_before_counterfactual_outcomes": True,
    }
    passed = all(gate.values())
    report = {
        "format": "h3wam-c41-fresh-parent-source-expansion-v1",
        "parent": "D0-H32-s14000/replan8/no-ensemble",
        "selection": "all suites/tasks, trials12..15; trials12..14 train-source pool, trial15 reserved final-source pool",
        "episodes": len(rows), "successes": sum(successes.values()),
        "successes_by_suite": dict(sorted(successes.items())), "gate": gate,
        "status": "PASS_C41_FRESH_PARENT_SOURCE_EXPANSION" if passed else "FAIL_C41_FRESH_PARENT_SOURCE_EXPANSION",
        "permission": "GO_C41_COUNTERFACTUAL_COLLECTION" if passed else "NO_GO_C41_COUNTERFACTUAL_COLLECTION",
        "rows": rows,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({k: report[k] for k in ("status", "permission", "episodes", "successes", "successes_by_suite", "gate")}, indent=2))


if __name__ == "__main__":
    main()
