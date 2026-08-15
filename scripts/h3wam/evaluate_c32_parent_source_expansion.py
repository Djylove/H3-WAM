#!/usr/bin/env python3
"""Audit fixed-parent LIBERO trial8..11 sources after the C30 val shortfall."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")
TRIALS = (8, 9, 10, 11)
TASKS = tuple(range(10))
CHECKPOINT_NAME = "d0_h32_s14000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/mnt/h3-wam"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def result_path(root: Path, suite: str, task: int, trial: int) -> Path:
    slug = suite.removeprefix("libero_")
    return root / f"{CHECKPOINT_NAME}_{slug}_task{task}_trial{trial}_replan8" / "results.json"


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    result_root = workspace / "outputs" / "eval-dense-d0-long"
    rows = []
    for trial in TRIALS:
        marker = workspace / "eval/c32-fresh-parent-sources-v1" / f"trial{trial}.COMPLETED"
        if not marker.is_file():
            raise FileNotFoundError(f"missing trial marker {marker}")
        for suite in SUITES:
            for task in TASKS:
                path = result_path(result_root, suite, task, trial)
                payload = json.loads(path.read_text(encoding="utf-8"))
                task_rows = payload.get("tasks")
                if not isinstance(task_rows, list) or len(task_rows) != 1:
                    raise ValueError(f"unexpected task result contract: {path}")
                episodes = task_rows[0].get("episodes")
                if not isinstance(episodes, list) or len(episodes) != 1:
                    raise ValueError(f"unexpected episode result contract: {path}")
                episode = episodes[0]
                if int(episode["trial"]) != trial:
                    raise ValueError(f"trial mismatch: {path}")
                rows.append({
                    "suite": suite, "task": task, "trial": trial,
                    "success": bool(episode["success"]), "steps": int(episode["steps"]),
                    "path": str(path), "sha256": sha256_file(path),
                })
    if len(rows) != 160 or len({(r["suite"], r["task"], r["trial"]) for r in rows}) != 160:
        raise ValueError("C32 must contain exactly 160 unique episodes")
    successes = Counter(row["suite"] for row in rows if row["success"])
    gate = {
        "all_160_results_present": len(rows) == 160,
        "at_least_32_success_sources": sum(successes.values()) >= 32,
        "goal_at_least_4": successes["libero_goal"] >= 4,
        "object_at_least_10": successes["libero_object"] >= 10,
        "spatial_at_least_6": successes["libero_spatial"] >= 6,
    }
    passed = all(gate.values())
    report = {
        "format": "h3wam-c32-fresh-parent-source-expansion-v1",
        "parent": "D0-H32-s14000/replan8/no-ensemble",
        "selection": "all four LIBERO suites, tasks0..9, benchmark init-state trials8..11",
        "freshness": "trial indices 8..11 are absent from every C21..C30 causal branch",
        "episodes": len(rows), "successes": sum(successes.values()),
        "successes_by_suite": dict(sorted(successes.items())), "gate": gate,
        "status": "PASS_C32_FRESH_PARENT_SOURCE_EXPANSION" if passed else "FAIL_C32_FRESH_PARENT_SOURCE_EXPANSION",
        "permission": "GO_FRESH_RANKING_VALIDATION_COLLECTION" if passed else "NO_GO_FRESH_RANKING_VALIDATION_COLLECTION",
        "effect_conclusion": "NOT_A_MODEL_COMPARISON",
        "rows": rows,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({key: report[key] for key in (
        "status", "permission", "episodes", "successes", "successes_by_suite", "gate"
    )}, indent=2))


if __name__ == "__main__":
    main()
