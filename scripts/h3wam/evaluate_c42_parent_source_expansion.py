#!/usr/bin/env python3
"""Audit C41+C42 fixed-parent sources and their predeclared split roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")
TASKS = tuple(range(10))
TRAIN_TRIALS = (12, 13, 14, 16, 17)
FINAL_TRIALS = (15, 18, 19, 20, 21)
ALL_TRIALS = tuple(sorted(TRAIN_TRIALS + FINAL_TRIALS))
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

    c41_path = workspace / "eval/c41-fresh-parent-sources-v1/COMPLETED"
    c41 = json.loads(c41_path.read_text())
    if c41.get("status") != "PASS_C41_FRESH_PARENT_SOURCE_EXPANSION":
        raise ValueError("C41 source expansion did not pass")
    if tuple(sorted({int(row["trial"]) for row in c41["rows"]})) != (12, 13, 14, 15):
        raise ValueError("C41 trial inventory mismatch")

    for trial in range(16, 22):
        marker = workspace / "eval/c42-fresh-parent-sources-v1" / f"trial{trial}.COMPLETED"
        if not marker.is_file():
            raise FileNotFoundError(marker)

    rows = []
    for trial in ALL_TRIALS:
        role = "ranker_train_source" if trial in TRAIN_TRIALS else "future_final_source"
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
                    "suite": suite,
                    "task": task,
                    "trial": trial,
                    "success": bool(episode["success"]),
                    "steps": int(episode["steps"]),
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "role": role,
                })

    success_by_role = Counter(row["role"] for row in rows if row["success"])
    success_suites: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["success"]:
            success_suites[row["role"]][row["suite"]] += 1
    train_role, final_role = "ranker_train_source", "future_final_source"
    gate = {
        "all_400_results_present": len(rows) == 400,
        "ranker_train_success_sources_at_least_36": success_by_role[train_role] >= 36,
        "future_final_success_sources_at_least_36": success_by_role[final_role] >= 36,
        "ranker_train_successes_cover_at_least_3_suites": len(success_suites[train_role]) >= 3,
        "future_final_successes_cover_at_least_3_suites": len(success_suites[final_role]) >= 3,
        "trial_roles_fixed_before_c42_outcomes": True,
    }
    passed = all(gate.values())
    report = {
        "format": "h3wam-c42-fresh-parent-source-expansion-v1",
        "parent": "D0-H32-s14000/replan8/no-ensemble",
        "selection": (
            "complete suites/tasks for trials12..21; train trials12,13,14,16,17; "
            "final trials15,18,19,20,21"
        ),
        "episodes": len(rows),
        "successes": sum(success_by_role.values()),
        "successes_by_role": dict(sorted(success_by_role.items())),
        "successes_by_role_and_suite": {
            role: dict(sorted(counts.items())) for role, counts in sorted(success_suites.items())
        },
        "gate": gate,
        "status": (
            "PASS_C42_FRESH_PARENT_SOURCE_EXPANSION"
            if passed else "FAIL_C42_FRESH_PARENT_SOURCE_EXPANSION"
        ),
        "permission": "GO_C43_COUNTERFACTUAL_COLLECTION" if passed else "NO_GO_C43_COUNTERFACTUAL_COLLECTION",
        "c41_completed": str(c41_path),
        "c41_completed_sha256": sha256_file(c41_path),
        "rows": rows,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        key: report[key] for key in (
            "status", "permission", "episodes", "successes", "successes_by_role",
            "successes_by_role_and_suite", "gate",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
