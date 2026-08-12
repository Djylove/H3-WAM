#!/usr/bin/env python3
"""Check that a training manifest exactly covers the standard 40 LIBERO tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from libero.libero import benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_tasks = {str(row["task"]) for row in rows}
    benchmark_tasks = set()
    suites = {}
    for suite_name in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        suite = benchmark.get_benchmark_dict()[suite_name]()
        tasks = {suite.get_task(index).language for index in range(suite.n_tasks)}
        suites[suite_name] = len(tasks)
        benchmark_tasks.update(tasks)
    missing = sorted(benchmark_tasks - manifest_tasks)
    extra = sorted(manifest_tasks - benchmark_tasks)
    if missing or extra:
        raise ValueError(f"LIBERO task mismatch: missing={missing}, extra={extra}")
    print(
        json.dumps(
            {
                "event": "complete",
                "manifest_tasks": len(manifest_tasks),
                "benchmark_tasks": len(benchmark_tasks),
                "suites": suites,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
