#!/usr/bin/env python3
"""Read-only progress summary for an in-flight C69/C58b direct paired eval."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ARMS = ("c69_action_only", "c58b_fastwam")


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("_direct_progress_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load aggregation base: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-script", type=Path, required=True)
    parser.add_argument(
        "--verify-initial-state",
        action="store_true",
        help="hash paired trajectories and compare initial object joints",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    base = load_base(args.base_script.resolve())
    jobs = [
        json.loads(line)
        for line in (root / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    completed: dict[tuple[str, str, int, int], dict] = {}
    invalid_results: list[str] = []
    arm_counts = {arm: 0 for arm in ARMS}
    arm_successes = {arm: 0 for arm in ARMS}
    shard_counts = {str(index): 0 for index in range(5)}

    for job in jobs:
        result = Path(job["output"]) / "results.json"
        if not result.is_file():
            continue
        try:
            payload = json.loads(result.read_text(encoding="utf-8"))
            task, trial = job["tasks"][0], job["trials"][0]
            episode = base.episode_map(payload)[(task, trial)]
            row = {
                "success": bool(episode["success"]),
                "initial_object_joints": episode["initial_object_joints"],
                "trajectory": Path(episode["trajectory"]),
            }
        except Exception as error:  # Keep monitoring robust to an atomic write in flight.
            invalid_results.append(f"{result}: {error}")
            continue
        key = (job["arm"], job["suite"], task, trial)
        completed[key] = row
        arm_counts[job["arm"]] += 1
        arm_successes[job["arm"]] += int(row["success"])
        shard_counts[str(job["pair_id"] % 5)] += 1

    pairs: list[dict] = []
    state_mismatches: list[dict] = []
    for suite in base.SUITES:
        for task in range(10):
            for trial in range(33, 50):
                c69 = completed.get((ARMS[0], suite, task, trial))
                c58b = completed.get((ARMS[1], suite, task, trial))
                if c69 is None or c58b is None:
                    continue
                if args.verify_initial_state:
                    c69_digest = base.initial_state_digest(c69["trajectory"])
                    c58b_digest = base.initial_state_digest(c58b["trajectory"])
                    if c69_digest != c58b_digest or not base.same_object_joints(
                        c69["initial_object_joints"], c58b["initial_object_joints"]
                    ):
                        state_mismatches.append(
                            {"suite": suite, "task": task, "trial": trial}
                        )
                        continue
                pairs.append(
                    {
                        "suite": suite,
                        "task": task,
                        "trial": trial,
                        "candidate": c69["success"],
                        "control": c58b["success"],
                    }
                )

    overall = base.paired_summary(pairs) if pairs else None
    per_suite = {
        suite: base.paired_summary([row for row in pairs if row["suite"] == suite])
        for suite in base.SUITES
        if any(row["suite"] == suite for row in pairs)
    }
    print(
        json.dumps(
            {
                "format": "h3wam-c69-c58b-direct-progress-v1",
                "status": "COMPLETE" if (root / "COMPLETED.json").is_file() else "RUNNING",
                "complete_jobs": sum(arm_counts.values()),
                "total_jobs": len(jobs),
                "completion_fraction": sum(arm_counts.values()) / len(jobs),
                "arm_counts": arm_counts,
                "arm_successes_unpaired": arm_successes,
                "shard_counts": shard_counts,
                "valid_complete_pairs": len(pairs),
                "initial_state_verification": args.verify_initial_state,
                "initial_state_mismatches": state_mismatches,
                "invalid_results": invalid_results,
                "partial_overall": overall,
                "partial_per_suite": per_suite,
                "shard_markers": len(list(root.glob("SHARD_*_COMPLETE.json"))),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
