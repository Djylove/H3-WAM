#!/usr/bin/env python3
"""Freeze paired first-action execution-horizon challengers from C23."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c23-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    c23 = json.loads((args.c23_root / "COMPLETED").read_text())
    if c23["status"] != "PASS_FIRST_ACTION_CAUSAL_CANARY":
        raise ValueError("C24 requires a passed C23 parent")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    shutil.copyfile(args.c23_root / "selection.jsonl", output / "selection.jsonl")
    for horizon in (16, 32):
        (output / f"h{horizon}" / "runs").mkdir(parents=True)
        (output / f"h{horizon}" / "logs").mkdir()
    preregistration = {
        "format": "h3wam-c24-first-action-horizon-sweep-v1",
        "falsifiable_hypothesis": (
            "With identical state, first chunks, first/continuation seeds, and later replan8, "
            "executing the first chunk for 16 or 32 steps yields at least three mixed-outcome "
            "groups spanning at least two suites for one candidate."
        ),
        "parent": "C23 first-action-only causal canary with first execution horizon8",
        "candidates": [16, 32],
        "only_variable": "number of actions executed from the first predicted chunk",
        "fixed_continuation": "all later replans execute 8 actions and reuse C23 seed schedules",
        "groups_per_candidate": 8,
        "branches_per_candidate": 32,
        "total_branches": 64,
        "max_environment_steps": 25_600,
        "promotion": (
            "mechanical identity and seed gates pass, and one candidate has >=3 mixed groups "
            "covering >=2 suites"
        ),
        "pass_permission": "GO_CAUSAL_DATASET_WITH_SELECTED_EXECUTION_HORIZON",
        "fail_permission": "NO_GO_LONGER_FIRST_ACTION_LABEL_EXPANSION",
        "effect_conclusion": "NOT_EVIDENCE_READY",
        "c23_artifact": str((args.c23_root / "COMPLETED").resolve()),
    }
    atomic_json(output / "preregistration.json", preregistration)
    print(json.dumps({"output_root": str(output), "total_branches": 64}, indent=2))


if __name__ == "__main__":
    main()
