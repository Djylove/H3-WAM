#!/usr/bin/env python3
"""Aggregate the C57 train and strict-restore gates into one launch record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--restore", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = json.loads(args.train.read_text())
    restore = json.loads(args.restore.read_text())
    history = train.get("history", [])
    losses = [float(item["loss"]) for item in history]
    seconds = [float(item["seconds"]) for item in history]
    updates = [float(item["head_update_max_abs"]) for item in history]
    completed_steps = int(train.get("completed_steps", train.get("steps", -1)))
    passed = (
        completed_steps == 10
        and len(history) == 10
        and all(math.isfinite(value) for value in losses + seconds + updates)
        and min(updates) > 0
        and restore.get("status") == "PASS"
        and int(restore.get("completed_steps", -1)) == 10
        and float(restore.get("restore_max_abs", -1)) == 0.0
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "completed_steps": completed_steps,
        "restore_max_abs": float(restore.get("restore_max_abs", -1)),
        "loss_min": min(losses),
        "loss_max": max(losses),
        "seconds_per_step_min": min(seconds),
        "seconds_per_step_max": max(seconds),
        "seconds_per_step_mean": sum(seconds) / len(seconds),
        "head_update_min": min(updates),
        "training_samples": int(train["training_samples"]),
        "peak_allocated_gib": float(train["peak_allocated_gib"]),
        "train_report_sha256": sha(args.train),
        "restore_report_sha256": sha(args.restore),
        "checkpoint_sha256": restore["checkpoint_sha256"],
    }
    if not passed:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
