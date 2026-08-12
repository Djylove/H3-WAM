#!/usr/bin/env python3
"""Promote a rollout-validated mixture mode into an H3 action checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", type=int, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--successes", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.checkpoint.resolve()
    output = args.output.resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    num_modes = int(checkpoint.get("num_action_modes", 1))
    if not 0 <= args.mode < num_modes:
        raise ValueError(f"mode {args.mode} is outside checkpoint range [0, {num_modes})")
    if not 0 <= args.successes <= args.trials:
        raise ValueError("successes must be between zero and trials")
    checkpoint["recommended_action_mode"] = args.mode
    checkpoint["action_mode_promotion"] = {
        "method": "task_level_rollout_validation",
        "suite": args.suite,
        "task_id": args.task_id,
        "trials": args.trials,
        "successes": args.successes,
        "success_rate": args.successes / args.trials,
        "source_checkpoint": str(source),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(
        f"promoted mode {args.mode} ({args.successes}/{args.trials}) to {output}"
    )


if __name__ == "__main__":
    main()
