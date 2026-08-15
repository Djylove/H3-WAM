#!/usr/bin/env python3
"""Audit incumbent rollout trajectories for a FACT-style value dataset.

This script does not relabel failures or create training examples.  It proves
which causal transitions exist and records the missing failure-onset contract
before any consequence/value training is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


RESULT_PATTERN = re.compile(
    r"(?P<prefix>.+)_(?P<suite>goal|object|spatial|10)_task(?P<task>\d+)_"
    r"trial(?P<trial>\d+)_replan(?P<replan>\d+)/results\.json$"
)
REQUIRED_TRAJECTORY_KEYS = {
    "step",
    "agentview_image",
    "wristview_image",
    "eef_pos",
    "eef_quat",
    "gripper_qpos",
    "previous_action",
    "policy_actions",
    "sim_state",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--prefix", default="d0_h32_s14000")
    parser.add_argument("--replan", type=int, default=8)
    parser.add_argument("--validation-trial", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    records = []
    identity_rows = []
    for result_path in sorted(root.glob("*/results.json")):
        match = RESULT_PATTERN.fullmatch(f"{result_path.parent.name}/results.json")
        if match is None:
            continue
        fields = match.groupdict()
        if fields["prefix"] != args.prefix or int(fields["replan"]) != args.replan:
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        task = payload["tasks"][0]
        episode = task["episodes"][0]
        trajectory_path = Path(episode["trajectory"]).resolve()
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"missing trajectory: {trajectory_path}")
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            missing = REQUIRED_TRAJECTORY_KEYS - set(trajectory.files)
            if missing:
                raise ValueError(f"{trajectory_path} misses keys: {sorted(missing)}")
            replans = int(trajectory["step"].shape[0])
            shapes = {key: list(trajectory[key].shape) for key in sorted(trajectory.files)}
            if trajectory["policy_actions"].shape != (replans, 32, 7):
                raise ValueError(f"unexpected action chunks: {trajectory_path}")
            if trajectory["previous_action"].shape != (replans, 7):
                raise ValueError(f"unexpected previous actions: {trajectory_path}")
            if trajectory["agentview_image"].shape[0] != replans:
                raise ValueError(f"agentview/replan mismatch: {trajectory_path}")
            if trajectory["sim_state"].shape[0] != replans:
                raise ValueError(f"sim-state/replan mismatch: {trajectory_path}")
        trial = int(fields["trial"])
        split = "val" if trial == args.validation_trial else "train"
        record = {
            "suite": fields["suite"],
            "task_id": int(fields["task"]),
            "trial": trial,
            "split": split,
            "success": bool(episode["success"]),
            "replans": replans,
            "causal_transitions": max(0, replans - 1),
            "trajectory": str(trajectory_path),
            "trajectory_bytes": trajectory_path.stat().st_size,
            "shapes": shapes,
        }
        records.append(record)
        identity_rows.append(
            json.dumps(
                {
                    "result": str(result_path.resolve()),
                    "result_sha256": sha256_bytes(result_path.read_bytes()),
                    "trajectory": str(trajectory_path),
                    "trajectory_bytes": trajectory_path.stat().st_size,
                    "shapes": shapes,
                },
                sort_keys=True,
            )
        )
    if not records:
        raise ValueError("no matching rollout trajectories")

    split_report = {}
    for split in ("train", "val"):
        items = [record for record in records if record["split"] == split]
        split_report[split] = {
            "episodes": len(items),
            "successful_episodes": sum(record["success"] for record in items),
            "failed_episodes": sum(not record["success"] for record in items),
            "replan_states": sum(record["replans"] for record in items),
            "causal_transitions": sum(record["causal_transitions"] for record in items),
            "by_suite": dict(sorted(Counter(record["suite"] for record in items).items())),
            "tasks": len({(record["suite"], record["task_id"]) for record in items}),
            "trials": sorted({record["trial"] for record in items}),
        }
    overlap = {
        (record["suite"], record["task_id"], record["trial"])
        for record in records
        if record["split"] == "train"
    } & {
        (record["suite"], record["task_id"], record["trial"])
        for record in records
        if record["split"] == "val"
    }
    by_suite_success = defaultdict(lambda: {"episodes": 0, "successes": 0})
    for record in records:
        item = by_suite_success[record["suite"]]
        item["episodes"] += 1
        item["successes"] += int(record["success"])

    report = {
        "format": "h3wam-fact-rollout-transition-audit-v1",
        "source": {
            "project": "FACT",
            "revision": "618a6c16868699b6d4138941de6a863589ac00dd",
            "code_contract": (
                "value is normalized time-to-go; failure_active receives a penalty; "
                "best-of-N selects argmin predicted value"
            ),
        },
        "result_root": str(root),
        "prefix": args.prefix,
        "replan_steps": args.replan,
        "episodes": len(records),
        "successes": sum(record["success"] for record in records),
        "replan_states": sum(record["replans"] for record in records),
        "causal_transitions": sum(record["causal_transitions"] for record in records),
        "trajectory_bytes": sum(record["trajectory_bytes"] for record in records),
        "by_suite": dict(sorted(by_suite_success.items())),
        "split": split_report,
        "split_overlap": len(overlap),
        "identity_manifest_sha256": sha256_bytes("\n".join(identity_rows).encode()),
        "available_fields": sorted(REQUIRED_TRAJECTORY_KEYS),
        "contract_gates": {
            "next_observation_transition": "PASS",
            "executed_action_chunk": "PASS_POLICY_CHUNK_AND_PREVIOUS_ACTION_RECORDED",
            "simulator_episode_success": "PASS",
            "failure_onset": "UNKNOWN",
            "counterfactual_action_outcome": "MISSING",
            "failure_imitation": "NO_GO",
            "value_only_diagnostic": "GO_CANARY_AFTER_FEATURE_AND_TARGET_DOSSIER",
            "best_of_n_rollout": "NO_GO_UNTIL_HELD_OUT_VALUE_RANKING_PASS",
        },
        "evidence_boundary": (
            "Episode-level failure is known, but causal failure onset and outcomes for "
            "alternative sampled actions are not. These trajectories can support a "
            "value-only diagnostic; they cannot yet reproduce FACT failure-aware action training."
        ),
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite FACT audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
