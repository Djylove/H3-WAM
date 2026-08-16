#!/usr/bin/env python3
"""Audit tri-arm C55 rollout mechanics without aggregating success outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


COMPARE_KEYS = (
    "step",
    "agentview_image",
    "wristview_image",
    "eef_pos",
    "eef_quat",
    "gripper_qpos",
    "previous_action",
    "sim_state",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.stage_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in (root / "jobs.jsonl").read_text().splitlines()]
    groups = defaultdict(dict)
    result_hashes = {}
    for row in rows:
        result_path = Path(row["output"]) / "results.json"
        payload = json.loads(result_path.read_text())
        episode = payload["tasks"][0]["episodes"][0]
        trajectory = Path(episode["trajectory"])
        archive = np.load(trajectory)
        if (
            payload["replan_steps"] != 8
            or payload["action_horizon"] != 32
            or payload["normalized_action_pre_clamp"] is not True
            or payload["trial_indices"] != [row["trial"]]
            or payload["task_ids"] != [row["task"]]
            or payload["suite"] != row["suite"]
            or Path(payload["checkpoint"]).resolve() != Path(row["checkpoint"]).resolve()
        ):
            raise ValueError(f"C55 rollout contract mismatch: {result_path}")
        key = (row["trial"], row["suite"], row["task"])
        groups[key][row["arm"]] = {
            "archive": archive,
            "episode_seed": episode["episode_seed"],
            "replan_noise_seeds": episode["replan_noise_seeds"],
        }
        result_hashes[str(result_path)] = sha256_file(result_path)
    for key, arms in groups.items():
        if set(arms) != {"d0_parent", "action_only", "joint_aux"}:
            raise ValueError(f"C55 tri-arm group incomplete: {key}")
        reference = arms["d0_parent"]
        for arm in ("action_only", "joint_aux"):
            candidate = arms[arm]
            if candidate["episode_seed"] != reference["episode_seed"]:
                raise ValueError(f"C55 episode seed mismatch: {key}/{arm}")
            shared = min(
                len(candidate["replan_noise_seeds"]),
                len(reference["replan_noise_seeds"]),
            )
            if (
                shared <= 0
                or candidate["replan_noise_seeds"][:shared]
                != reference["replan_noise_seeds"][:shared]
            ):
                raise ValueError(f"C55 policy noise schedule mismatch: {key}/{arm}")
            for name in COMPARE_KEYS:
                if not np.array_equal(
                    candidate["archive"][name][0], reference["archive"][name][0]
                ):
                    raise ValueError(f"C55 initial state mismatch: {key}/{arm}/{name}")
    report = {
        "format": "h3wam-c55-closed-loop-mechanical-audit-v1",
        "status": "PASS_C55_MECHANICAL_CANARY",
        "permission": "GO_C55_REMAINING_FRESH_TRIALS",
        "jobs": len(rows),
        "tri_arm_groups": len(groups),
        "initial_state_keys": list(COMPARE_KEYS),
        "effect_boundary": "Success fields exist in source results but were not accessed or aggregated by this audit.",
        "manifest_sha256": sha256_file(root / "jobs.jsonl"),
        "result_sha256": result_hashes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({key: report[key] for key in ("status", "permission", "jobs", "tri_arm_groups")}, sort_keys=True))


if __name__ == "__main__":
    main()
