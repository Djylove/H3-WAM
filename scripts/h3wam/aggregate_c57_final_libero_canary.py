#!/usr/bin/env python3
"""Audit paired C57/D0 canary outputs and publish one atomic result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def audit_trace(path: Path) -> dict[str, Any]:
    traces = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("event") == "c57_persistent_trace":
            traces.append(row)
    commands = [row.get("command") for row in traces]
    if not commands or commands[0] != "c57_reset" or commands.count("c57_reset") != 1:
        raise ValueError(f"C57 trace does not begin with exactly one reset: {path}")
    index = 1
    committed = 0
    terminal_tail = "none"
    while index < len(traces):
        if traces[index].get("command") != "predict":
            raise ValueError(f"C57 trace expected predict at index {index}: {path}")
        lifecycle = traces[index].get("lifecycle")
        if lifecycle != "reset_predict_obs4_commit8":
            raise ValueError(f"C57 predict lacks persistent lifecycle declaration: {path}")
        index += 1
        if index == len(traces):
            terminal_tail = "predict_without_obs4"
            break
        obs4 = traces[index]
        if obs4.get("command") != "c57_feedback" or int(obs4.get("action_count", -1)) != 4 or obs4.get("committed") is not False:
            raise ValueError(f"C57 trace expected non-committing obs4 at index {index}: {path}")
        index += 1
        if index == len(traces):
            terminal_tail = "obs4_without_commit8"
            break
        commit8 = traces[index]
        if commit8.get("command") != "c57_feedback" or int(commit8.get("action_count", -1)) != 8 or commit8.get("committed") is not True:
            raise ValueError(f"C57 trace expected commit8 at index {index}: {path}")
        committed += 1
        index += 1
    if committed == 0:
        raise ValueError(f"C57 canary did not complete one persistent transaction: {path}")
    return {
        "trace_lines": len(traces),
        "predicts": commands.count("predict"),
        "obs4": sum(row.get("command") == "c57_feedback" and row.get("action_count") == 4 for row in traces),
        "commit8": committed,
        "terminal_tail": terminal_tail,
        "status": "PASS_RESET_PREDICT_OBS4_COMMIT8",
    }


def require_rollout(result: dict[str, Any], pair: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    expected = {
        "policy": "h3_dreamwam_kv_int8",
        "suite": pair["suite"],
        "task_ids": [pair["task_id"]],
        "trial_indices": [pair["trial_index"]],
        "trials_per_task": 1,
        "max_steps": 400,
        "replan_steps": 8,
        "action_horizon": 32,
        "wait_steps": 0,
        "environment_seed": pair["environment_seed"],
        "policy_noise_seed_base": pair["policy_noise_seed_base"],
        "normalized_action_pre_clamp": True,
        "use_action_ensembler": False,
        "model_evaluations": 10,
        "save_trajectories": False,
    }
    mismatch = {key: (result.get(key), value) for key, value in expected.items() if result.get(key) != value}
    if Path(result.get("checkpoint", "")).resolve() != checkpoint:
        mismatch["checkpoint"] = (result.get("checkpoint"), str(checkpoint))
    if mismatch:
        raise ValueError(f"rollout contract mismatch: {mismatch}")
    if int(result.get("episodes", -1)) != 1 or int(result.get("successes", -1)) not in (0, 1):
        raise ValueError("paired canary result is incomplete")
    return {
        "success": int(result["successes"]),
        "success_rate": float(result["success_rate"]),
        "duration_seconds": float(result["duration_seconds"]),
    }


def aggregate(plan_path: Path, root: Path, c57_checkpoint: Path, d0_checkpoint: Path) -> dict[str, Any]:
    plan = load_json(plan_path)
    if plan.get("schema") != "c57_final_fresh_libero_canary_v1" or int(plan.get("decision_checkpoint_step", -1)) != 5000:
        raise ValueError("C57 fresh canary plan identity mismatch")
    pairs = plan.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 4 or len({row["suite"] for row in pairs}) != 4:
        raise ValueError("C57 fresh canary must contain one pair from each of four suites")
    records = []
    for pair in pairs:
        slug = f"{pair['suite']}_task{pair['task_id']}_trial{pair['trial_index']}"
        d0_dir = root / slug / "d0"
        c57_dir = root / slug / "c57"
        d0_path = d0_dir / "results.json"
        c57_path = c57_dir / "results.json"
        d0 = require_rollout(load_json(d0_path), pair, d0_checkpoint)
        c57 = require_rollout(load_json(c57_path), pair, c57_checkpoint)
        trace = audit_trace(c57_dir / "policy_server.log")
        records.append({
            **pair,
            "d0": d0,
            "c57": c57,
            "c57_minus_d0_success": c57["success"] - d0["success"],
            "trace": trace,
            "artifacts": {
                "d0_results": str(d0_path.resolve()),
                "c57_results": str(c57_path.resolve()),
                "c57_policy_log": str((c57_dir / "policy_server.log").resolve()),
            },
        })
    d0_successes = sum(row["d0"]["success"] for row in records)
    c57_successes = sum(row["c57"]["success"] for row in records)
    return {
        "format": "h3wam-c57-final-fresh-libero-canary-results-v1",
        "status": "PASS_C57_FRESH_LIBERO_CANARY_COMPLETE",
        "effect_status": "NOT_EVIDENCE_READY",
        "next_permission": "GO_LARGER_PAIRED_ROLLOUT" if c57_successes > d0_successes else "HOLD_C57",
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "c57_checkpoint": str(c57_checkpoint),
        "c57_checkpoint_sha256": sha256_file(c57_checkpoint),
        "d0_checkpoint": str(d0_checkpoint),
        "d0_checkpoint_sha256": sha256_file(d0_checkpoint),
        "episodes_per_arm": len(records),
        "d0_successes": d0_successes,
        "c57_successes": c57_successes,
        "d0_success_rate": d0_successes / len(records),
        "c57_success_rate": c57_successes / len(records),
        "paired_success_delta": (c57_successes - d0_successes) / len(records),
        "pairs": records,
        "claim_boundary": (
            "This four-episode paired canary proves the real C57 persistent lifecycle and provides an early effect signal. "
            "A larger pre-registered paired LIBERO evaluation is required for a benchmark/generalization claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--c57-checkpoint", type=Path, required=True)
    parser.add_argument("--d0-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(
        args.plan.resolve(), args.root.resolve(),
        args.c57_checkpoint.resolve(), args.d0_checkpoint.resolve(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
