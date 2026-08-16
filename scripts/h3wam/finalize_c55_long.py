#!/usr/bin/env python3
"""Apply the pre-registered C55 paired offline gate to all long milestones."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


STEPS = (1000, 2000, 3000, 4000, 5000, 6000)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_step(action: dict, joint: dict, mechanism: dict, baseline_h3: float) -> dict:
    action_metrics = action["metrics"]
    joint_metrics = joint["metrics"]
    consequence = mechanism["metrics"]
    action_norm = float(action_metrics["normalized_clip5_model_domain"]["action_mse"])
    joint_norm = float(joint_metrics["normalized_clip5_model_domain"]["action_mse"])
    action_physical = float(action_metrics["denormalized_official_minmax_clamp"]["action_mse"])
    joint_physical = float(joint_metrics["denormalized_official_minmax_clamp"]["action_mse"])
    action_gripper = float(action_metrics["gripper_sign"]["macro_f1"])
    joint_gripper = float(joint_metrics["gripper_sign"]["macro_f1"])
    h3_clean = float(consequence["future_h3_clean"])
    gates = {
        "normalized_action_mse_improves_at_least_1pct": joint_norm <= action_norm * 0.99,
        "physical_action_mse_improves_at_least_1pct": joint_physical <= action_physical * 0.99,
        "gripper_macro_f1_drop_at_most_0_005": joint_gripper >= action_gripper - 0.005,
        "future_h3_mse_improves_at_least_5pct_from_step10": h3_clean <= baseline_h3 * 0.95,
        "future_h3_shuffle_degradation_at_least_0_01": float(
            consequence["future_h3_shuffle_degradation"]
        ) >= 0.01,
    }
    return {
        "action_only": {
            "normalized_action_mse": action_norm,
            "physical_action_mse": action_physical,
            "gripper_macro_f1": action_gripper,
        },
        "joint_aux": {
            "normalized_action_mse": joint_norm,
            "physical_action_mse": joint_physical,
            "gripper_macro_f1": joint_gripper,
            "future_h3_clean_mse": h3_clean,
            "future_h3_shuffle_degradation": float(
                consequence["future_h3_shuffle_degradation"]
            ),
        },
        "relative_improvement": {
            "normalized_action_mse": 1.0 - joint_norm / action_norm,
            "physical_action_mse": 1.0 - joint_physical / action_physical,
            "future_h3_mse_from_step10": 1.0 - h3_clean / baseline_h3,
            "gripper_macro_f1_delta": joint_gripper - action_gripper,
        },
        "gates": gates,
        "eligible": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite C55 final report: {output}")
    step10_path = root / "mechanism" / "step10.json"
    step10 = json.loads(step10_path.read_text())
    baseline_h3 = float(step10["metrics"]["future_h3_clean"])
    rows = []
    sources = {"mechanism_step10": sha256_file(step10_path)}
    for step in STEPS:
        paths = {
            "action_only": root / "evaluations" / "action_only" / f"step{step}.balanced80.json",
            "joint_aux": root / "evaluations" / "joint_aux" / f"step{step}.balanced80.json",
            "mechanism": root / "mechanism" / f"step{step}.json",
        }
        payloads = {key: json.loads(path.read_text()) for key, path in paths.items()}
        summary = summarize_step(
            payloads["action_only"], payloads["joint_aux"], payloads["mechanism"], baseline_h3
        )
        summary["step"] = step
        summary["source_sha256"] = {key: sha256_file(path) for key, path in paths.items()}
        rows.append(summary)
        for key, path in paths.items():
            sources[f"step{step}_{key}"] = sha256_file(path)
    eligible = [row for row in rows if row["eligible"]]
    selected = min(
        eligible,
        key=lambda row: (row["joint_aux"]["physical_action_mse"], row["step"]),
        default=None,
    )
    report = {
        "format": "h3wam-c55-long-offline-final-v1",
        "status": "PASS_C55_OFFLINE_GATE" if selected else "FAIL_C55_OFFLINE_GATE",
        "permission": "GO_FRESH_CLOSED_LOOP" if selected else "NO_GO_C55_CLOSED_LOOP",
        "effect_status": "NOT_CLOSED_LOOP_EVIDENCE",
        "selection_rule": "eligible milestone with lowest joint physical action MSE; tie -> lower step",
        "step10_future_h3_clean_mse": baseline_h3,
        "selected": selected,
        "milestones": rows,
        "sources": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({key: report[key] for key in ("status", "permission", "selected")}, sort_keys=True))


if __name__ == "__main__":
    main()
