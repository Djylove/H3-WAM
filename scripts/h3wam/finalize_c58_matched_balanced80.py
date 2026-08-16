#!/usr/bin/env python3
"""Apply the pre-registered C58/full30 versus matched-D0 offline gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MILESTONES = tuple(range(1_000, 10_001, 1_000))
SELECTED_IDS_SHA256 = "26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_identity(payload: dict[str, Any], *, candidate: str, step: int) -> None:
    if payload.get("candidate") != candidate:
        raise ValueError(f"step{step} candidate mismatch: {payload.get('candidate')}")
    checkpoint = payload.get("checkpoint", {})
    contract = checkpoint.get("contract", {})
    if int(checkpoint.get("completed_steps", -1)) != step:
        raise ValueError(f"step{step} checkpoint milestone mismatch")
    if contract.get("candidate") != candidate:
        raise ValueError(f"step{step} checkpoint contract candidate mismatch")
    if contract.get("d0_parent_optimizer_restored") is not False:
        raise ValueError(f"step{step} did not use a fresh parent optimizer")
    if float(checkpoint.get("fresh_restore", {}).get("max_abs", -1.0)) != 0.0:
        raise ValueError(f"step{step} fresh restore is not exact")
    protocol = payload.get("protocol_identity", {})
    if protocol.get("actual_selected_ids_sha256") != SELECTED_IDS_SHA256:
        raise ValueError(f"step{step} balanced-80 IDs changed")


def summarize_pair(candidate: dict[str, Any], control: dict[str, Any], step: int) -> dict[str, Any]:
    require_identity(candidate, candidate="C58_FASTWAM_FULL30_H3_LAYER49", step=step)
    require_identity(control, candidate="C58_MATCHED_D0_FRESH_OPTIMIZER", step=step)
    c_metrics = candidate["metrics"]
    d_metrics = control["metrics"]
    values = {
        "candidate": {
            "normalized_action_mse": float(
                c_metrics["normalized_clip5_model_domain"]["action_mse"]
            ),
            "physical_action_mse": float(
                c_metrics["denormalized_official_minmax_clamp"]["action_mse"]
            ),
            "gripper_macro_f1": float(c_metrics["gripper_sign"]["macro_f1"]),
            "language_mean_abs_delta": float(
                c_metrics["language_replacement_sensitivity"][
                    "mean_abs_prediction_delta"
                ]
            ),
            "visual_shuffle_delta_mse": float(
                c_metrics["visual_feature_shuffle"][
                    "baseline_vs_shuffle_action_delta"
                ]["normalized_model_domain"]["action_mse"]
            ),
        },
        "control": {
            "normalized_action_mse": float(
                d_metrics["normalized_clip5_model_domain"]["action_mse"]
            ),
            "physical_action_mse": float(
                d_metrics["denormalized_official_minmax_clamp"]["action_mse"]
            ),
            "gripper_macro_f1": float(d_metrics["gripper_sign"]["macro_f1"]),
            "language_mean_abs_delta": float(
                d_metrics["language_replacement_sensitivity"][
                    "mean_abs_prediction_delta"
                ]
            ),
            "visual_shuffle_delta_mse": float(
                d_metrics["visual_feature_shuffle"][
                    "baseline_vs_shuffle_action_delta"
                ]["normalized_model_domain"]["action_mse"]
            ),
        },
    }
    c = values["candidate"]
    d = values["control"]
    gates = {
        "normalized_action_mse_improves_at_least_1pct": (
            c["normalized_action_mse"] <= d["normalized_action_mse"] * 0.99
        ),
        "physical_action_mse_improves_at_least_1pct": (
            c["physical_action_mse"] <= d["physical_action_mse"] * 0.99
        ),
        "gripper_macro_f1_drop_at_most_0_005": (
            c["gripper_macro_f1"] >= d["gripper_macro_f1"] - 0.005
        ),
        "language_sensitivity_retains_95pct_and_at_least_0_01": (
            c["language_mean_abs_delta"]
            >= max(0.01, d["language_mean_abs_delta"] * 0.95)
        ),
        "visual_sensitivity_retains_95pct_and_at_least_1e_4": (
            c["visual_shuffle_delta_mse"]
            >= max(1.0e-4, d["visual_shuffle_delta_mse"] * 0.95)
        ),
    }
    return {
        "step": step,
        **values,
        "candidate_relative_improvement": {
            "normalized_action_mse": 1.0
            - c["normalized_action_mse"] / d["normalized_action_mse"],
            "physical_action_mse": 1.0
            - c["physical_action_mse"] / d["physical_action_mse"],
            "gripper_macro_f1_delta": c["gripper_macro_f1"]
            - d["gripper_macro_f1"],
            "language_sensitivity_ratio": c["language_mean_abs_delta"]
            / d["language_mean_abs_delta"],
            "visual_sensitivity_ratio": c["visual_shuffle_delta_mse"]
            / d["visual_shuffle_delta_mse"],
        },
        "gates": gates,
        "eligible_for_fresh_closed_loop": all(gates.values()),
    }


def finalize(root: Path) -> dict[str, Any]:
    milestones = []
    sources = {}
    for step in MILESTONES:
        candidate_path = root / "candidate" / f"step{step}.balanced80.json"
        control_path = root / "control" / f"step{step}.balanced80.json"
        if not candidate_path.is_file() or not control_path.is_file():
            break
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        control = json.loads(control_path.read_text(encoding="utf-8"))
        summary = summarize_pair(candidate, control, step)
        summary["source_sha256"] = {
            "candidate": sha256_file(candidate_path),
            "control": sha256_file(control_path),
        }
        milestones.append(summary)
        sources[f"step{step}_candidate"] = summary["source_sha256"]["candidate"]
        sources[f"step{step}_control"] = summary["source_sha256"]["control"]
    complete = len(milestones) == len(MILESTONES)
    eligible = [item for item in milestones if item["eligible_for_fresh_closed_loop"]]
    selected = min(
        eligible,
        key=lambda item: (item["candidate"]["physical_action_mse"], item["step"]),
        default=None,
    )
    permission = (
        "WAIT_REMAINING_MILESTONES"
        if not complete
        else ("GO_FRESH_PAIRED_LIBERO" if selected else "NO_GO_C58_CLOSED_LOOP")
    )
    return {
        "format": "h3wam-c58-matched-balanced80-final-v1",
        "status": "COMPLETE" if complete else "PARTIAL",
        "effect_status": "NOT_CLOSED_LOOP_EVIDENCE",
        "permission": permission,
        "offline_gate": {
            "action_mse_relative_improvement_minimum": 0.01,
            "gripper_macro_f1_maximum_drop": 0.005,
            "condition_sensitivity_minimum_retention": 0.95,
            "language_absolute_minimum": 0.01,
            "visual_delta_mse_absolute_minimum": 1.0e-4,
            "selection_rule": "eligible milestone with lowest C58 physical MSE; tie -> earlier step",
        },
        "closed_loop_promotion_gate_preregistered_before_results": {
            "protocol": "fresh LIBERO trials33..49, same initial state/environment seed/policy noise, replan8/no ensemble",
            "episodes_per_arm": 680,
            "candidate_vs_matched_control": {
                "success_rate_delta_minimum_percentage_points": 3.0,
                "paired_net_wins_minimum": 20,
                "one_sided_exact_mcnemar_p_maximum": 0.05,
                "maximum_suite_degradation_percentage_points": 3.0,
            },
            "candidate_vs_incumbent_d0": {
                "overall_success_rate_must_not_be_lower": True,
                "maximum_suite_degradation_percentage_points": 3.0,
            },
        },
        "selected": selected,
        "milestones": milestones,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize(args.root.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({key: report[key] for key in ("status", "permission", "selected")}, sort_keys=True))


if __name__ == "__main__":
    main()
