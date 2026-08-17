#!/usr/bin/env python3
"""Aggregate the pre-registered C60 s1k..s10k offline learning curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


FORMAT = "h3wam-c60-fact-milestone-balanced80-v1"
MILESTONES = tuple(range(1000, 10001, 1000))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def paired_delta(left: dict[str, Any], right: dict[str, Any], metric: str) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("milestone per-sample identities differ")
    left_wins = right_wins = ties = 0
    deltas = []
    for sample_id in sorted(left):
        first = float(left[sample_id][metric])
        second = float(right[sample_id][metric])
        deltas.append(second - first)
        if abs(first - second) <= 1e-12:
            ties += 1
        elif first < second:
            left_wins += 1
        else:
            right_wins += 1
    return {
        "left_wins": left_wins, "right_wins": right_wins, "ties": ties,
        "right_minus_left_mean": mean(deltas),
    }


def summarize(report: dict[str, Any]) -> dict[str, float | int | bool]:
    metrics = report["arm"]["metrics"]
    normalized = metrics["normalized_clip5_model_domain"]
    physical = metrics["denormalized_official_minmax_clamp"]
    gripper = metrics["gripper_sign"]
    language = metrics["language_replacement_end_to_end_h3_and_action"]
    visual = metrics["visual_feature_shuffle_baseline_delta"]
    return {
        "milestone": int(report["milestone"]),
        "normalized_action_mse": float(normalized["action_mse"]),
        "physical_action_mse": float(physical["action_mse"]),
        "prediction_std": float(normalized["prediction_std"]),
        "gripper_macro_f1": float(gripper["macro_f1"]),
        "language_mean_abs_delta": float(language["mean_abs_prediction_delta"]),
        "visual_shuffle_action_mse": float(visual["action_mse"]),
        "conditioning_pass": all(report["conditioning_gates"].values()),
    }


def aggregate(root: Path, training_root: Path) -> dict[str, Any]:
    root, training_root = root.resolve(), training_root.resolve()
    reports, report_hashes = {}, {}
    identity = None
    for step in MILESTONES:
        path = root / f"reports/s{step}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("format") != FORMAT
            or report.get("status") not in {
                "PASS_FIXED_BALANCED80", "FAIL_CONDITIONING_COLLAPSE"
            }
            or report.get("effect_status") != "DIAGNOSTIC_NOT_CHECKPOINT_SELECTION"
            or report.get("milestone") != step
            or report.get("arm", {}).get("checkpoint_completed_steps") != step
            or report.get("arm", {}).get("strict_fresh_restore", {}).get("max_abs") != 0.0
            or report.get("data", {}).get("selection", {}).get("selected_ids_sha256")
            != "26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42"
        ):
            raise ValueError(f"invalid milestone report: s{step}")
        current_identity = {
            "training_contract_sha256": report["training_contract_sha256"],
            "data": report["data"], "execution": report["execution"],
            "evaluated_ids_sha256": report["arm"]["evaluated_ids_sha256"],
        }
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError(f"milestone evaluation identity drift: s{step}")
        reports[step] = report
        report_hashes[str(step)] = sha256_file(path)

    curve = [summarize(reports[step]) for step in MILESTONES]
    training_curve = []
    for step in MILESTONES:
        audit_path = training_root / f"milestone-audit/s{step}.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("status") != "PASS_C56B_MILESTONE_STRICT_RESTORE"
            or audit.get("milestone") != step
            or audit.get("restore_max_abs") != 0.0
            or not all(audit.get("gate", {}).values())
        ):
            raise ValueError(f"training milestone audit failed: s{step}")
        training_curve.append({
            "milestone": step,
            "mean_loss_last1000": float(audit["mean_loss"]),
            "mean_action_loss_last1000": float(audit["mean_action_loss"]),
            "minimum_block_gradient": float(audit["minimum_block_gradient"]),
            "audit_sha256": sha256_file(audit_path),
        })

    by_step = {int(row["milestone"]): row for row in curve}
    mid_steps, late_steps = (4000, 5000, 6000), (8000, 9000, 10000)
    mid_physical = mean([float(by_step[s]["physical_action_mse"]) for s in mid_steps])
    late_physical = mean([float(by_step[s]["physical_action_mse"]) for s in late_steps])
    mid_normalized = mean([float(by_step[s]["normalized_action_mse"]) for s in mid_steps])
    late_normalized = mean([float(by_step[s]["normalized_action_mse"]) for s in late_steps])
    s5, s10 = by_step[5000], by_step[10000]
    gates = {
        "all_ten_reports_complete": len(reports) == 10,
        "all_milestones_strict_restore": len(training_curve) == 10,
        "all_conditioning_gates_pass": all(bool(row["conditioning_pass"]) for row in curve),
        "late_window_physical_not_worse_than_mid": late_physical <= mid_physical,
        "late_window_normalized_not_worse_than_mid": late_normalized <= mid_normalized,
        "s10_physical_not_worse_than_s5": (
            float(s10["physical_action_mse"]) <= float(s5["physical_action_mse"])
        ),
        "s10_normalized_not_worse_than_s5": (
            float(s10["normalized_action_mse"]) <= float(s5["normalized_action_mse"])
        ),
        "s10_gripper_within_0_005_of_s5": (
            float(s10["gripper_macro_f1"]) >= float(s5["gripper_macro_f1"]) - 0.005
        ),
        "s10_language_preserves_90pct_of_s5": (
            float(s10["language_mean_abs_delta"])
            >= 0.9 * float(s5["language_mean_abs_delta"])
        ),
        "s10_visual_preserves_90pct_of_s5": (
            float(s10["visual_shuffle_action_mse"])
            >= 0.9 * float(s5["visual_shuffle_action_mse"])
        ),
    }
    continuation_signal = all(gates.values())
    first_samples = reports[1000]["arm"]["per_sample"]
    final_samples = reports[10000]["arm"]["per_sample"]
    paired = {
        metric: paired_delta(first_samples, final_samples, metric)
        for metric in ("normalized_action_mse", "physical_action_mse")
    }
    return {
        "format": "h3wam-c60-fact-milestone-balanced80-curve-v1",
        "status": "PASS_COMPLETE_FIXED_CURVE",
        "effect_status": "DIAGNOSTIC_NOT_CLOSED_LOOP_PROMOTION",
        "training_permission": (
            "ELIGIBLE_TO_AUTHOR_S20K_DOSSIER"
            if continuation_signal else "NO_EVIDENCE_FOR_S20K_CONTINUATION"
        ),
        "hypothesis": (
            "If s10000 remains undertrained, fixed balanced80 action error should "
            "continue improving from the middle to late milestones without losing "
            "gripper, language, or visual conditioning."
        ),
        "milestones": list(MILESTONES),
        "evaluated_samples_per_milestone": 80,
        "total_model_sample_evaluations": 800,
        "identity": identity,
        "report_sha256": report_hashes,
        "offline_curve": curve,
        "training_curve": training_curve,
        "window_diagnostics": {
            "mid_steps": list(mid_steps), "late_steps": list(late_steps),
            "mid_physical_mse": mid_physical, "late_physical_mse": late_physical,
            "mid_normalized_mse": mid_normalized,
            "late_normalized_mse": late_normalized,
        },
        "paired_s1000_vs_s10000": paired,
        "continuation_gates": gates,
        "closed_loop_boundary": {
            "completed_c60_result": "KEEP_C58_PARENT",
            "c60_successes": 313, "c58_successes": 295, "pairs": 680,
            "closed_loop_was_not_used_to_select_an_existing_milestone": True,
        },
        "claim_boundary": (
            "This curve diagnoses optimization budget only. It cannot promote any "
            "milestone or reverse the completed C60 closed-loop failure."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = aggregate(args.root, args.training_root)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({
        "status": report["status"],
        "training_permission": report["training_permission"],
        "continuation_gates": report["continuation_gates"],
        "offline_curve": report["offline_curve"],
    }, indent=2))


if __name__ == "__main__":
    main()
