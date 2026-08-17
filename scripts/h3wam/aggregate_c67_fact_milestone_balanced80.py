#!/usr/bin/env python3
"""Aggregate the preregistered C67 s1k..s20k offline budget experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


REPORT_FORMAT = "h3wam-c67-fact-milestone-balanced80-v1"
RESULT_FORMAT = "h3wam-c67-budget-balanced80-result-v1"
TRAINING_COMPLETE_FORMAT = "h3wam-c67-c60-budget-ablation-training-complete-v1"
MILESTONES = tuple(range(1_000, 20_001, 1_000))
SELECTED_IDS_SHA256 = "26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty collection")
    return sum(values) / len(values)


def summarize(report: dict[str, Any]) -> dict[str, float | int | bool]:
    metrics = report["arm"]["metrics"]
    normalized = metrics["normalized_clip5_model_domain"]
    physical = metrics["denormalized_official_minmax_clamp"]
    gripper = metrics["gripper_sign"]
    language = metrics["language_replacement_end_to_end_h3_and_action"]
    visual = metrics["visual_feature_shuffle_baseline_delta"]
    values = {
        "normalized_action_mse": float(normalized["action_mse"]),
        "physical_action_mse": float(physical["action_mse"]),
        "prediction_std": float(normalized["prediction_std"]),
        "gripper_macro_f1": float(gripper["macro_f1"]),
        "language_mean_abs_delta": float(language["mean_abs_prediction_delta"]),
        "visual_shuffle_action_mse": float(visual["action_mse"]),
    }
    return {
        "milestone": int(report["milestone"]),
        **values,
        "all_metrics_finite": all(math.isfinite(value) for value in values.values()),
        "conditioning_pass": all(report["conditioning_gates"].values()),
    }


def paired_delta(control: dict[str, Any], treatment: dict[str, Any], metric: str) -> dict[str, Any]:
    if set(control) != set(treatment) or len(control) != 80:
        raise ValueError("C67 s10/s20 per-sample identities differ or are incomplete")
    control_wins = treatment_wins = ties = 0
    deltas = []
    for sample_id in sorted(control):
        left = float(control[sample_id][metric])
        right = float(treatment[sample_id][metric])
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError(f"C67 non-finite paired error: {sample_id}/{metric}")
        deltas.append(right - left)
        if abs(left - right) <= 1e-12:
            ties += 1
        elif right < left:
            treatment_wins += 1
        else:
            control_wins += 1
    non_ties = treatment_wins + control_wins
    return {
        "control_wins": control_wins,
        "treatment_wins": treatment_wins,
        "ties": ties,
        "treatment_win_rate_all_pairs": treatment_wins / len(control),
        "treatment_win_rate_excluding_ties": (
            treatment_wins / non_ties if non_ties else 0.0
        ),
        "treatment_minus_control_mean": mean(deltas),
    }


def endpoint_identity(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "milestone": int(report["milestone"]),
        "checkpoint": str(Path(report["checkpoint"]).resolve()),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "restore_audit": str(Path(report["restore_audit"]).resolve()),
        "restore_audit_sha256": report["restore_audit_sha256"],
    }


def aggregate(root: Path, training_complete_path: Path) -> dict[str, Any]:
    root = root.resolve()
    training_complete_path = training_complete_path.resolve()
    complete = load_json(training_complete_path)
    complete_sha = sha256_file(training_complete_path)
    audits = complete.get("milestone_audits", [])
    if (
        complete.get("format") != TRAINING_COMPLETE_FORMAT
        or complete.get("status") != "PASS_C67_BUDGET_TRAINING_COMPLETE"
        or complete.get("permission") != "READY_FOR_PREREGISTERED_OFFLINE_ONLY"
        or complete.get("effect_status") != "NOT_EVIDENCE_READY"
        or complete.get("completed_steps") != 20_000
        or complete.get("global_batch") != 8
        or complete.get("training_samples") != 160_000
        or len(audits) != 20
        or {int(audit.get("milestone", -1)) for audit in audits} != set(MILESTONES)
        or any(not audit.get("gate") or not all(audit["gate"].values()) for audit in audits)
    ):
        raise ValueError("C67 training-complete evidence failed before offline aggregation")

    reports: dict[int, dict[str, Any]] = {}
    report_sha256: dict[str, str] = {}
    shared_identity = None
    for milestone in MILESTONES:
        path = root / f"reports/s{milestone}.json"
        report = load_json(path)
        selection = report.get("data", {}).get("selection", {})
        arm = report.get("arm", {})
        if (
            report.get("format") != REPORT_FORMAT
            or report.get("status") not in {"PASS_FIXED_BALANCED80", "FAIL_CONDITIONING_COLLAPSE"}
            or report.get("permission") != "DIAGNOSTIC_ONLY_PENDING_FIXED_AGGREGATION"
            or report.get("effect_status") != "DIAGNOSTIC_NOT_CHECKPOINT_SELECTION"
            or report.get("milestone") != milestone
            or report.get("training_complete_sha256") != complete_sha
            or Path(report.get("training_complete", "")).resolve() != training_complete_path
            or arm.get("checkpoint_completed_steps") != milestone
            or arm.get("strict_fresh_restore", {}).get("max_abs") != 0.0
            or arm.get("evaluated_ids_sha256") != SELECTED_IDS_SHA256
            or len(arm.get("per_sample", {})) != 80
            or selection.get("selected_ids_sha256") != SELECTED_IDS_SHA256
            or selection.get("selected_task_count") != 40
            or any(count != 2 for count in selection.get("task_counts", {}).values())
        ):
            raise ValueError(f"invalid C67 fixed milestone report: s{milestone}")
        identity = {
            "training_complete_sha256": report["training_complete_sha256"],
            "training_contract_sha256": report["training_contract_sha256"],
            "data": report["data"],
            "execution": report["execution"],
            "evaluated_ids_sha256": arm["evaluated_ids_sha256"],
            "per_sample_ids": sorted(arm["per_sample"]),
        }
        if shared_identity is None:
            shared_identity = identity
        elif identity != shared_identity:
            raise ValueError(f"C67 milestone evaluation identity drift: s{milestone}")
        reports[milestone] = report
        report_sha256[str(milestone)] = sha256_file(path)
    assert shared_identity is not None

    curve = [summarize(reports[milestone]) for milestone in MILESTONES]
    by_step = {int(row["milestone"]): row for row in curve}
    control_steps = (10_000, 11_000, 12_000)
    treatment_steps = (18_000, 19_000, 20_000)
    control_window = {
        metric: mean([float(by_step[step][metric]) for step in control_steps])
        for metric in ("normalized_action_mse", "physical_action_mse")
    }
    treatment_window = {
        metric: mean([float(by_step[step][metric]) for step in treatment_steps])
        for metric in ("normalized_action_mse", "physical_action_mse")
    }
    s10, s20 = by_step[10_000], by_step[20_000]
    paired = {
        metric: paired_delta(
            reports[10_000]["arm"]["per_sample"],
            reports[20_000]["arm"]["per_sample"],
            metric,
        )
        for metric in ("normalized_action_mse", "physical_action_mse")
    }
    gates = {
        "all_20_reports_complete": len(reports) == 20,
        "all_20_training_restores_pass": len(audits) == 20,
        "all_20_fresh_restores_exact": all(
            reports[step]["arm"]["strict_fresh_restore"]["max_abs"] == 0.0
            for step in MILESTONES
        ),
        "all_20_conditioning_gates_pass": all(
            bool(row["conditioning_pass"]) for row in curve
        ),
        "all_20_metrics_finite": all(bool(row["all_metrics_finite"]) for row in curve),
        "s18_s20_normalized_improves_s10_s12_by_1pct": (
            treatment_window["normalized_action_mse"]
            <= 0.99 * control_window["normalized_action_mse"]
        ),
        "s18_s20_physical_improves_s10_s12_by_1pct": (
            treatment_window["physical_action_mse"]
            <= 0.99 * control_window["physical_action_mse"]
        ),
        "s20_normalized_improves_s10_by_1pct": (
            float(s20["normalized_action_mse"])
            <= 0.99 * float(s10["normalized_action_mse"])
        ),
        "s20_physical_improves_s10_by_1pct": (
            float(s20["physical_action_mse"])
            <= 0.99 * float(s10["physical_action_mse"])
        ),
        "s20_normalized_error_win_rate_at_least_55pct": (
            paired["normalized_action_mse"]["treatment_win_rate_all_pairs"] >= 0.55
        ),
        "s20_physical_error_win_rate_at_least_55pct": (
            paired["physical_action_mse"]["treatment_win_rate_all_pairs"] >= 0.55
        ),
        "s20_gripper_within_0_005_of_s10": (
            float(s20["gripper_macro_f1"]) >= float(s10["gripper_macro_f1"]) - 0.005
        ),
        "s20_language_preserves_90pct_of_s10": (
            float(s20["language_mean_abs_delta"])
            >= 0.9 * float(s10["language_mean_abs_delta"])
        ),
        "s20_visual_preserves_90pct_of_s10": (
            float(s20["visual_shuffle_action_mse"])
            >= 0.9 * float(s10["visual_shuffle_action_mse"])
        ),
    }
    passed = all(gates.values())
    return {
        "format": RESULT_FORMAT,
        "status": (
            "PASS_C67_BUDGET_BALANCED80_GATE"
            if passed else "FAIL_C67_BUDGET_BALANCED80_GATE"
        ),
        "permission": (
            "GO_C67_PAIRED_680_ROLLOUT"
            if passed else "NO_C67_PAIRED_680_ROLLOUT"
        ),
        "effect_status": "OFFLINE_ONLY_NOT_CLOSED_LOOP_EVIDENCE",
        "hypothesis": (
            "Within the fixed C67 horizon-20000 trajectory, s20000 must improve "
            "offline action errors over its own s10000 control without losing conditioning."
        ),
        "training_complete": {
            "path": str(training_complete_path),
            "sha256": complete_sha,
            "contract_sha256": complete["contract_sha256"],
        },
        "endpoint_identity": {
            "matched_control": endpoint_identity(reports[10_000]),
            "treatment": endpoint_identity(reports[20_000]),
        },
        "milestones": list(MILESTONES),
        "evaluated_samples_per_milestone": 80,
        "total_model_sample_evaluations": 1_600,
        "identity": shared_identity,
        "report_sha256": report_sha256,
        "curve": curve,
        "window_comparison": {
            "control_steps": list(control_steps),
            "treatment_steps": list(treatment_steps),
            "control_mean": control_window,
            "treatment_mean": treatment_window,
        },
        "paired_s10000_vs_s20000": paired,
        "gates": gates,
        "claim_boundary": (
            "A pass authorizes only the preregistered paired C67-s20 versus C67-s10 "
            "680-state LIBERO test. It is not closed-loop evidence and cannot promote "
            "either endpoint by itself."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-complete", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = aggregate(args.root, args.training_complete)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({
        "status": report["status"],
        "permission": report["permission"],
        "endpoint_identity": report["endpoint_identity"],
        "gates": report["gates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
