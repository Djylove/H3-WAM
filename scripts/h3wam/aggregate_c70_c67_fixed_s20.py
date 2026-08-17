#!/usr/bin/env python3
"""Compare the fixed C70-s20 sampler endpoint against fixed C67-s20."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SELECTED_IDS_SHA256 = "26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42"
C67_S20_SHA256 = "9ae1929e7b6ebba303e547727f58e3fd35578b17aa7d4a98da76d0b29ac1272e"
C67_REPORT_SHA256 = "8e383b183300f444f6c222f2d2ca7821921812c88a8b5f81d194eac0d58ab7e9"


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


def summary(report: dict[str, Any]) -> dict[str, float]:
    metrics = report["arm"]["metrics"]
    values = {
        "normalized_action_mse": float(
            metrics["normalized_clip5_model_domain"]["action_mse"]
        ),
        "physical_action_mse": float(
            metrics["denormalized_official_minmax_clamp"]["action_mse"]
        ),
        "gripper_macro_f1": float(metrics["gripper_sign"]["macro_f1"]),
        "language_mean_abs_delta": float(
            metrics["language_replacement_end_to_end_h3_and_action"]
            ["mean_abs_prediction_delta"]
        ),
        "visual_shuffle_action_mse": float(
            metrics["visual_feature_shuffle_baseline_delta"]["action_mse"]
        ),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("C70/C67 endpoint contains non-finite metrics")
    return values


def paired(control: dict[str, Any], candidate: dict[str, Any], metric: str) -> dict[str, Any]:
    if set(control) != set(candidate) or len(control) != 80:
        raise ValueError("C70/C67 paired sample identities differ or are incomplete")
    wins = losses = ties = 0
    deltas: list[float] = []
    for sample_id in sorted(control):
        left = float(control[sample_id][metric])
        right = float(candidate[sample_id][metric])
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError(f"non-finite paired metric: {sample_id}/{metric}")
        deltas.append(right - left)
        if abs(left - right) <= 1e-12:
            ties += 1
        elif right < left:
            wins += 1
        else:
            losses += 1
    return {
        "candidate_wins": wins,
        "control_wins": losses,
        "ties": ties,
        "candidate_win_rate_all_pairs": wins / 80,
        "candidate_minus_control_mean": sum(deltas) / len(deltas),
    }


def _report_gate(report: dict[str, Any], *, variant: str) -> None:
    expected = {
        "c67": (
            "h3wam-c67-fact-milestone-balanced80-v1",
            "DIAGNOSTIC_ONLY_PENDING_FIXED_AGGREGATION",
        ),
        "c70": (
            "h3wam-c70-sampler-coverage-milestone-balanced80-v1",
            "DIAGNOSTIC_ONLY_PENDING_FIXED_CROSS_ARM_AGGREGATION",
        ),
    }[variant]
    arm = report.get("arm", {})
    selection = report.get("data", {}).get("selection", {})
    if (
        report.get("format") != expected[0]
        or (
            report.get("variant") not in ({None, "c67"} if variant == "c67" else {"c70"})
        )
        or report.get("status") not in {"PASS_FIXED_BALANCED80", "FAIL_CONDITIONING_COLLAPSE"}
        or report.get("permission") != expected[1]
        or report.get("effect_status") != "DIAGNOSTIC_NOT_CHECKPOINT_SELECTION"
        or report.get("milestone") != 20_000
        or arm.get("checkpoint_completed_steps") != 20_000
        or arm.get("strict_fresh_restore", {}).get("max_abs") != 0.0
        or arm.get("evaluated_ids_sha256") != SELECTED_IDS_SHA256
        or len(arm.get("per_sample", {})) != 80
        or selection.get("selected_ids_sha256") != SELECTED_IDS_SHA256
        or selection.get("selected_task_count") != 40
        or any(count != 2 for count in selection.get("task_counts", {}).values())
        or not report.get("conditioning_gates")
        or any(
            not isinstance(value, bool)
            for value in report["conditioning_gates"].values()
        )
    ):
        raise ValueError(f"invalid fixed {variant.upper()}-s20 report")


def aggregate(c67_path: Path, c70_path: Path, sealed_path: Path) -> dict[str, Any]:
    c67_path, c70_path, sealed_path = (
        c67_path.resolve(), c70_path.resolve(), sealed_path.resolve()
    )
    c67, c70, sealed = map(load_json, (c67_path, c70_path, sealed_path))
    _report_gate(c67, variant="c67")
    _report_gate(c70, variant="c70")
    if (
        c67.get("checkpoint_sha256") != C67_S20_SHA256
        or sha256_file(c67_path) != C67_REPORT_SHA256
    ):
        raise ValueError("fixed C67-s20 checkpoint identity mismatch")
    if (
        sealed.get("format") != "h3wam-c70-sealed-preview-manifest-v1"
        or sealed.get("status") != "PASS_C70_PREVIEWS_REBOUND_TO_TRAINING_COMPLETE"
        or sealed.get("permission")
        != "READY_FOR_FIXED_C67_VS_C70_S20_AGGREGATION_ONLY"
        or sealed.get("effect_status") != "NOT_EVIDENCE_READY"
        or sealed.get("milestones") != list(range(1_000, 20_001, 1_000))
        or sealed.get("model_reevaluations_during_seal") != 0
        or sealed.get("reports_sha256", {}).get("20000") != sha256_file(c70_path)
    ):
        raise ValueError("C70 sealed preview manifest mismatch")
    complete_path = Path(sealed.get("training_complete", "")).resolve()
    complete = load_json(complete_path)
    if (
        sha256_file(complete_path) != sealed.get("training_complete_sha256")
        or complete.get("format")
        != "h3wam-c70-sampler-coverage-training-complete-v1"
        or complete.get("status") != "PASS_C70_SAMPLER_TRAINING_COMPLETE"
        or complete.get("candidate", {}).get("checkpoint_sha256")
        != c70.get("checkpoint_sha256")
        or complete.get("matched_control", {}).get("checkpoint_sha256")
        != C67_S20_SHA256
    ):
        raise ValueError("C70 training-complete endpoint identity mismatch")
    if (
        c67.get("data") != c70.get("data")
        or c67.get("execution") != c70.get("execution")
        or sorted(c67["arm"]["per_sample"]) != sorted(c70["arm"]["per_sample"])
    ):
        raise ValueError("C70/C67 evaluator, data, noise or sample identity drift")

    control, candidate = summary(c67), summary(c70)
    paired_metrics = {
        metric: paired(
            c67["arm"]["per_sample"], c70["arm"]["per_sample"], metric
        )
        for metric in ("normalized_action_mse", "physical_action_mse")
    }
    gates = {
        "both_endpoint_conditioning_gates_pass": all(c67["conditioning_gates"].values())
        and all(c70["conditioning_gates"].values()),
        "normalized_mean_improves_c67_by_1pct": candidate["normalized_action_mse"]
        <= 0.99 * control["normalized_action_mse"],
        "physical_mean_improves_c67_by_1pct": candidate["physical_action_mse"]
        <= 0.99 * control["physical_action_mse"],
        "normalized_sample_win_rate_at_least_55pct": paired_metrics
        ["normalized_action_mse"]["candidate_win_rate_all_pairs"] >= 0.55,
        "physical_sample_win_rate_at_least_55pct": paired_metrics
        ["physical_action_mse"]["candidate_win_rate_all_pairs"] >= 0.55,
        "gripper_within_0_005_of_c67": candidate["gripper_macro_f1"]
        >= control["gripper_macro_f1"] - 0.005,
        "language_preserves_90pct_of_c67": candidate["language_mean_abs_delta"]
        >= 0.9 * control["language_mean_abs_delta"],
        "visual_preserves_90pct_of_c67": candidate["visual_shuffle_action_mse"]
        >= 0.9 * control["visual_shuffle_action_mse"],
    }
    passed = all(gates.values())
    return {
        "format": "h3wam-c70-c67-fixed-s20-offline-result-v1",
        "status": (
            "PASS_C70_SAMPLER_BALANCED80_GATE"
            if passed else "FAIL_C70_SAMPLER_BALANCED80_GATE"
        ),
        "permission": (
            "GO_C70_VS_C67_PAIRED_680_ROLLOUT"
            if passed else "NO_C70_VS_C67_PAIRED_680_ROLLOUT"
        ),
        "effect_status": "OFFLINE_ONLY_NOT_CLOSED_LOOP_EVIDENCE",
        "control": {"summary": control, "report": str(c67_path), "report_sha256": sha256_file(c67_path)},
        "candidate": {"summary": candidate, "report": str(c70_path), "report_sha256": sha256_file(c70_path)},
        "paired": paired_metrics,
        "gates": gates,
        "sealed_manifest": {"path": str(sealed_path), "sha256": sha256_file(sealed_path)},
        "claim_boundary": (
            "A pass authorizes only the preregistered fresh paired C70-s20 versus "
            "C67-s20 LIBERO evaluation. It is not closed-loop evidence and cannot "
            "promote C70 by itself."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c67-report", type=Path, required=True)
    parser.add_argument("--c70-report", type=Path, required=True)
    parser.add_argument("--c70-sealed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = aggregate(args.c67_report, args.c70_report, args.c70_sealed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": report["status"], "permission": report["permission"],
        "gates": report["gates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
