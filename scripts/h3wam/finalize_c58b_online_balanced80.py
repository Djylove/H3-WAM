#!/usr/bin/env python3
"""Audit C58b online balanced-80 and atomically release fresh LIBERO."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SELECTED_IDS_SHA256 = "26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42"
H3_SHA256 = "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
LAYERS = (0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24,
          25, 27, 29, 30, 32, 34, 35, 37, 39, 41, 42, 44, 46, 47, 49)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def finalize(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("format") != "h3wam-c58b-online-h3-balanced80-v1":
        raise ValueError("C58b balanced80 report format mismatch")
    if report.get("candidate") != "C58B_FASTWAM_FULL30_H3_LAYERWISE":
        raise ValueError("C58b candidate identity mismatch")
    checkpoint = report.get("checkpoint", {})
    if checkpoint.get("completed_steps") != 10_000:
        raise ValueError("C58b balanced80 did not evaluate s10000")
    if checkpoint.get("fresh_restore", {}).get("max_abs") != 0.0:
        raise ValueError("C58b balanced80 fresh restore is not exact")
    execution = report.get("execution", {})
    required_execution = {
        "h3": "online_frozen_int8",
        "h3_checkpoint_sha256": H3_SHA256,
        "disk_kv_read": False,
        "disk_feature_read": False,
        "carrier_layers": list(LAYERS),
        "carrier_mapping": "one_to_one_uniform_h3_50_to_action30",
    }
    mismatches = {
        key: {"actual": execution.get(key), "expected": value}
        for key, value in required_execution.items()
        if execution.get(key) != value
    }
    if mismatches:
        raise ValueError(f"C58b online execution mismatch: {mismatches}")
    data = report.get("data", {})
    selection = data.get("selection", {})
    if (
        data.get("selected_sample_ids_sha256") != SELECTED_IDS_SHA256
        or selection.get("selected_ids_sha256") != SELECTED_IDS_SHA256
        or selection.get("selected_items") != 80
        or selection.get("selected_task_count") != 40
        or any(value != 2 for value in selection.get("task_counts", {}).values())
    ):
        raise ValueError("C58b balanced80 sample identity/count mismatch")
    split = data.get("split_audit", {})
    if split.get("window_overlap") != 0 or split.get("episode_overlap") != 0:
        raise ValueError("C58b balanced80 is not episode-disjoint")
    inference = report.get("inference", {})
    required_inference = {
        "shift": 5.0,
        "steps": 10,
        "seed": 42,
        "batch_size": 1,
        "same_noise_for_baseline_language_visual": True,
    }
    if any(inference.get(key) != value for key, value in required_inference.items()):
        raise ValueError("C58b balanced80 inference protocol mismatch")

    metrics = report.get("metrics", {})
    normalized = metrics.get("normalized_clip5_model_domain", {})
    physical = metrics.get("denormalized_official_minmax_clamp", {})
    gripper = metrics.get("gripper_sign", {})
    language = metrics.get("language_replacement_sensitivity", {})
    visual = metrics.get("visual_feature_shuffle", {}).get(
        "baseline_vs_shuffle_action_delta", {}
    ).get("normalized_model_domain", {})
    observed = {
        "normalized_action_mse": normalized.get("action_mse"),
        "physical_action_mse": physical.get("action_mse"),
        "prediction_std": normalized.get("prediction_std"),
        "gripper_macro_f1": gripper.get("macro_f1"),
        "language_mean_abs_delta": language.get("mean_abs_prediction_delta"),
        "visual_shuffle_delta_mse": visual.get("action_mse"),
    }
    if not all(_finite(value) for value in observed.values()):
        raise ValueError(f"C58b balanced80 has non-finite metrics: {observed}")
    gates = {
        "completed_all_80_episode_disjoint_samples": True,
        "online_h3_no_disk_kv_or_feature": True,
        "strict_s10000_restore_exact": True,
        "prediction_not_constant_std_at_least_0_01": (
            observed["prediction_std"] >= 0.01
        ),
        "language_sensitivity_at_least_0_01": (
            observed["language_mean_abs_delta"] >= 0.01
        ),
        "visual_sensitivity_delta_mse_at_least_1e_4": (
            observed["visual_shuffle_delta_mse"] >= 1.0e-4
        ),
    }
    passed = all(gates.values())
    return {
        "format": "h3wam-c58b-online-balanced80-ready-v1",
        "status": "PASS" if passed else "FAIL",
        "permission": "GO_FRESH_LIBERO" if passed else "NO_GO_FRESH_LIBERO",
        "report": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "checkpoint": checkpoint.get("path"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "selected_sample_ids_sha256": SELECTED_IDS_SHA256,
        "observed": observed,
        "gates": gates,
        "closed_loop_protocol": {
            "phase": "fresh_trial33_strict_paired_c58b_vs_d0",
            "arms": [
                "candidate_c58b_online_h3_full30_layerwise",
                "control_d0_online_h3_repeat_layer49",
            ],
            "control_d0_checkpoint_sha256": (
                "36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"
            ),
            "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
            "tasks_per_suite": 10,
            "trial_indices": [33],
            "paired_episodes": 40,
            "episodes_per_arm": 40,
            "total_episodes": 80,
            "wait_steps": 30,
            "environment_seed": None,
            "policy_noise_seed_base": None,
            "episode_seed_contract": "trial_index_times_1000_plus_seed",
            "expected_trial33_episode_seed": 33_042,
            "action_horizon": 32,
            "replan_interval": 8,
            "inference_steps": 10,
            "ensemble": False,
            "carrier": "online frozen INT8 H3, exact 30-layer mapping",
        },
        "claim_boundary": (
            "This gate proves a non-collapsed held-out online-H3 action mechanism; "
            "only the released fresh LIBERO rollout measures execution success."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.report.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2), flush=True)
    if result["permission"] != "GO_FRESH_LIBERO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
