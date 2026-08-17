from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h3wam/aggregate_c70_c67_fixed_s20.py"
SPEC = importlib.util.spec_from_file_location("_c70_fixed_s20_gate_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(variant: str, normalized: float, physical: float) -> dict:
    c70 = variant == "c70"
    per_sample = {
        f"sample-{index:02d}": {
            "normalized_action_mse": normalized,
            "physical_action_mse": physical,
        }
        for index in range(80)
    }
    return {
        "format": (
            "h3wam-c70-sampler-coverage-milestone-balanced80-v1"
            if c70 else "h3wam-c67-fact-milestone-balanced80-v1"
        ),
        "variant": "c70" if c70 else None,
        "status": "PASS_FIXED_BALANCED80",
        "permission": (
            "DIAGNOSTIC_ONLY_PENDING_FIXED_CROSS_ARM_AGGREGATION"
            if c70 else "DIAGNOSTIC_ONLY_PENDING_FIXED_AGGREGATION"
        ),
        "effect_status": "DIAGNOSTIC_NOT_CHECKPOINT_SELECTION",
        "milestone": 20_000,
        "checkpoint_sha256": "c" * 64 if c70 else MODULE.C67_S20_SHA256,
        "data": {
            "selection": {
                "selected_ids_sha256": MODULE.SELECTED_IDS_SHA256,
                "selected_task_count": 40,
                "task_counts": {f"task-{index:02d}": 2 for index in range(40)},
            }
        },
        "execution": {"seed": 42, "inference_steps": 10, "shift": 5.0},
        "arm": {
            "checkpoint_completed_steps": 20_000,
            "strict_fresh_restore": {"max_abs": 0.0},
            "evaluated_ids_sha256": MODULE.SELECTED_IDS_SHA256,
            "per_sample": per_sample,
            "metrics": {
                "normalized_clip5_model_domain": {"action_mse": normalized},
                "denormalized_official_minmax_clamp": {"action_mse": physical},
                "gripper_sign": {"macro_f1": 0.93},
                "language_replacement_end_to_end_h3_and_action": {
                    "mean_abs_prediction_delta": 0.2
                },
                "visual_feature_shuffle_baseline_delta": {"action_mse": 0.03},
            },
        },
        "conditioning_gates": {
            "prediction_not_constant": True,
            "gripper_metric_finite": True,
            "language_sensitive": True,
            "visual_sensitive": True,
        },
    }


def fixture(tmp_path: Path, candidate_error: float) -> tuple[Path, Path, Path]:
    c67_path, c70_path = tmp_path / "c67.json", tmp_path / "c70.json"
    c67_path.write_text(json.dumps(report("c67", 1.0, 1.0)))
    c70_path.write_text(json.dumps(report("c70", candidate_error, candidate_error)))
    MODULE.C67_REPORT_SHA256 = sha(c67_path)
    complete = tmp_path / "TRAINING_COMPLETE.json"
    complete.write_text(json.dumps({
        "format": "h3wam-c70-sampler-coverage-training-complete-v1",
        "status": "PASS_C70_SAMPLER_TRAINING_COMPLETE",
        "candidate": {"checkpoint_sha256": "c" * 64},
        "matched_control": {"checkpoint_sha256": MODULE.C67_S20_SHA256},
    }))
    sealed = tmp_path / "SEALED.json"
    sealed.write_text(json.dumps({
        "format": "h3wam-c70-sealed-preview-manifest-v1",
        "status": "PASS_C70_PREVIEWS_REBOUND_TO_TRAINING_COMPLETE",
        "permission": "READY_FOR_FIXED_C67_VS_C70_S20_AGGREGATION_ONLY",
        "effect_status": "NOT_EVIDENCE_READY",
        "milestones": list(range(1_000, 20_001, 1_000)),
        "model_reevaluations_during_seal": 0,
        "reports_sha256": {"20000": sha(c70_path)},
        "training_complete": str(complete),
        "training_complete_sha256": sha(complete),
    }))
    return c67_path, c70_path, sealed


def test_fixed_s20_gate_passes_only_the_preregistered_improvement(tmp_path: Path) -> None:
    c67, c70, sealed = fixture(tmp_path, 0.98)
    result = MODULE.aggregate(c67, c70, sealed)
    assert result["status"] == "PASS_C70_SAMPLER_BALANCED80_GATE"
    assert result["permission"] == "GO_C70_VS_C67_PAIRED_680_ROLLOUT"
    assert all(result["gates"].values())


def test_fixed_s20_gate_rejects_no_action_improvement(tmp_path: Path) -> None:
    c67, c70, sealed = fixture(tmp_path, 1.01)
    result = MODULE.aggregate(c67, c70, sealed)
    assert result["status"] == "FAIL_C70_SAMPLER_BALANCED80_GATE"
    assert result["permission"] == "NO_C70_VS_C67_PAIRED_680_ROLLOUT"
    assert not result["gates"]["normalized_mean_improves_c67_by_1pct"]


def test_final_watcher_never_launches_rollout() -> None:
    source = (ROOT / "scripts/h3wam/watch_c70_final_offline_gate.sh").read_text()
    assert "seal_c70_milestone_previews.py" in source
    assert "aggregate_c70_c67_fixed_s20.py" in source
    assert "rollout_libero" not in source
