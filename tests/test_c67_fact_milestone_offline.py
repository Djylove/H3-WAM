from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h3wam/aggregate_c67_fact_milestone_balanced80.py"
SPEC = importlib.util.spec_from_file_location("_c67_offline_aggregate_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_payload(step: int) -> dict:
    progress = max(0.0, min(1.0, (step - 10_000) / 10_000))
    normalized = 0.1 - 0.02 * progress
    physical = 0.05 - 0.01 * progress
    return {
        "normalized_clip5_model_domain": {
            "action_mse": normalized,
            "prediction_std": 0.2,
        },
        "denormalized_official_minmax_clamp": {"action_mse": physical},
        "gripper_sign": {"macro_f1": 0.9},
        "language_replacement_end_to_end_h3_and_action": {
            "mean_abs_prediction_delta": 0.2
        },
        "visual_feature_shuffle_baseline_delta": {"action_mse": 0.03},
    }


def make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "offline"
    reports = root / "reports"
    reports.mkdir(parents=True)
    audits = [
        {
            "milestone": step,
            "checkpoint": str(tmp_path / f"c67_online_s{step}.pt"),
            "gate": {"strict_restore": True},
        }
        for step in MODULE.MILESTONES
    ]
    complete = tmp_path / "TRAINING_COMPLETE.json"
    complete.write_text(json.dumps({
        "format": MODULE.TRAINING_COMPLETE_FORMAT,
        "status": "PASS_C67_BUDGET_TRAINING_COMPLETE",
        "permission": "READY_FOR_PREREGISTERED_OFFLINE_ONLY",
        "effect_status": "NOT_EVIDENCE_READY",
        "completed_steps": 20_000,
        "global_batch": 8,
        "training_samples": 160_000,
        "contract_sha256": "c" * 64,
        "milestone_audits": audits,
    }))
    complete_sha = sha(complete)
    sample_ids = [f"sample-{index:02d}" for index in range(80)]
    selection = {
        "selected_ids_sha256": MODULE.SELECTED_IDS_SHA256,
        "selected_task_count": 40,
        "task_counts": {f"task-{index:02d}": 2 for index in range(40)},
    }
    data = {
        "source_manifest_sha256": "1" * 64,
        "demo_manifest_sha256": "2" * 64,
        "demo_stats_sha256": "3" * 64,
        "validation_manifest_sha256": "4" * 64,
        "selection": selection,
        "split_audit": {"episode_disjoint": True},
        "visual_shuffle": {"self_maps": 0},
    }
    execution = {
        "h3": "online_frozen_int8",
        "h3_checkpoint_sha256": "5" * 64,
        "disk_kv_read": False,
        "disk_kv_write": False,
        "disk_feature_read": False,
        "carrier_layers": list(range(30)),
        "same_selected_samples_noise_solver_normalization": True,
        "seed": 42,
        "inference_steps": 10,
        "shift": 5.0,
    }
    for step in MODULE.MILESTONES:
        progress = max(0.0, min(1.0, (step - 10_000) / 10_000))
        report = {
            "format": MODULE.REPORT_FORMAT,
            "status": "PASS_FIXED_BALANCED80",
            "permission": "DIAGNOSTIC_ONLY_PENDING_FIXED_AGGREGATION",
            "effect_status": "DIAGNOSTIC_NOT_CHECKPOINT_SELECTION",
            "milestone": step,
            "checkpoint": str(tmp_path / f"c67_online_s{step}.pt"),
            "checkpoint_sha256": f"{step // 1000:064x}",
            "restore_audit": str(tmp_path / f"restore_s{step}.json"),
            "restore_audit_sha256": f"{100 + step // 1000:064x}",
            "training_complete": str(complete.resolve()),
            "training_complete_sha256": complete_sha,
            "training_contract_sha256": "c" * 64,
            "data": data,
            "execution": execution,
            "arm": {
                "checkpoint_completed_steps": step,
                "strict_fresh_restore": {"max_abs": 0.0},
                "evaluated_ids_sha256": MODULE.SELECTED_IDS_SHA256,
                "metrics": metric_payload(step),
                "per_sample": {
                    sample_id: {
                        "normalized_action_mse": 0.1 - 0.02 * progress,
                        "physical_action_mse": 0.05 - 0.01 * progress,
                    }
                    for sample_id in sample_ids
                },
            },
            "conditioning_gates": {
                "prediction_not_constant": True,
                "gripper_metric_finite": True,
                "language_sensitive": True,
                "visual_sensitive": True,
            },
        }
        (reports / f"s{step}.json").write_text(json.dumps(report))
    return root, complete


def test_aggregate_releases_only_fixed_s10_s20_budget_effect(tmp_path: Path):
    root, complete = make_fixture(tmp_path)
    result = MODULE.aggregate(root, complete)
    assert result["format"] == "h3wam-c67-budget-balanced80-result-v1"
    assert result["status"] == "PASS_C67_BUDGET_BALANCED80_GATE"
    assert result["permission"] == "GO_C67_PAIRED_680_ROLLOUT"
    assert result["endpoint_identity"]["matched_control"]["milestone"] == 10_000
    assert result["endpoint_identity"]["treatment"]["milestone"] == 20_000
    assert result["total_model_sample_evaluations"] == 1_600
    assert all(result["gates"].values())


def test_aggregate_fails_closed_on_endpoint_regression(tmp_path: Path):
    root, complete = make_fixture(tmp_path)
    path = root / "reports/s20000.json"
    report = json.loads(path.read_text())
    report["arm"]["metrics"]["normalized_clip5_model_domain"]["action_mse"] = 0.101
    path.write_text(json.dumps(report))
    result = MODULE.aggregate(root, complete)
    assert result["status"] == "FAIL_C67_BUDGET_BALANCED80_GATE"
    assert result["permission"] == "NO_C67_PAIRED_680_ROLLOUT"
    assert result["gates"]["s20_normalized_improves_s10_by_1pct"] is False


def test_aggregate_rejects_identity_drift(tmp_path: Path):
    root, complete = make_fixture(tmp_path)
    path = root / "reports/s13000.json"
    report = json.loads(path.read_text())
    report["execution"]["seed"] = 43
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="identity drift"):
        MODULE.aggregate(root, complete)


def test_aggregate_rejects_incomplete_per_sample_pairs(tmp_path: Path):
    root, complete = make_fixture(tmp_path)
    path = root / "reports/s20000.json"
    report = json.loads(path.read_text())
    report["arm"]["per_sample"].pop("sample-00")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="invalid C67 fixed milestone report"):
        MODULE.aggregate(root, complete)


def test_ties_remain_in_the_preregistered_80_sample_win_rate_denominator(tmp_path: Path):
    root, complete = make_fixture(tmp_path)
    path = root / "reports/s20000.json"
    report = json.loads(path.read_text())
    for index, sample_id in enumerate(sorted(report["arm"]["per_sample"])):
        if index >= 30:
            report["arm"]["per_sample"][sample_id] = {
                "normalized_action_mse": 0.1,
                "physical_action_mse": 0.05,
            }
    path.write_text(json.dumps(report))
    result = MODULE.aggregate(root, complete)
    normalized = result["paired_s10000_vs_s20000"]["normalized_action_mse"]
    assert normalized["treatment_wins"] == 30
    assert normalized["ties"] == 50
    assert normalized["treatment_win_rate_all_pairs"] == 0.375
    assert normalized["treatment_win_rate_excluding_ties"] == 1.0
    assert result["gates"]["s20_normalized_error_win_rate_at_least_55pct"] is False


def test_evaluator_and_queue_freeze_c67_contract_without_rollout():
    evaluator = (
        ROOT / "scripts/h3wam/evaluate_c67_fact_milestone_balanced80.py"
    ).read_text()
    queue = (
        ROOT / "scripts/h3wam/launch_c67_fact_milestone_balanced80_queue.sh"
    ).read_text()
    for frozen in (
        'MILESTONES = tuple(range(1_000, 20_001, 1_000))',
        '"scheduler_horizon": 20_000',
        'RESTORE_FORMAT = "h3wam-c67-budget-milestone-restore-audit-v1"',
        '"demo_manifest_sha256": "b0d611c2',
        '"source_manifest_sha256": "cab8876f',
    ):
        assert frozen in evaluator
    assert "seq 1000 1000 20000" in queue
    assert "for gpu in 0 1 2 3 4 5 6 7" in queue
    assert "TRAINING_COMPLETE.json" in queue
    assert "rollout_libero" not in queue
    assert "RESULTS.json" in queue
