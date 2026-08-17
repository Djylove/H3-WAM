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
SEAL_SCRIPT = ROOT / "scripts/h3wam/seal_c67_milestone_previews.py"
SEAL_SPEC = importlib.util.spec_from_file_location(
    "_c67_preview_seal_test", SEAL_SCRIPT
)
assert SEAL_SPEC is not None and SEAL_SPEC.loader is not None
SEAL = importlib.util.module_from_spec(SEAL_SPEC)
sys.modules[SEAL_SPEC.name] = SEAL
SEAL_SPEC.loader.exec_module(SEAL)


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


def make_preview_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    final_root, complete_path = make_fixture(tmp_path)
    train_root = tmp_path / "train"
    preview_root = tmp_path / "preview"
    (train_root / "checkpoints").mkdir(parents=True)
    (train_root / "milestone-audit").mkdir()
    (preview_root / "preview-audit").mkdir(parents=True)
    (preview_root / "reports").mkdir()
    complete = json.loads(complete_path.read_text())
    audits = {int(row["milestone"]): row for row in complete["milestone_audits"]}
    for step in MODULE.MILESTONES:
        checkpoint = train_root / f"checkpoints/c67_online_s{step}.pt"
        checkpoint.write_bytes(f"fixed-checkpoint-{step}".encode())
        checkpoint_sha = sha(checkpoint)
        audit = audits[step]
        audit["checkpoint"] = str(checkpoint.resolve())
        audit["checkpoint_size_bytes"] = checkpoint.stat().st_size
        audit["restore_max_abs"] = 0.0
        (train_root / f"milestone-audit/s{step}.json").write_text(json.dumps(audit))
        preview_audit_path = preview_root / f"preview-audit/s{step}.json"
        preview_audit = {
            "format": SEAL.PREVIEW_AUDIT_FORMAT,
            "status": "PASS_C67_MILESTONE_PREVIEW_AUDIT",
            "permission": "PREVIEW_EVALUATION_ONLY",
            "effect_status": "NOT_EVIDENCE_READY",
            "milestone": step,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "training_contract_sha256": "c" * 64,
            "milestone_audit": audit,
        }
        preview_audit_path.write_text(json.dumps(preview_audit))
        report = json.loads((final_root / f"reports/s{step}.json").read_text())
        report.update({
            "permission": "PREVIEW_ONLY_PENDING_TRAINING_COMPLETE_REBIND",
            "effect_status": "PREVIEW_NOT_EVIDENCE_NOT_FOR_EARLY_STOPPING",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "restore_audit": str(preview_audit_path.resolve()),
            "restore_audit_sha256": sha(preview_audit_path),
            "training_complete": None,
            "training_complete_sha256": None,
        })
        (preview_root / f"reports/s{step}.json").write_text(json.dumps(report))
    complete_path.write_text(json.dumps(complete))
    return preview_root, train_root, complete_path


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


def test_previews_rebind_without_model_reevaluation_and_aggregate(tmp_path: Path):
    preview, train, complete = make_preview_fixture(tmp_path)
    sealed = tmp_path / "sealed"
    manifest = SEAL.seal(preview, train, complete, sealed)
    assert manifest["status"] == "PASS_C67_PREVIEWS_REBOUND_TO_TRAINING_COMPLETE"
    assert manifest["model_reevaluations_during_seal"] == 0
    report = json.loads((sealed / "reports/s10000.json").read_text())
    assert report["permission"] == "DIAGNOSTIC_ONLY_PENDING_FIXED_AGGREGATION"
    assert report["preview_provenance"]["rebound_without_model_reevaluation"] is True
    result = MODULE.aggregate(sealed, complete)
    assert result["status"] == "PASS_C67_BUDGET_BALANCED80_GATE"


def test_preview_seal_rejects_audit_drift(tmp_path: Path):
    preview, train, complete = make_preview_fixture(tmp_path)
    path = preview / "preview-audit/s7000.json"
    audit = json.loads(path.read_text())
    audit["milestone_audit"]["gate"]["strict_restore"] = False
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="preview/final audit mismatch"):
        SEAL.seal(preview, train, complete, tmp_path / "sealed")


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


def test_preview_queue_cannot_change_training_or_skip_final_rebinding():
    evaluator = (
        ROOT / "scripts/h3wam/evaluate_c67_fact_milestone_balanced80.py"
    ).read_text()
    queue = (
        ROOT / "scripts/h3wam/launch_c67_fact_milestone_preview_queue.sh"
    ).read_text()
    sealer = (ROOT / "scripts/h3wam/seal_c67_milestone_previews.py").read_text()
    assert "--preview-audit" in evaluator
    assert "PREVIEW_NOT_EVIDENCE_NOT_FOR_EARLY_STOPPING" in evaluator
    assert "seq 1000 1000 20000" in queue
    assert "for gpu in 0 1 2 3 4 5 6 7" in queue
    assert "while [[ ! -s \"${training_complete}\" ]]" in queue
    assert "seal_c67_milestone_previews.py" in queue
    assert "aggregate_c67_fact_milestone_balanced80.py" in queue
    assert "torch.distributed.run" not in queue
    assert "train_c56b_fact_online.py" not in queue
    assert "rollout_libero" not in queue
    assert "rebound_without_model_reevaluation" in sealer
