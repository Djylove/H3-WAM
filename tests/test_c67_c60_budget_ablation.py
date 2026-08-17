from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h3wam/finalize_c67_c60_budget_ablation_20k.py"
SPEC = importlib.util.spec_from_file_location("_c67_budget_finalizer_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def contract() -> dict:
    return {
        "format": MODULE.FORMAT,
        "classification": "FACT_full_backbone_port_online_frozen_int8_h3",
        "rank_categories": [
            "expert_demo", "expert_demo", "expert_demo", "expert_demo",
            "success_rollout", "success_rollout", "observational_failure",
            "causal_failure",
        ],
        "loss_weights": [10.0, 1.0, 0.4, 0.4],
        "target_norm_sha256": MODULE.TARGET_NORM_SHA256,
        "h3_sha256": MODULE.H3_SHA256,
        "d0_sha256": MODULE.D0_SHA256,
        "initialization": {
            "initialization_contract": "strict_online_c58b_parent_v1",
            "c58_completed_steps": 10_000,
        },
        "c58_parent_sha256": MODULE.C58_SHA256,
        "demo_manifest_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "demo_stats_sha256": "3" * 64,
        "c48_dataset_sha256": "4" * 64,
        "c48_observations_sha256": "5" * 64,
        "c59_completed_sha256": "6" * 64,
        "c59_sample_labels_sha256": "7" * 64,
        "causal_failure_dataset_sha256": MODULE.C60_DATASET_SHA256,
        "causal_failure_observations_sha256": MODULE.C60_OBSERVATIONS_SHA256,
        "base_lr": 2e-5,
        "action_lr": 2e-4,
        "warmup_steps": 500,
        "scheduler_horizon": 20_000,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "seed": 20260816,
        "gradient_checkpointing": True,
        "action_horizon": 32,
        "action_shift": 5.0,
        "h3_carrier_layers": list(MODULE.LAYERS),
        "h3_execution": "online_frozen_int8_per_rank_v1",
        "no_kv_cache": True,
    }


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(MODULE, "MIN_CHECKPOINT_BYTES", 0)
    root = tmp_path / "c67"
    for name in ("checkpoints", "reports", "restore"):
        (root / name).mkdir(parents=True, exist_ok=True)
    fixed = contract()
    for milestone in MODULE.MILESTONES:
        factor = MODULE.expected_lr_factor(milestone)
        rates = {"base": 2e-5 * factor, "action": 2e-4 * factor}
        history = [{
            "step": step,
            "loss": 1.0,
            "action_loss": 0.1,
            "future_representation_loss": 0.2,
            "future_state_loss": 0.3,
            "value_loss": 0.4,
            "sum_rank_future_leak_abs": 0.0,
            "block_gradient_norms_mean_across_ranks": [0.1] * 30,
            "learning_rates": rates,
        } for step in range(milestone - 999, milestone + 1)]
        checkpoint = root / f"checkpoints/c67_online_s{milestone}.pt"
        torch.save({
            "schema_version": 1,
            "completed_steps": milestone,
            "model": {"weight": torch.ones(1)},
            "optimizer": {},
            "lr_scheduler": {"last_epoch": milestone},
            "contract": fixed,
            "probe_step": milestone,
            "probe_predictions": [torch.ones(1) for _ in range(8)],
        }, checkpoint)
        previous = None if milestone == 1_000 else str(
            (root / f"checkpoints/c67_online_s{milestone - 1_000}.pt").resolve()
        )
        (root / f"reports/train_s{milestone}.json").write_text(json.dumps({
            "format": MODULE.FORMAT,
            "status": "PASS_C56B_ONLINE_TRAINING_INVOCATION",
            "effect_status": "NOT_EVIDENCE_READY",
            "completed_steps": milestone,
            "history": history,
            "contract": fixed,
            "checkpoint": str(checkpoint.resolve()),
            "loaded_checkpoint": previous,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "restore_at_load_max_abs": None if previous is None else 0.0,
        }))
        (root / f"restore/restore_s{milestone}.json").write_text(json.dumps({
            "format": MODULE.FORMAT,
            "status": "PASS_C56B_STRICT_RESTORE",
            "restore_max_abs": 0.0,
            "checkpoint": str(checkpoint.resolve()),
            "loaded_checkpoint": str(checkpoint.resolve()),
        }))
    c58_ready = tmp_path / "C58_READY.json"
    c58_ready.write_text(json.dumps({
        "status": "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE",
        "permission": "READY_FOR_CHILD_BRANCH_AND_LIBERO_EVAL",
        "completed_steps": 10_000,
        "checkpoint_sha256": MODULE.C58_SHA256,
    }))
    return root, c58_ready


def test_finalizer_freezes_same_trajectory_control_and_treatment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, c58_ready = fixture(tmp_path, monkeypatch)
    report = MODULE.finalize(root, c58_ready)
    assert report["status"] == "PASS_C67_BUDGET_TRAINING_COMPLETE"
    assert report["permission"] == "READY_FOR_PREREGISTERED_OFFLINE_ONLY"
    assert report["effect_status"] == "NOT_EVIDENCE_READY"
    assert report["matched_control"]["milestone"] == 10_000
    assert report["treatment"]["milestone"] == 20_000
    assert len(report["milestone_audits"]) == 20
    assert report["scheduler"]["s10000_factor"] > 0.5
    assert report["scheduler"]["s20000_factor"] == 0.0


def test_finalizer_rejects_broken_predecessor_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, c58_ready = fixture(tmp_path, monkeypatch)
    path = root / "reports/train_s11000.json"
    report = json.loads(path.read_text())
    report["loaded_checkpoint"] = str(root / "checkpoints/c67_online_s9000.pt")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="predecessor_lineage"):
        MODULE.finalize(root, c58_ready)


def test_finalizer_rejects_lr_zero_at_internal_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, c58_ready = fixture(tmp_path, monkeypatch)
    path = root / "reports/train_s10000.json"
    report = json.loads(path.read_text())
    report["history"][-1]["learning_rates"] = {"base": 0.0, "action": 0.0}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="milestone_lr"):
        MODULE.finalize(root, c58_ready)


def test_launcher_is_manual_release_locked_and_runs_fixed_20k_segments():
    source = (
        ROOT / "scripts/h3wam/launch_c67_c60_budget_ablation_20k_8gpu.sh"
    ).read_text()
    assert "C67_RELEASE_FILE:?" in source
    assert "GO_C67_BUDGET_ABLATION_20K" in source
    assert "source_sha256" in source
    assert "seq 1000 1000 20000" in source
    assert "--scheduler-horizon 20000" in source
    assert "--steps 1000" in source
    assert "--restore-check-only" in source
    assert "322122547200" in source
    assert "rollout_libero" not in source
    assert "evaluate_c56b_fact_online_paired.py" not in source


def test_dossier_preregisters_budget_and_effect_gates():
    dossier = json.loads((
        ROOT / "experiments/dossiers/h3_c67_c60_budget_ablation_v1.json"
    ).read_text())
    assert dossier["classification"] == "controlled_ablation"
    assert dossier["budget"]["training_samples"] == 160_000
    assert math.isclose(dossier["budget"]["effective_epochs"], 0.733522)
    assert dossier["decision"]["automatic_launch"] is False
    document = (ROOT / "docs/C67_C60_BUDGET_ABLATION_2026-08-17.md").read_text()
    for frozen in (
        "s18–s20", ">=55%", ">=+3pp", "net wins `>=20`",
        "p<=0.05", "EVIDENCE_READY_BUDGET_ABLATION_ONLY",
        "历史 C60-s10 仅是外部",
    ):
        assert frozen in document
