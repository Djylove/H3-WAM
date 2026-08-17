from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h3wam/finalize_c70_sampler_coverage_20k.py"
SPEC = importlib.util.spec_from_file_location("_c70_sampler_finalizer_test", SCRIPT)
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
            "expert_demo", "expert_demo", "success_rollout",
            "alternating_observational_failure_causal_failure",
        ],
        "rank_schedule": {
            "name": "c70_6_1_half_half",
            "period_steps": 2,
            "odd_step_rank7": "observational_failure",
            "even_step_rank7": "causal_failure",
            "mean_streams_per_step": {
                "expert_demo": 6.0,
                "success_rollout": 1.0,
                "observational_failure": 0.5,
                "causal_failure": 0.5,
            },
        },
        "loss_weights": [10.0, 1.0, 0.4, 0.4],
        "target_norm_sha256": MODULE.TARGET_NORM_SHA256,
        "h3_sha256": MODULE.H3_SHA256,
        "d0_sha256": MODULE.D0_SHA256,
        "c58_parent_sha256": MODULE.C58_SHA256,
        "initialization": {
            "initialization_contract": "strict_online_c58b_parent_v1",
            "c58_completed_steps": 10_000,
        },
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
        "demo_manifest_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "demo_stats_sha256": "3" * 64,
        "c48_dataset_sha256": "4" * 64,
        "c48_observations_sha256": "5" * 64,
        "c59_completed_sha256": "6" * 64,
        "c59_sample_labels_sha256": "7" * 64,
    }


def test_c70_finalizer_accepts_only_the_preregistered_sampler() -> None:
    MODULE.require_contract(contract())
    wrong = contract()
    wrong["rank_schedule"] = {**wrong["rank_schedule"], "period_steps": 1}
    with pytest.raises(ValueError, match="rank_schedule"):
        MODULE.require_contract(wrong)


def test_c70_long_launcher_is_release_and_restore_locked() -> None:
    source = (ROOT / "scripts/h3wam/launch_c70_sampler_coverage_20k_8gpu.sh").read_text()
    for required in (
        "C70_RELEASE_FILE:?", "C70_CANARY_GO_LONG:?",
        "GO_C70_SAMPLER_COVERAGE_20K", "c70_6_1_half_half",
        "seq 1000 1000 20000", "--steps 1000", "--restore-check-only",
        "c70_sampler_s${milestone}.pt", "--c67-control", "322122547200",
    ):
        assert required in source
    assert "rollout_libero" not in source


def test_complete_source_freeze_covers_c70_long_execution() -> None:
    source = (ROOT / "scripts/h3wam/freeze_c67_rollout_source.py").read_text()
    assert '"scripts/h3wam/launch_c70_sampler_coverage_20k_8gpu.sh"' in source
    assert '"scripts/h3wam/finalize_c70_sampler_coverage_20k.py"' in source
    assert '"scripts/h3wam/prepare_c70_milestone_preview_audit.py"' in source
    assert '"scripts/h3wam/launch_c70_sampler_milestone_preview_queue.sh"' in source


def test_c70_preview_queue_is_read_only_and_uses_fixed_balanced80() -> None:
    source = (ROOT / "scripts/h3wam/launch_c70_sampler_milestone_preview_queue.sh").read_text()
    for required in (
        "c70_sampler_s${milestone}.pt", "--variant c70", "--preview-audit",
        "seq 1000 1000 20000", "manifest_val.jsonl",
        "PASS_C70_ALL_20_PREVIEWS_COMPLETE",
    ):
        assert required in source
    assert "--save-checkpoint" not in source
    assert "--load-checkpoint" not in source
