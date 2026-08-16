from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h3wam/finalize_c56b_fact_online_long10000.py"
SPEC = importlib.util.spec_from_file_location("_c56b_finalizer_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fixture(tmp_path: Path, *, c61: bool = False):
    root = tmp_path / "long"
    (root / "checkpoints").mkdir(parents=True)
    (root / "reports").mkdir()
    (root / "restore").mkdir()
    c58_sha = "a" * 64
    c58_ready = tmp_path / "c58_READY.json"
    c58_ready.write_text(json.dumps({
        "status": "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE",
        "completed_steps": 10000,
        "checkpoint_sha256": c58_sha,
    }))
    causal_ready = None
    dataset_sha = MODULE.C60_DATASET_SHA256
    observations_sha = MODULE.C60_OBSERVATIONS_SHA256
    if c61:
        dataset_sha, observations_sha = "b" * 64, "c" * 64
        causal_ready = tmp_path / "C61_COMPLETED.json"
        causal_ready.write_text(json.dumps({
            "format": "h3wam-c61-finalized-fact-failure-dataset-v1",
            "status": "PASS_C61_FINALIZED_FACT_FAILURE_DATASET",
            "gates": {"exact": "PASS", "mask": "PASS"},
            "dataset_sha256": dataset_sha,
            "observations_sha256": observations_sha,
        }))
    contract = {
        "format": MODULE.FORMAT,
        "classification": "FACT_full_backbone_port_online_frozen_int8_h3",
        "rank_categories": [
            "expert_demo", "expert_demo", "expert_demo", "expert_demo",
            "success_rollout", "success_rollout", "observational_failure",
            "causal_failure",
        ],
        "loss_weights": [10.0, 1.0, 0.4, 0.4],
        "target_norm_sha256": MODULE.TARGET_NORM_SHA256,
        "base_lr": 2e-5,
        "action_lr": 2e-4,
        "warmup_steps": 500,
        "scheduler_horizon": 10000,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "seed": 20260816,
        "demo_manifest_sha256": "d" * 64,
        "source_manifest_sha256": "e" * 64,
        "demo_stats_sha256": "f" * 64,
        "c48_dataset_sha256": "1" * 64,
        "c48_observations_sha256": "2" * 64,
        "c59_completed_sha256": "3" * 64,
        "c59_sample_labels_sha256": "4" * 64,
        "action_horizon": 32,
        "action_shift": 5.0,
        "h3_carrier_layers": [
            0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24,
            25, 27, 29, 30, 32, 34, 35, 37, 39, 41, 42, 44, 46, 47, 49,
        ],
        "h3_execution": "online_frozen_int8_per_rank_v1",
        "no_kv_cache": True,
        "c58_parent_sha256": c58_sha,
        "causal_failure_dataset_sha256": dataset_sha,
        "causal_failure_observations_sha256": observations_sha,
    }
    checkpoint = root / "checkpoints/c56b_online_s10000.pt"
    torch.save({
        "schema_version": 1,
        "completed_steps": 10000,
        "model": {"weight": torch.ones(1)},
        "optimizer": {},
        "lr_scheduler": {},
        "contract": contract,
        "probe_step": 10000,
        "probe_predictions": [torch.ones(1) for _ in range(8)],
    }, checkpoint)
    history = [{
        "step": step,
        "loss": 1.0,
        "action_loss": 0.1,
        "future_representation_loss": 1.0,
        "future_state_loss": 1.0,
        "value_loss": 1.0,
        "sum_rank_future_leak_abs": 0.0,
        "block_gradient_norms_mean_across_ranks": [0.1] * 30,
    } for step in range(9001, 10001)]
    (root / "reports/train_s10000.json").write_text(json.dumps({
        "format": MODULE.FORMAT,
        "status": "PASS_C56B_ONLINE_TRAINING_INVOCATION",
        "effect_status": "NOT_EVIDENCE_READY",
        "completed_steps": 10000,
        "history": history,
        "contract": contract,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }))
    (root / "restore/restore_s10000.json").write_text(json.dumps({
        "format": MODULE.FORMAT,
        "status": "PASS_C56B_STRICT_RESTORE",
        "restore_max_abs": 0.0,
        "checkpoint": str(checkpoint),
    }))
    return root, c58_ready, causal_ready


@pytest.mark.parametrize("c61,arm", [(False, "C60_MAIN"), (True, "C61_MATCHED")])
def test_finalizer_accepts_only_complete_strict_endpoint(tmp_path, c61, arm):
    root, c58_ready, causal_ready = fixture(tmp_path, c61=c61)
    report = MODULE.finalize(root, c58_ready, causal_ready)
    assert report["status"] == "PASS_C56B_ONLINE_LONG10000_STRICT_RESTORE"
    assert report["permission"] == "READY_FOR_PAIRED_HELDOUT"
    assert report["arm"] == arm
    assert all(report["gate"].values())


def test_finalizer_rejects_final_stage_gradient_gap(tmp_path):
    root, c58_ready, causal_ready = fixture(tmp_path)
    path = root / "reports/train_s10000.json"
    report = json.loads(path.read_text())
    report["history"][5]["block_gradient_norms_mean_across_ranks"][2] = 0.0
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="all_30_gradients"):
        MODULE.finalize(root, c58_ready, causal_ready)


def test_long_launcher_publishes_ready_after_last_restore():
    source = (ROOT / "scripts/h3wam/launch_c56b_fact_online_long10000_8gpu.sh").read_text()
    assert source.index("restore_s${milestone}.json") < source.index("finalize_c56b_fact_online_long10000.py")
    assert '"${output_root}/READY.json"' in source


def test_checkpoint_contract_comparison_canonicalizes_tuples():
    checkpoint_contract = {
        "h3_carrier_layers": (0, 2, 49),
        "nested": {"indices": ((1, 2), (3, 4))},
    }
    json_contract = {
        "h3_carrier_layers": [0, 2, 49],
        "nested": {"indices": [[1, 2], [3, 4]]},
    }
    assert MODULE.json_canonical(checkpoint_contract) == json_contract
    json_contract["h3_carrier_layers"][-1] = 48
    assert MODULE.json_canonical(checkpoint_contract) != json_contract
