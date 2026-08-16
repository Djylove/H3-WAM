from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h3wam/evaluate_c56b_fact_online_paired.py"
SPEC = importlib.util.spec_from_file_location("_c56b_pair_eval_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def contract(*, causal: str, observations: str) -> dict:
    hashes = {
        "demo_manifest_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "demo_stats_sha256": "3" * 64,
        "c48_dataset_sha256": "4" * 64,
        "c48_observations_sha256": "5" * 64,
        "c59_completed_sha256": "6" * 64,
        "c59_sample_labels_sha256": "7" * 64,
    }
    return {
        "format": MODULE.TRAIN.FORMAT,
        "classification": "FACT_full_backbone_port_online_frozen_int8_h3",
        "rank_categories": list(MODULE.TRAIN.RANK_CATEGORIES),
        "loss_weights": [10.0, 1.0, 0.4, 0.4],
        "target_norm_sha256": "95df1f65eba1b1c3bfb9cebea90983ca54dffa69f60e6135354eb67e8551d000",
        "h3_sha256": MODULE.TRAIN.EXPECTED_H3_SHA256,
        "d0_sha256": MODULE.TRAIN.EXPECTED_D0_SHA256,
        "initialization": {
            "initialization_contract": "strict_online_c58b_parent_v1",
            "c58_completed_steps": 10000,
        },
        "c58_parent_sha256": "8" * 64,
        **hashes,
        "causal_failure_dataset_sha256": causal,
        "causal_failure_observations_sha256": observations,
        "base_lr": 2e-5,
        "action_lr": 2e-4,
        "warmup_steps": 500,
        "scheduler_horizon": 10000,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "seed": 20260816,
        "gradient_checkpointing": True,
        "action_horizon": 32,
        "action_shift": 5.0,
        "h3_carrier_layers": list(MODULE.LAYERWISE_H3_50_TO_ACTION_30),
        "h3_execution": "online_frozen_int8_per_rank_v1",
        "no_kv_cache": True,
    }


def endpoint(tmp_path: Path, arm: str, payload_contract: dict) -> Path:
    checkpoint = tmp_path / f"{arm}.pt"
    torch.save({
        "schema_version": 1,
        "completed_steps": 10000,
        "model": {"weight": torch.ones(1)},
        "optimizer": {},
        "lr_scheduler": {},
        "contract": payload_contract,
        "probe_step": 10000,
        "probe_predictions": [torch.ones(1) for _ in range(8)],
    }, checkpoint)
    ready = tmp_path / f"{arm}.json"
    ready.write_text(json.dumps({
        "format": "h3wam-c56b-fact-online-long10000-ready-v1",
        "status": "PASS_C56B_ONLINE_LONG10000_STRICT_RESTORE",
        "permission": "READY_FOR_PAIRED_HELDOUT",
        "effect_status": "NOT_EVIDENCE_READY",
        "arm": arm,
        "completed_steps": 10000,
        "world_size": 8,
        "global_batch": 8,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": MODULE.sha256_file(checkpoint),
        "c58_parent_sha256": payload_contract["c58_parent_sha256"],
        "causal_failure_dataset_sha256": payload_contract[
            "causal_failure_dataset_sha256"
        ],
        "causal_failure_observations_sha256": payload_contract[
            "causal_failure_observations_sha256"
        ],
        "gate": {"strict_restore": True, "online_no_cache": True},
    }))
    return ready


def test_pair_accepts_exactly_two_causal_identity_differences():
    main = contract(causal="a" * 64, observations="b" * 64)
    c61 = contract(causal="c" * 64, observations="d" * 64)
    result = MODULE.assert_only_causal_failure_diff(main, c61)
    assert result["status"] == "PASS_ONLY_CAUSAL_FAILURE_POOL_DIFFERS"
    assert set(result["different_keys"]) == MODULE.CAUSAL_KEYS


def test_pair_rejects_learning_rate_drift():
    main = contract(causal="a" * 64, observations="b" * 64)
    c61 = contract(causal="c" * 64, observations="d" * 64)
    c61["action_lr"] = 1e-4
    with pytest.raises(ValueError, match="only in causal failure"):
        MODULE.assert_only_causal_failure_diff(main, c61)


@pytest.mark.parametrize("arm", ["C60_MAIN", "C61_MATCHED"])
def test_endpoint_requires_ready_bound_checkpoint_and_full_contract(tmp_path, arm):
    payload_contract = contract(
        causal=("a" if arm == "C60_MAIN" else "c") * 64,
        observations=("b" if arm == "C60_MAIN" else "d") * 64,
    )
    ready = endpoint(tmp_path, arm, payload_contract)
    loaded_ready, payload = MODULE._load_ready(ready, arm)
    assert loaded_ready["checkpoint_sha256"] == MODULE.sha256_file(
        Path(loaded_ready["checkpoint"])
    )
    assert payload["contract"] == payload_contract


def test_endpoint_rejects_unbound_seed(tmp_path):
    payload_contract = contract(causal="a" * 64, observations="b" * 64)
    payload_contract.pop("seed")
    ready = endpoint(tmp_path, "C60_MAIN", payload_contract)
    with pytest.raises(ValueError, match="fixed training contract mismatch"):
        MODULE._load_ready(ready, "C60_MAIN")


def test_source_has_no_disk_kv_interface():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"disk_kv_read": False' in source
    assert '"disk_kv_write": False' in source
    assert "kv_subdir" not in source
    assert "torch.save(" not in source
