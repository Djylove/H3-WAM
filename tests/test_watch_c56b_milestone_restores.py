from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h3wam/watch_c56b_milestone_restores.py"
SPEC = importlib.util.spec_from_file_location("_c56b_milestone_audit_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def make_milestone(tmp_path: Path, *, milestone: int = 1000):
    root = tmp_path / "run"
    for name in ("checkpoints", "reports", "restore"):
        (root / name).mkdir(parents=True, exist_ok=True)
    checkpoint = (root / f"checkpoints/c56b_online_s{milestone}.pt").resolve()
    checkpoint.write_bytes(b"checkpoint")
    parent, dataset, observations = "a" * 64, "b" * 64, "c" * 64
    contract = {
        "c58_parent_sha256": parent,
        "causal_failure_dataset_sha256": dataset,
        "causal_failure_observations_sha256": observations,
        "h3_execution": "online_frozen_int8_per_rank_v1",
        "no_kv_cache": True,
        "base_lr": 2e-5,
        "action_lr": 2e-4,
        "warmup_steps": 500,
        "scheduler_horizon": 10000,
        "seed": 20260816,
    }
    history = [{
        "step": step,
        "loss": 1.0,
        "action_loss": 0.1,
        "future_representation_loss": 1.0,
        "future_state_loss": 1.0,
        "value_loss": 1.0,
        "block_gradient_norms_mean_across_ranks": [0.1] * 30,
        "sum_rank_future_leak_abs": 0.0,
    } for step in range(milestone - 999, milestone + 1)]
    (root / f"reports/train_s{milestone}.json").write_text(json.dumps({
        "format": M.FORMAT,
        "status": "PASS_C56B_ONLINE_TRAINING_INVOCATION",
        "completed_steps": milestone,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "contract": contract,
        "history": history,
    }))
    (root / f"restore/restore_s{milestone}.json").write_text(json.dumps({
        "format": M.FORMAT,
        "status": "PASS_C56B_STRICT_RESTORE",
        "restore_max_abs": 0.0,
        "checkpoint": str(checkpoint),
    }))
    return root, parent, dataset, observations


def test_accepts_exact_1k_segment_and_restore(tmp_path):
    root, parent, dataset, observations = make_milestone(tmp_path)
    report = M.validate_milestone(
        root, 1000, parent_sha256=parent,
        causal_dataset_sha256=dataset,
        causal_observations_sha256=observations,
    )
    assert report["status"] == "PASS_C56B_MILESTONE_STRICT_RESTORE"
    assert report["minimum_block_gradient"] == 0.1


@pytest.mark.parametrize("field", ["restore", "leak", "gradient"])
def test_rejects_restore_or_training_contract_gap(tmp_path, field):
    root, parent, dataset, observations = make_milestone(tmp_path)
    if field == "restore":
        path = root / "restore/restore_s1000.json"
        value = json.loads(path.read_text())
        value["restore_max_abs"] = 0.01
    else:
        path = root / "reports/train_s1000.json"
        value = json.loads(path.read_text())
        if field == "leak":
            value["history"][0]["sum_rank_future_leak_abs"] = 1.0
        else:
            value["history"][0]["block_gradient_norms_mean_across_ranks"][0] = 0.0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        M.validate_milestone(
            root, 1000, parent_sha256=parent,
            causal_dataset_sha256=dataset,
            causal_observations_sha256=observations,
        )
