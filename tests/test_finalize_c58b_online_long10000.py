import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_finalizer():
    path = ROOT / "scripts/h3wam/finalize_c58b_online_long10000.py"
    spec = importlib.util.spec_from_file_location("_test_c58b_online_finalizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


F = load_finalizer()


def contract():
    return {
        "candidate": "C58B_FASTWAM_FULL30_H3_LAYERWISE",
        "fastwam_commit": F.FASTWAM_COMMIT,
        "d0_parent_sha256": F.D0_SHA256,
        "h3_checkpoint_sha256": F.H3_SHA256,
        "h3_execution": "online_frozen_int8_per_rank_v1",
        "disk_kv_training_input": False,
        "kv_subdir": None,
        "action_horizon": 32,
        "action_shift": 5.0,
        "model_spec": {"action_layers": 30, "carrier_layers": list(F.LAYERS)},
        "action_block_to_h3_layer": list(F.LAYERS),
    }


def test_finalizer_accepts_exact_online_identity_and_30_layer_final_stage(tmp_path):
    checkpoint = (tmp_path / "c58b_online_s10000.pt").resolve()
    history = [
        {"step": step, "block_gradient_norms": [float(step)] * 30}
        for step in range(9001, 10001)
    ]
    train = {
        "event": "h3_c58b_online_frozen_h3_full30_train",
        "completed_steps": 10000,
        "world_size": 8,
        "saved_checkpoint": str(checkpoint),
        "history": history,
    }
    restore = {
        "event": "h3_c58b_online_frozen_h3_full30_train",
        "completed_steps": 10000,
        "world_size": 8,
        "loaded_checkpoint": str(checkpoint),
        "restore_probe_max_abs": 0.0,
        "training_samples": 0,
        "history": [],
    }
    F.require_train_report(train, checkpoint)
    F.require_restore_report(restore, checkpoint)
    F.require_contract(contract())


def test_finalizer_rejects_nonexact_restore_and_carrier_identity(tmp_path):
    checkpoint = (tmp_path / "c58b_online_s10000.pt").resolve()
    restore = {
        "event": "h3_c58b_online_frozen_h3_full30_train",
        "completed_steps": 10000,
        "world_size": 8,
        "loaded_checkpoint": str(checkpoint),
        "restore_probe_max_abs": 0.01,
        "training_samples": 0,
        "history": [],
    }
    with pytest.raises(ValueError, match="not bit-exact"):
        F.require_restore_report(restore, checkpoint)
    bad = contract()
    bad["action_block_to_h3_layer"] = [49] * 30
    with pytest.raises(ValueError, match="mapping mismatch"):
        F.require_contract(bad)
