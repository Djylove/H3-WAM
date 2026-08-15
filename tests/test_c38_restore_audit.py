from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "c38_restore_audit_test_module",
    ROOT / "scripts/h3wam/audit_c38_restore_and_reaggregate.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_eval_to_eval_restore_is_bit_exact(tmp_path):
    from fastwam.models.h3wam.fact_lite_consequence import TemporalFutureH3ConsequenceModel

    kwargs = {
        "state_dim": 8, "action_dim": 7, "action_horizon": 4,
        "actions_per_latent": 2, "h3_feature_dim": 16,
        "target_dim": 4, "hidden_dim": 8, "num_heads": 2,
    }
    model = TemporalFutureH3ConsequenceModel(**kwargs)
    path = tmp_path / "checkpoint.pt"
    torch.save({
        "model_variant": "temporal", "model_kwargs": kwargs,
        "models": {"conditioned": model.state_dict()},
    }, path)
    result = MODULE.audit_checkpoint(path, 42, torch.device("cpu"))
    assert result["restored_weights_exact"] is True
    assert result["eval_to_eval_max_abs"] == 0.0
    assert result["passed"] is True
