from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _trainer():
    path = Path(__file__).resolve().parents[1] / "scripts/h3wam/train_c71_lightwam_online.py"
    spec = importlib.util.spec_from_file_location("_c71_trainer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_masked_direct_action_mse_ignores_padding() -> None:
    module = _trainer()
    prediction = torch.tensor([[[1.0], [99.0]]])
    target = torch.zeros_like(prediction)
    assert module.masked_direct_action_mse(prediction, target, torch.tensor([[False, True]])).item() == 1.0


def test_scheduler_uses_warmup_then_cosine_floor() -> None:
    module = _trainer()
    assert module.scheduler_factor(0, warmup_steps=10, horizon=100, minimum_ratio=0.01) == pytest.approx(0.1)
    assert module.scheduler_factor(10, warmup_steps=10, horizon=100, minimum_ratio=0.01) == pytest.approx(1.0)
    assert module.scheduler_factor(100, warmup_steps=10, horizon=100, minimum_ratio=0.01) == pytest.approx(0.01)


def test_checkpoint_contract_rejects_mutation() -> None:
    module = _trainer()
    contract = {"objective": "direct"}
    payload = {
        "schema_version": 1,
        "completed_steps": 10,
        "model": {},
        "optimizer": {},
        "lr_scheduler": {},
        "contract": contract,
        "probe_prediction": torch.zeros(1),
        "probe_sample_ids": ["x"],
        "rng_states": [],
        "data_state": {},
    }
    module.validate_checkpoint(payload, contract)
    with pytest.raises(ValueError, match="contract"):
        module.validate_checkpoint(payload, {"objective": "flow"})
