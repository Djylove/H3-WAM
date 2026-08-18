from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/h3wam/evaluate_c71_lightwam_balanced80.py"
    spec = importlib.util.spec_from_file_location("_c71_eval_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_provider_input_routes_replacement_through_h3() -> None:
    module = _module()
    batch = {
        "current_h3_input": torch.tensor([1]),
        "shuffled_h3_input": torch.tensor([2]),
        "text_context": torch.tensor([3]),
        "replacement_text_context": torch.tensor([4]),
        "shuffled_h3_text_context": torch.tensor([5]),
        "text_token_tags": torch.tensor([6]),
        "replacement_text_token_tags": torch.tensor([7]),
        "shuffled_h3_text_token_tags": torch.tensor([8]),
    }
    replacement = module.provider_input(batch, "replacement_language")
    assert replacement["current_h3_input"].item() == 1
    assert replacement["text_context"].item() == 4
    assert replacement["text_token_tags"].item() == 7
    with pytest.raises(ValueError, match="unsupported"):
        module.provider_input(batch, "unknown")


def test_predict_direct_uses_zero_action_placeholders() -> None:
    module = _module()

    class Policy(torch.nn.Module):
        def forward(self, actions, timestep, **kwargs):
            assert torch.count_nonzero(actions) == 0
            assert torch.count_nonzero(timestep) == 0
            return actions + 0.25

    batch = {
        "actions": torch.ones(1, 4, 7),
        "text_context": torch.ones(1, 2, 3),
        "text_mask": torch.ones(1, 2, dtype=torch.bool),
        "proprio": torch.ones(1, 8),
    }
    prediction = module.predict_direct(Policy(), batch, {})
    assert prediction.shape == (1, 4, 7)
    assert torch.all(prediction == 0.25)
