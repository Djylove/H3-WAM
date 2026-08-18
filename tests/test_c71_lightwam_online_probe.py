import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def load_probe():
    path = Path(__file__).resolve().parents[1] / "scripts/h3wam/probe_c71_lightwam_online.py"
    spec = importlib.util.spec_from_file_location("_c71_lightwam_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROBE = load_probe()


def test_masked_direct_action_mse_ignores_padding():
    prediction = torch.tensor([[[1.0, 2.0], [100.0, 200.0]]])
    target = torch.zeros_like(prediction)
    pad = torch.tensor([[False, True]])
    loss = PROBE.masked_direct_action_mse(prediction, target, pad)
    assert float(loss) == pytest.approx((1.0 + 4.0) / 2.0)


def test_masked_direct_action_mse_rejects_bad_contracts():
    prediction = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="share"):
        PROBE.masked_direct_action_mse(prediction, torch.zeros(1, 2, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match=r"\[B,T\]"):
        PROBE.masked_direct_action_mse(prediction, prediction, torch.zeros(1, 2, 1))
    with pytest.raises(ValueError, match="valid scalar"):
        PROBE.masked_direct_action_mse(
            prediction, prediction, torch.ones(1, 2, dtype=torch.bool)
        )


def test_probe_source_keeps_zero_step_zero_checkpoint_claim_boundary():
    source = Path(PROBE.__file__).read_text(encoding="utf-8")
    assert '"optimizer_steps": 0' in source
    assert '"checkpoint": None' in source
    assert '"permission": "PROBE_ONLY"' in source
    assert '"effect_status": "NOT_EVIDENCE_READY"' in source
    assert "optimizer.step" not in source
    assert "torch.save" not in source
