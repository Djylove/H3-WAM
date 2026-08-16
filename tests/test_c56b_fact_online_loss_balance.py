import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_probe():
    path = ROOT / "scripts/h3wam/probe_c56b_fact_online_loss_balance.py"
    spec = importlib.util.spec_from_file_location("_test_c56b_online_balance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BALANCE = load_probe()


def test_selected_gradient_norm_handles_zero_and_nonzero_paths():
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    zero = BALANCE.gradient_norm((parameter * 0).sum(), [parameter])
    nonzero = BALANCE.gradient_norm(parameter.square().sum(), [parameter])
    assert zero == 0.0
    assert nonzero > 0.0


def test_balance_is_online_mixed_and_does_not_step_or_checkpoint():
    source = (ROOT / "scripts/h3wam/probe_c56b_fact_online_loss_balance.py").read_text()
    assert 'split="train"' in source
    assert 'split="validation"' not in source
    assert "expert_demo" in source
    assert "success_rollout" in source
    assert "observational_failure" in source
    assert "causal_failure" in source
    assert "materialize_kv_for_autograd_consumer" in source
    assert "optimizer.step" not in source
    assert "torch.save" not in source
