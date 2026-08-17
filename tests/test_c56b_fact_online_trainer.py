import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load_trainer():
    path = ROOT / "scripts/h3wam/train_c56b_fact_online.py"
    spec = importlib.util.spec_from_file_location("_test_c69_action_only", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_online_trainer_preserves_full_fact_contract():
    source = (ROOT / "scripts/h3wam/train_c56b_fact_online.py").read_text()
    assert "RANK_CATEGORIES" in source
    assert source.count('"expert_demo"') >= 4
    assert '"success_rollout"' in source
    assert '"observational_failure"' in source
    assert '"causal_failure"' in source
    assert "fact_backbone_port_losses" in source
    assert "DistributedDataParallel" in source
    assert "materialize_kv_for_autograd_consumer" in source
    assert "future_state_loss_mask" in source
    assert "--expected-causal-dataset-sha256" in source
    assert "--expected-causal-observations-sha256" in source
    assert "a.expected_causal_dataset_sha256" in source
    assert '"causal_failure_dataset_sha256"' in source
    assert "optimizer.step()" in source
    assert "load_state_dict(loaded[\"model\"], strict=True)" in source
    assert "CachedDreamWAMKVDataset" not in source
    assert "kv-subdir" not in source


def test_long_launcher_is_milestoned_and_requires_fixed_c58_parent():
    source = (ROOT / "scripts/h3wam/launch_c56b_fact_online_long10000_8gpu.sh").read_text()
    assert "C58_PARENT_CHECKPOINT" in source
    assert "C58_PARENT_READY" in source
    assert "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE" in source
    assert "READY_FOR_CHILD_BRANCH_AND_LIBERO_EVAL" in source
    assert 'ready.get("checkpoint_sha256")' in source
    assert "seq 1000 1000 10000" in source
    assert "restore-check-only" in source
    assert "GO_LONG" in source
    assert "CAUSAL_FAILURE_DATASET" in source
    assert "EXPECTED_CAUSAL_DATASET_SHA256" in source


def test_formal_parent_must_be_the_fixed_s10000_checkpoint():
    source = (ROOT / "scripts/h3wam/train_c56b_fact_online.py").read_text()
    assert 'int(c58_parent.get("completed_steps", -1)) != 10000' in source
    assert "fixed online C58b s10000 layerwise arm" in source


def test_c56_watcher_is_fail_closed_and_waits_for_all_eight_gpus():
    source = (ROOT / "scripts/h3wam/watch_and_launch_c56b_after_c58b_final.sh").read_text()
    assert "c56_canary_gate" in source
    assert "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE" in source
    assert "READY_FOR_CHILD_BRANCH_AND_LIBERO_EVAL" in source
    assert 'ready.get("checkpoint_sha256")' in source
    assert "refusing duplicate launch" in source
    assert "--query-compute-apps=pid" in source
    assert 'gpu_count' in source
    assert "custom causal data requires CAUSAL_FAILURE_READY" in source
    assert "PASS_C61_FINALIZED_FACT_FAILURE_DATASET" in source
    assert "dataset_sha256" in source
    assert "PASS_C61_MATCHED_DATA_GATE" in source
    assert "C61_DATA_READY.json" in source


def test_c61_matched_arm_reuses_the_exact_c56_launcher():
    source = (ROOT / "scripts/h3wam/watch_and_launch_c56b_c61_matched.sh").read_text()
    assert "watch_and_launch_c56b_after_c58b_final.sh" in source
    assert "CAUSAL_FAILURE_READY" in source
    assert "online-long10000-c61-matched-v1" in source
    assert "base-lr" not in source
    assert "action-lr" not in source


def test_c69_action_only_is_the_exact_c67_global_action_component(monkeypatch):
    trainer = load_trainer()

    def fake_all_reduce(value, op=None):
        if value.numel() == 1:
            value.fill_(6.0)
        else:
            value.copy_(torch.tensor([6.0, 8.0, 8.0, 8.0]))

    monkeypatch.setattr(trainer.dist, "all_reduce", fake_all_reduce)
    raw_action = torch.tensor(2.0, requires_grad=True)
    losses = {
        "action_loss": raw_action,
        "future_representation_loss": torch.tensor(3.0, requires_grad=True),
        "future_state_loss": torch.tensor(4.0, requires_grad=True),
        "value_loss": torch.tensor(5.0, requires_grad=True),
    }
    targets = {
        "action_loss_mask": torch.ones(1),
        "future_loss_mask": torch.ones(1),
        "future_state_loss_mask": torch.ones(1),
        "value_loss_mask": torch.ones(1),
    }
    joint = trainer.globally_normalize_masked_losses(losses, targets, world=8)
    action_only = trainer.globally_normalize_action_only_losses(
        losses, targets, world=8
    )
    torch.testing.assert_close(
        action_only["loss"], 10.0 * joint["action_loss"], rtol=0, atol=0
    )
    joint_action_gradient = torch.autograd.grad(
        10.0 * joint["action_loss"], raw_action, retain_graph=True
    )[0]
    action_only_gradient = torch.autograd.grad(
        action_only["loss"], raw_action
    )[0]
    torch.testing.assert_close(
        action_only_gradient, joint_action_gradient, rtol=0, atol=0
    )
    assert not action_only["future_representation_loss"].requires_grad
    assert not action_only["future_state_loss"].requires_grad
    assert not action_only["value_loss"].requires_grad


class _TinyC69Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower = nn.Linear(2, 2)
        for name in (
            "future_state_encoder",
            "value_encoder",
            "future_representation_encoder",
            "future_state_decoder",
            "value_decoder",
            "future_representation_decoder",
        ):
            setattr(self, name, nn.Linear(2, 2))


def test_c69_freezes_only_auxiliary_heads_and_excludes_them_from_adamw():
    trainer = load_trainer()
    model = _TinyC69Policy()
    frozen = trainer.freeze_action_only_auxiliary_heads(model)
    assert frozen
    assert all(not dict(model.named_parameters())[name].requires_grad for name in frozen)
    assert all(parameter.requires_grad for parameter in model.tower.parameters())
    groups = trainer.optimizer_groups(model, base_lr=2e-5, action_lr=2e-4, wd=1e-4)
    optimized = {id(parameter) for group in groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model.tower.parameters()}
