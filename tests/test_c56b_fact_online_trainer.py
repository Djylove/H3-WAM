import importlib.util
import sys
from pathlib import Path

import pytest
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


def load_c69_finalizer():
    path = ROOT / "scripts/h3wam/finalize_c69_matched_action_only_20k.py"
    spec = importlib.util.spec_from_file_location("_test_c69_finalizer", path)
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


def test_c69_finalizer_rejects_joint_loss_or_incomplete_auxiliary_freeze():
    finalizer = load_c69_finalizer()
    contract = {
        "format": finalizer.FORMAT,
        "classification": "FACT_full_backbone_port_online_frozen_int8_h3",
        "objective_mode": "action_only",
        "rank_categories": [
            "expert_demo", "expert_demo", "expert_demo", "expert_demo",
            "success_rollout", "success_rollout", "observational_failure",
            "causal_failure",
        ],
        "loss_weights": [10.0, 0.0, 0.0, 0.0],
        "target_norm_sha256": finalizer.TARGET_NORM_SHA256,
        "h3_sha256": finalizer.H3_SHA256,
        "d0_sha256": finalizer.D0_SHA256,
        "c58_parent_sha256": finalizer.C58_SHA256,
        "causal_failure_dataset_sha256": finalizer.C60_DATASET_SHA256,
        "causal_failure_observations_sha256": finalizer.C60_OBSERVATIONS_SHA256,
        "base_lr": 2e-5, "action_lr": 2e-4, "warmup_steps": 500,
        "scheduler_horizon": 20_000, "weight_decay": 1e-4,
        "max_grad_norm": 1.0, "seed": 20260816,
        "gradient_checkpointing": True, "action_horizon": 32,
        "action_shift": 5.0, "h3_carrier_layers": list(finalizer.LAYERS),
        "h3_execution": "online_frozen_int8_per_rank_v1", "no_kv_cache": True,
        "initialization": {
            "initialization_contract": "strict_online_c58b_parent_v1",
            "c58_completed_steps": 10_000,
        },
        "frozen_auxiliary_parameters": [
            prefix + "0.weight" for prefix in finalizer.AUXILIARY_PREFIXES
        ],
    }
    for name in (
        "demo_manifest_sha256", "source_manifest_sha256", "demo_stats_sha256",
        "c48_dataset_sha256", "c48_observations_sha256",
        "c59_completed_sha256", "c59_sample_labels_sha256",
    ):
        contract[name] = "a" * 64
    finalizer.require_contract(contract)
    joint = dict(contract, loss_weights=[10.0, 1.0, 0.4, 0.4])
    with pytest.raises(ValueError, match="fixed contract"):
        finalizer.require_contract(joint)
    incomplete = dict(
        contract,
        frozen_auxiliary_parameters=contract["frozen_auxiliary_parameters"][:-1],
    )
    with pytest.raises(ValueError, match="auxiliary freeze"):
        finalizer.require_contract(incomplete)


def test_c69_long_launcher_keeps_the_c67_sample_and_schedule_contract():
    source = (
        ROOT / "scripts/h3wam/launch_c69_matched_action_only_20k_8gpu.sh"
    ).read_text()
    assert "--objective-mode action_only" in source
    assert "--scheduler-horizon 20000" in source
    assert "seq 1000 1000 20000" in source
    assert "--nproc-per-node 8" in source
    assert "restore-check-only" in source
    assert "C69_CANARY_GO_LONG" in source
    assert "source_freeze_sha256" in source


def test_c69_preview_queue_is_read_only_and_cannot_select_a_checkpoint():
    queue = (
        ROOT / "scripts/h3wam/launch_c69_action_only_milestone_preview_queue.sh"
    ).read_text()
    evaluator = (
        ROOT / "scripts/h3wam/evaluate_c67_fact_milestone_balanced80.py"
    ).read_text()
    assert "--variant c69" in queue
    assert "c69_action_only_s${milestone}.pt" in queue
    assert "PREVIEW_ONLY_PENDING_TRAINING_COMPLETE_REBIND" in queue
    assert "freeze_c67_rollout_source.py" in queue
    assert "train_c56b_fact_online.py" not in queue
    assert "objective_mode" in evaluator
    assert "[10.0, 0.0, 0.0, 0.0]" in evaluator
    assert "C69 final rebinding requires the fixed cross-arm aggregator" in evaluator
