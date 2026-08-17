import sys
from pathlib import Path
import importlib.util

import pytest
import torch

from fastwam.models.h3wam.c62_miniworld_context import (
    C62MiniWorldRollingContextPolicy,
    MiniWorldRollingContextState,
    verify_miniworld_execution_source,
)
from fastwam.models.h3wam.dreamwam_kv_carrier import H3DreamWAMKVCarrierPolicy
from fastwam.models.h3wam.fastwam_full_tower import (
    H3FastWAMFullTowerPolicy,
    LAYERWISE_H3_50_TO_ACTION_30,
    initialize_full_tower_from_d0,
)


ROOT = Path(__file__).resolve().parents[1]


def _d0():
    return H3DreamWAMKVCarrierPolicy(
        enabled=True,
        carrier_layers=(9, 19, 29, 39, 49),
        carrier_source_mode="repeat_layer49",
        action_dim=2,
        proprio_dim=3,
        context_dim=6,
        hidden_dim=8,
        ffn_dim=16,
        num_heads=2,
        attn_head_dim=4,
        freq_dim=8,
    )


def _parent():
    model = H3FastWAMFullTowerPolicy(
        enabled=True,
        carrier_layers=LAYERWISE_H3_50_TO_ACTION_30,
        action_block_to_h3_layer=LAYERWISE_H3_50_TO_ACTION_30,
        action_dim=2,
        proprio_dim=3,
        context_dim=6,
        hidden_dim=8,
        ffn_dim=16,
        num_heads=2,
        attn_head_dim=4,
        freq_dim=8,
        num_layers=30,
    )
    initialize_full_tower_from_d0(model, _d0().state_dict())
    return model


def _kv(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return {
        layer: {
            "k": torch.randn(1, 3, 2, 4, generator=generator),
            "v": torch.randn(1, 3, 2, 4, generator=generator),
        }
        for layer in LAYERWISE_H3_50_TO_ACTION_30
    }


def _inputs():
    return {
        "noisy_actions": torch.randn(1, 4, 2),
        "timestep": torch.tensor([500.0]),
        "text_context": torch.randn(1, 5, 6),
        "proprio": torch.randn(1, 3),
        "text_mask": torch.tensor([[True, True, True, False, False]]),
    }


def test_fixed_official_source_import_and_semantics():
    source = ROOT / "third_party/MiniWorld"
    sys.path.insert(0, str(source))
    try:
        report = verify_miniworld_execution_source(source)
    finally:
        sys.path.remove(str(source))
    assert report["commit"].startswith("e484206")
    assert report["action_alignment_shape"] == [1, 3, 28]


def test_default_off_is_bit_exact_to_c58_parent():
    torch.manual_seed(10)
    parent = _parent().eval()
    model = C62MiniWorldRollingContextPolicy(parent, context_enabled=False).eval()
    values = _inputs()
    current = _kv(11)
    with torch.no_grad():
        expected = parent(video_kv_cache=current, **values)
        actual = model(video_kv_cache=current, **values)
    assert torch.equal(actual, expected)


def test_sink_fifo_causal_order_and_snapshot_restore():
    state = MiniWorldRollingContextState(
        layers=LAYERWISE_H3_50_TO_ACTION_30,
        max_cache_chunks=3,
        episode_key="ep0",
    )
    for index in range(5):
        state.append(
            _kv(index),
            None if index == 0 else torch.full((1, 8, 2), float(index)),
        )
    # Persistent sink 0 plus the two newest chunks; no future or picked middle.
    assert state.audit()["update_ids"] == [0, 3, 4]
    restored = MiniWorldRollingContextState.from_snapshot(state.snapshot())
    assert restored.audit() == state.audit()
    for expected, actual in zip(state.entries, restored.entries, strict=True):
        for layer in LAYERWISE_H3_50_TO_ACTION_30:
            for name in ("k", "v"):
                assert torch.equal(
                    expected.observation_kv[layer][name],
                    actual.observation_kv[layer][name],
                )


def test_context_shape_action_gradient_and_strict_model_restore():
    torch.manual_seed(12)
    model = C62MiniWorldRollingContextPolicy(
        _parent(), context_enabled=True, max_cache_chunks=3
    )
    state = model.new_context_state("ep1")
    model.commit_real_observation(
        state, observation_kv=_kv(13), actions_before_observation=None
    )
    model.commit_real_observation(
        state,
        observation_kv=_kv(14),
        actions_before_observation=torch.randn(1, 8, 2),
    )
    current = _kv(15)
    values = _inputs()
    rolling = model._rolling_carrier(
        state, current, torch.randn(1, 8, 2)
    )
    assert all(item["k"].shape[1] == 9 for item in rolling.values())
    prediction = model(
        video_kv_cache=current,
        context_state=state,
        actions_before_current=torch.randn(1, 8, 2),
        **values,
    )
    prediction.square().mean().backward()
    assert model.modulator.shared_modulation.weight.grad is not None
    assert float(model.modulator.shared_modulation.weight.grad.norm()) > 0
    active_refiners = sum(
        float(refiner[-1].weight.grad.norm()) > 0
        for refiner in model.modulator.layer_refiners.values()
    )
    # The synthetic parent is the exact step-zero 5->30 expansion, so its 25
    # identity residual blocks intentionally cannot send carrier gradients yet.
    assert active_refiners >= 5
    assert all(
        float(block.self_attn.o.weight.grad.norm()) > 0
        for block in model.parent.action_expert.blocks
    )

    payload = {key: value.detach().clone() for key, value in model.state_dict().items()}
    fresh = C62MiniWorldRollingContextPolicy(
        _parent(), context_enabled=True, max_cache_chunks=3
    ).eval()
    assert not fresh.load_state_dict(payload, strict=True).missing_keys
    restored_state = MiniWorldRollingContextState.from_snapshot(state.snapshot())
    model.eval()
    with torch.no_grad():
        expected = model(
            video_kv_cache=current,
            context_state=state,
            actions_before_current=torch.ones(1, 8, 2),
            **values,
        )
        actual = fresh(
            video_kv_cache=current,
            context_state=restored_state,
            actions_before_current=torch.ones(1, 8, 2),
            **values,
        )
    assert torch.equal(actual, expected)


def test_frozen_h3_gradient_boundary_keeps_bridge_trainable():
    model = C62MiniWorldRollingContextPolicy(
        _parent(), context_enabled=True, max_cache_chunks=2
    )
    state = model.new_context_state("gradient-boundary")
    history = _kv(31)
    current = _kv(32)
    for item in (*history.values(), *current.values()):
        item["k"].requires_grad_(True)
        item["v"].requires_grad_(True)
    model.commit_real_observation(
        state,
        observation_kv=history,
        actions_before_observation=torch.randn(1, 8, 2, requires_grad=True),
    )
    rolling = model._rolling_carrier(
        state,
        current,
        torch.randn(1, 8, 2, requires_grad=True),
    )
    assert all(
        not tensor.requires_grad
        for entry in state.entries
        for item in entry.observation_kv.values()
        for tensor in item.values()
    )
    assert all(
        tensor.requires_grad
        for item in rolling.values()
        for tensor in item.values()
    )
    sum(tensor.float().mean() for item in rolling.values() for tensor in item.values()).backward()
    assert all(item["k"].grad is None and item["v"].grad is None for item in history.values())
    assert all(item["k"].grad is None and item["v"].grad is None for item in current.values())
    assert model.modulator.shared_modulation.weight.grad is not None


def test_rejects_non_aligned_action_history_and_context_args_when_disabled():
    model = C62MiniWorldRollingContextPolicy(_parent(), context_enabled=False)
    state = model.new_context_state("ep")
    with pytest.raises(ValueError, match="4n"):
        state.append(_kv(1), torch.randn(1, 7, 2))
    with pytest.raises(ValueError, match="require context"):
        model(video_kv_cache=_kv(2), context_state=state, **_inputs())


def test_real_c58_checkpoint_validator_uses_serialized_schema_fields():
    path = ROOT / "scripts/h3wam/probe_c62_miniworld_context.py"
    spec = importlib.util.spec_from_file_location("_c62_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    layers = LAYERWISE_H3_50_TO_ACTION_30
    payload = {
        "schema_version": 1,
        "completed_steps": 10_000,
        "contract": {
            "candidate": "C58B_FASTWAM_FULL30_H3_LAYERWISE",
            "classification": "action-only-on-frozen-layerwise-h3-kv_backbone_port",
            "fastwam_commit": module.EXPECTED_FASTWAM_COMMIT,
            "fastwam_action_dit_sha256": module.EXPECTED_ACTION_DIT_SHA256,
            "carrier_source_mode": "uniform_h3_50_to_action30",
            "kv_layers": list(layers),
            "action_block_to_h3_layer": list(layers),
            "model_spec": {
                "carrier_layers": layers,
                "action_layers": 30,
                "carrier_source_mode": "uniform_h3_50_to_action30",
            },
            "h3_execution": "online_frozen_int8_per_rank_v1",
            "disk_kv_training_input": False,
        },
    }
    assert all(module.validate_c58_parent_payload(payload).values())
    payload["contract"]["kv_layers"] = list(layers[:-1])
    with pytest.raises(ValueError, match="kv_layers"):
        module.validate_c58_parent_payload(payload)
