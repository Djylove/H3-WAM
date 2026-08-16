import importlib.util
import sys
from pathlib import Path

import torch

from fastwam.models.h3wam.dreamwam_kv_carrier import H3DreamWAMKVCarrierPolicy
from fastwam.models.h3wam.fastwam_full_tower import (
    H3FastWAMFullTowerPolicy,
    LAYERWISE_H3_50_TO_ACTION_30,
    initialize_full_tower_from_d0,
)


def build_d0():
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


def build_layerwise():
    return H3FastWAMFullTowerPolicy(
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


def inputs():
    return {
        "noisy_actions": torch.randn(2, 4, 2),
        "timestep": torch.tensor([100.0, 700.0]),
        "text_context": torch.randn(2, 5, 6),
        "proprio": torch.randn(2, 3),
        "text_mask": torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]]
        ),
    }


def cache():
    return {
        layer: {
            "k": torch.randn(2, 5, 2, 4),
            "v": torch.randn(2, 5, 2, 4),
        }
        for layer in LAYERWISE_H3_50_TO_ACTION_30
    }


def test_uniform_mapping_is_monotonic_unique_and_covers_endpoints():
    assert len(LAYERWISE_H3_50_TO_ACTION_30) == 30
    assert len(set(LAYERWISE_H3_50_TO_ACTION_30)) == 30
    assert LAYERWISE_H3_50_TO_ACTION_30[0] == 0
    assert LAYERWISE_H3_50_TO_ACTION_30[-1] == 49
    assert all(
        left < right
        for left, right in zip(
            LAYERWISE_H3_50_TO_ACTION_30,
            LAYERWISE_H3_50_TO_ACTION_30[1:],
        )
    )


def test_each_action_block_is_sensitive_to_its_own_h3_layer():
    torch.manual_seed(71)
    model = build_layerwise().eval()
    initialize_full_tower_from_d0(model, build_d0().state_dict())
    sample = inputs()
    base_cache = cache()
    with torch.no_grad():
        baseline = model(video_kv_cache=base_cache, **sample)
        changed_cache = {
            layer: {name: tensor.clone() for name, tensor in item.items()}
            for layer, item in base_cache.items()
        }
        changed_cache[0]["v"].add_(2.0)
        changed_early = model(video_kv_cache=changed_cache, **sample)
        changed_cache = {
            layer: {name: tensor.clone() for name, tensor in item.items()}
            for layer, item in base_cache.items()
        }
        changed_cache[49]["v"].add_(2.0)
        changed_late = model(video_kv_cache=changed_cache, **sample)
    assert not torch.equal(baseline, changed_early)
    assert not torch.equal(baseline, changed_late)


def test_layerwise_backward_reaches_all_thirty_blocks():
    torch.manual_seed(72)
    model = build_layerwise()
    initialize_full_tower_from_d0(model, build_d0().state_dict())
    model(video_kv_cache=cache(), **inputs()).square().mean().backward()
    norms = [
        float(block.self_attn.o.weight.grad.norm())
        for block in model.action_expert.blocks
    ]
    assert len(norms) == 30
    assert all(value > 0 for value in norms)


def test_layerwise_equals_repeat49_when_every_layer_has_layer49_values():
    torch.manual_seed(73)
    d0 = build_d0()
    layerwise = build_layerwise().eval()
    repeat = H3FastWAMFullTowerPolicy(
        enabled=True,
        carrier_layers=(9, 19, 29, 39, 49),
        action_dim=2,
        proprio_dim=3,
        context_dim=6,
        hidden_dim=8,
        ffn_dim=16,
        num_heads=2,
        attn_head_dim=4,
        freq_dim=8,
        num_layers=30,
    ).eval()
    initialize_full_tower_from_d0(layerwise, d0.state_dict())
    initialize_full_tower_from_d0(repeat, d0.state_dict())
    sample = inputs()
    final = {
        "k": torch.randn(2, 5, 2, 4),
        "v": torch.randn(2, 5, 2, 4),
    }
    repeat_cache = {
        layer: {name: tensor.clone() for name, tensor in final.items()}
        for layer in (9, 19, 29, 39, 49)
    }
    layerwise_cache = {
        layer: {name: tensor.clone() for name, tensor in final.items()}
        for layer in LAYERWISE_H3_50_TO_ACTION_30
    }
    with torch.no_grad():
        expected = repeat(video_kv_cache=repeat_cache, **sample)
        actual = layerwise(video_kv_cache=layerwise_cache, **sample)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_layerwise_rejects_incomplete_or_aliased_cache():
    model = build_layerwise()
    sample = inputs()
    values = cache()
    values.pop(0)
    try:
        model(video_kv_cache=values, **sample)
    except ValueError as error:
        assert "exactly match" in str(error)
    else:
        raise AssertionError("incomplete cache was accepted")

    values = cache()
    values[2]["k"] = values[0]["k"]
    try:
        model(video_kv_cache=values, **sample)
    except ValueError as error:
        assert "independent storage" in str(error)
    else:
        raise AssertionError("aliased cache was accepted")


def test_existing_int8_precompute_path_writes_exact_thirty_layer_schema():
    path = Path(__file__).resolve().parents[1] / "scripts/h3wam/precompute_h3_int8_features.py"
    spec = importlib.util.spec_from_file_location("_test_c58b_precompute", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    captured = {
        layer: {
            "k": torch.randn(1, 64, 2, 4),
            "v": torch.randn(1, 64, 2, 4),
        }
        for layer in LAYERWISE_H3_50_TO_ACTION_30
    }
    prepared = module.prepare_dreamwam_kv_cache(
        captured,
        layers=LAYERWISE_H3_50_TO_ACTION_30,
        token_count=32,
    )
    assert tuple(prepared) == LAYERWISE_H3_50_TO_ACTION_30
    tensors = [tensor for item in prepared.values() for tensor in item.values()]
    assert all(tuple(tensor.shape) == (32, 2, 4) for tensor in tensors)
    assert all(tensor.dtype == torch.bfloat16 for tensor in tensors)
    assert len({tensor.untyped_storage().data_ptr() for tensor in tensors}) == 60
