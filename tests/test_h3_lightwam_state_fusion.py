import tempfile
from pathlib import Path

import pytest
import torch

from fastwam.models.h3wam.lightwam_state_fusion import (
    LIGHTWAM_COMMIT,
    LIGHTWAM_H3_CARRIER_LAYERS,
    H3LightWAMStateFusionPolicy,
    _load_pinned_lightwam_state_fusion,
)


TEST_LAYERS = (1, 2, 3)


def build_policy(*, enabled: bool = True) -> H3LightWAMStateFusionPolicy:
    return H3LightWAMStateFusionPolicy(
        enabled=enabled,
        carrier_layers=TEST_LAYERS,
        action_dim=2,
        proprio_dim=3,
        num_heads=2,
        attn_head_dim=4,
        per_layer_dim=12,
        trunk_dim=16,
        num_trunk_blocks=1,
        step_pos_dim=8,
        token_pooling_num_queries=4,
        token_pooling_num_heads=2,
    )


def make_cache(*, batch: int = 2, tokens: int = 5):
    return {
        layer: {
            "k": torch.randn(batch, tokens, 2, 4),
            "v": torch.randn(batch, tokens, 2, 4),
        }
        for layer in TEST_LAYERS
    }


def make_inputs():
    return {
        "noisy_actions": torch.randn(2, 4, 2),
        "timestep": torch.tensor([100.0, 700.0]),
        "text_context": torch.randn(2, 5, 6),
        "text_mask": torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]]
        ),
        "proprio": torch.randn(2, 3),
    }


def test_official_source_is_pinned_and_mapping_is_declared():
    upstream = _load_pinned_lightwam_state_fusion()
    assert upstream.__module__ == (
        "_h3wam_lightwam_b2785f66.state_fusion_action_expert"
    )
    assert LIGHTWAM_COMMIT == "b2785f66e13fd9987e94ae1ecc1c441d5059c9ae"
    assert LIGHTWAM_H3_CARRIER_LAYERS == (14, 27, 41)


def test_disabled_default_has_no_parameters_or_forward():
    policy = build_policy(enabled=False)
    assert sum(parameter.numel() for parameter in policy.parameters()) == 0
    with pytest.raises(RuntimeError, match="disabled"):
        policy(video_kv_cache=make_cache(), **make_inputs())


def test_direct_action_shape_and_noise_timestep_invariance():
    torch.manual_seed(11)
    policy = build_policy().eval()
    cache = make_cache()
    inputs = make_inputs()
    with torch.no_grad():
        expected = policy(video_kv_cache=cache, **inputs)
        changed = dict(inputs)
        changed["noisy_actions"] = inputs["noisy_actions"] + 100.0
        changed["timestep"] = torch.tensor([900.0, 10.0])
        actual = policy(video_kv_cache=cache, **changed)
    assert expected.shape == (2, 4, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_visual_and_proprio_paths_change_output_and_receive_gradients():
    torch.manual_seed(17)
    policy = build_policy()
    cache = make_cache()
    inputs = make_inputs()
    baseline = policy(video_kv_cache=cache, **inputs)
    changed_cache = {
        layer: {name: value.clone() for name, value in item.items()}
        for layer, item in cache.items()
    }
    changed_cache[2]["v"].add_(1.5)
    visual = policy(video_kv_cache=changed_cache, **inputs)
    changed_inputs = dict(inputs)
    changed_inputs["proprio"] = inputs["proprio"] + 2.0
    proprio = policy(video_kv_cache=cache, **changed_inputs)
    assert not torch.equal(visual, baseline)
    assert not torch.equal(proprio, baseline)

    baseline.square().mean().backward()
    assert policy.proprio_encoder is not None
    assert policy.proprio_encoder.weight.grad is not None
    assert float(policy.proprio_encoder.weight.grad.norm()) > 0
    assert policy.state_fusion_action_expert is not None
    first_pooler = policy.state_fusion_action_expert.layer_poolers[0]["backbone"]
    assert first_pooler.query_tokens.grad is not None
    assert float(first_pooler.query_tokens.grad.norm()) > 0
    assert policy.state_fusion_action_expert.output[-1].weight.grad is not None


def test_strict_restore_is_exact():
    torch.manual_seed(23)
    policy = build_policy().eval()
    cache = make_cache()
    inputs = make_inputs()
    with torch.no_grad():
        expected = policy(video_kv_cache=cache, **inputs)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "lightwam.pt"
        torch.save({"model": policy.state_dict()}, path)
        restored = build_policy().eval()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        restored.load_state_dict(payload["model"], strict=True)
        with torch.no_grad():
            actual = restored(video_kv_cache=cache, **inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cache_contract_rejects_missing_layers_and_aliases():
    policy = build_policy()
    inputs = make_inputs()
    cache = make_cache()
    with pytest.raises(ValueError, match="exactly match"):
        policy(video_kv_cache={1: cache[1]}, **inputs)
    aliased = make_cache()
    aliased[2]["v"] = aliased[1]["v"]
    with pytest.raises(ValueError, match="independent storage"):
        policy(video_kv_cache=aliased, **inputs)
