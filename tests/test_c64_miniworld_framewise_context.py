import torch

from fastwam.models.h3wam.c64_miniworld_framewise_context import (
    C64MiniWorldFramewiseContextPolicy,
    H3TemporalKeyReindex,
)
from fastwam.models.h3wam.dreamwam_kv_carrier import H3DreamWAMKVCarrierPolicy
from fastwam.models.h3wam.fastwam_full_tower import (
    H3FastWAMFullTowerPolicy,
    LAYERWISE_H3_50_TO_ACTION_30,
    initialize_full_tower_from_d0,
)


INV_FREQ = torch.logspace(0, -3.75, 16, base=10.0)


def _parent():
    d0 = H3DreamWAMKVCarrierPolicy(
        enabled=True,
        carrier_layers=(9, 19, 29, 39, 49),
        carrier_source_mode="repeat_layer49",
        action_dim=2,
        proprio_dim=3,
        context_dim=16,
        hidden_dim=128,
        ffn_dim=256,
        num_heads=1,
        attn_head_dim=128,
        freq_dim=16,
    )
    model = H3FastWAMFullTowerPolicy(
        enabled=True,
        carrier_layers=LAYERWISE_H3_50_TO_ACTION_30,
        action_block_to_h3_layer=LAYERWISE_H3_50_TO_ACTION_30,
        action_dim=2,
        proprio_dim=3,
        context_dim=16,
        hidden_dim=128,
        ffn_dim=256,
        num_heads=1,
        attn_head_dim=128,
        freq_dim=16,
        num_layers=30,
    )
    initialize_full_tower_from_d0(model, d0.state_dict())
    return model


def _kv(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return {
        layer: {
            name: torch.randn(1, 3, 1, 128, generator=generator)
            for name in ("k", "v")
        }
        for layer in LAYERWISE_H3_50_TO_ACTION_30
    }


def _inputs():
    return {
        "noisy_actions": torch.randn(1, 4, 2),
        "timestep": torch.tensor([500.0]),
        "text_context": torch.randn(1, 3, 16),
        "proprio": torch.randn(1, 3),
        "text_mask": torch.ones(1, 3, dtype=torch.bool),
    }


def test_h3_temporal_reindex_matches_rotary_pair_equation():
    key = torch.randn(1, 5, 2, 128)
    module = H3TemporalKeyReindex(INV_FREQ)
    assert module(key, 0) is key
    actual = module(key, 3)
    angle = INV_FREQ * 3
    expected_first = key[..., :16] * angle.cos() - key[..., 48:64] * angle.sin()
    expected_second = key[..., 48:64] * angle.cos() + key[..., :16] * angle.sin()
    assert torch.allclose(actual[..., :16], expected_first)
    assert torch.allclose(actual[..., 48:64], expected_second)
    assert torch.equal(actual[..., 16:48], key[..., 16:48])
    assert torch.equal(actual[..., 64:], key[..., 64:])


def test_default_off_is_exact_and_frame_contract_rejects_eight_actions():
    torch.manual_seed(1)
    parent = _parent().eval()
    model = C64MiniWorldFramewiseContextPolicy(
        parent, temporal_inv_freq=INV_FREQ, context_enabled=False
    ).eval()
    values = _inputs()
    current = _kv(2)
    with torch.no_grad():
        expected = parent(video_kv_cache=current, **values)
        actual = model(video_kv_cache=current, **values)
    assert torch.equal(actual, expected)
    state = model.new_context_state("frame-contract")
    try:
        model.commit_real_observation(
            state,
            observation_kv=current,
            four_actions_before_observation=torch.randn(1, 8, 2),
        )
    except ValueError as error:
        assert "exactly four" in str(error)
    else:
        raise AssertionError("C64 accepted an eight-action averaged frame")


def test_pair_order_changes_reindexed_keys_and_action_pairing_has_gradient():
    torch.manual_seed(3)
    model = C64MiniWorldFramewiseContextPolicy(
        _parent(), temporal_inv_freq=INV_FREQ, context_enabled=True, max_cache_frames=3
    )
    state = model.new_context_state("ordered")
    model.commit_real_observation(
        state,
        observation_kv=_kv(4),
        four_actions_before_observation=None,
    )
    model.commit_real_observation(
        state,
        observation_kv=_kv(5),
        four_actions_before_observation=torch.randn(1, 4, 2),
    )
    actions = torch.randn(1, 4, 2)
    rolling = model._rolling_carrier(state, _kv(6), actions)
    assert all(item["k"].shape[1] == 9 for item in rolling.values())
    reversed_state = model.new_context_state("reversed")
    model.commit_real_observation(
        reversed_state,
        observation_kv=_kv(5),
        four_actions_before_observation=None,
    )
    model.commit_real_observation(
        reversed_state,
        observation_kv=_kv(4),
        four_actions_before_observation=state.entries[1].actions_before_observation,
    )
    reversed_rolling = model._rolling_carrier(reversed_state, _kv(6), actions)
    assert not torch.equal(
        rolling[LAYERWISE_H3_50_TO_ACTION_30[0]]["k"],
        reversed_rolling[LAYERWISE_H3_50_TO_ACTION_30[0]]["k"],
    )

    prediction = model(
        video_kv_cache=_kv(6),
        context_state=state,
        four_actions_before_current=actions,
        **_inputs(),
    )
    prediction.square().mean().backward()
    assert float(model.modulator.shared_modulation.weight.grad.norm()) > 0
    assert sum(
        float(refiner[-1].weight.grad.norm()) > 0
        for refiner in model.modulator.layer_refiners.values()
    ) >= 5


def test_sink_fifo_is_reindexed_contiguously_after_eviction():
    model = C64MiniWorldFramewiseContextPolicy(
        _parent(), temporal_inv_freq=INV_FREQ, context_enabled=True, max_cache_frames=3
    )
    state = model.new_context_state("fifo")
    for index in range(5):
        model.commit_real_observation(
            state,
            observation_kv=_kv(10 + index),
            four_actions_before_observation=(
                None if index == 0 else torch.full((1, 4, 2), float(index))
            ),
        )
    assert state.audit()["update_ids"] == [0, 3, 4]
    rolling = model._rolling_carrier(state, _kv(20), torch.ones(1, 4, 2))
    assert all(item["k"].shape[1] == 12 for item in rolling.values())
