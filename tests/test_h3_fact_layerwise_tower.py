import copy

import torch
from torch import nn

from fastwam.models.h3wam.dreamwam_kv_carrier import H3DreamWAMKVCarrierPolicy
from fastwam.models.h3wam.fact_layerwise_tower import H3FACTLayerwiseTowerPolicy
from fastwam.models.h3wam.fact_backbone_port import fact_backbone_port_losses
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


def build_model():
    tower = H3FastWAMFullTowerPolicy(
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
    initialize_full_tower_from_d0(tower, build_d0().state_dict())
    return H3FACTLayerwiseTowerPolicy(
        tower, future_state_dim=3, future_representation_dim=5
    )


def inputs():
    torch.manual_seed(6401)
    return {
        "noisy_actions": torch.randn(2, 4, 2),
        "timestep": torch.tensor([100.0, 700.0]),
        "clean_actions": torch.randn(2, 4, 2),
        "noisy_future_state": torch.randn(2, 1, 3),
        "noisy_value": torch.randn(2, 1, 1),
        "noisy_future_representation": torch.randn(2, 3, 5),
        "text_context": torch.randn(2, 5, 6),
        "proprio": torch.randn(2, 3),
        "text_mask": torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]]
        ),
        "video_kv_cache": {
            layer: {
                "k": torch.randn(2, 5, 2, 4),
                "v": torch.randn(2, 5, 2, 4),
            }
            for layer in LAYERWISE_H3_50_TO_ACTION_30
        },
    }


def action_inputs(sample):
    return {
        key: sample[key]
        for key in (
            "noisy_actions",
            "timestep",
            "text_context",
            "proprio",
            "text_mask",
            "video_kv_cache",
        )
    }


def test_stage1_is_exact_c58b_path_and_has_no_independent_transformer():
    model = build_model().eval()
    sample = inputs()
    with torch.no_grad():
        expected = model.tower(**action_inputs(sample))
        actual = model.forward_action(**action_inputs(sample))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not any(isinstance(module, nn.TransformerEncoder) for module in model.modules())


def test_joint_forward_uses_all_thirty_shared_blocks_and_action_is_causal():
    model = build_model().eval()
    sample = inputs()
    counts = [0] * 30
    handles = []
    for index, block in enumerate(model.shared_blocks):
        handles.append(
            block.self_attn.o.register_forward_hook(
                lambda _module, _args, _output, position=index: counts.__setitem__(
                    position, counts[position] + 1
                )
            )
        )
    with torch.no_grad():
        baseline = model(**sample)
        changed = dict(sample)
        changed["clean_actions"] = sample["clean_actions"] + 100.0
        changed["noisy_future_state"] = sample["noisy_future_state"] - 50.0
        changed["noisy_value"] = sample["noisy_value"] + 80.0
        changed["noisy_future_representation"] = (
            sample["noisy_future_representation"] * -30.0
        )
        perturbed = model(**changed)
    for handle in handles:
        handle.remove()
    assert counts == [2] * 30
    torch.testing.assert_close(
        baseline["action"], perturbed["action"], rtol=0, atol=0
    )
    assert baseline["layout"].future_representation_tokens == 3


def test_future_only_loss_updates_every_action_generation_block_and_shared_encoder():
    model = build_model()
    result = model(**inputs())
    loss = (
        result["future_state"].square().mean()
        + result["value"].square().mean()
        + result["future_representation"].square().mean()
    )
    loss.backward()
    block_norms = [
        float(block.self_attn.o.weight.grad.norm()) for block in model.shared_blocks
    ]
    assert len(block_norms) == 30
    assert all(value > 0 for value in block_norms)
    assert model.tower.action_expert.action_encoder.weight.grad is not None
    assert float(model.tower.action_expert.action_encoder.weight.grad.norm()) > 0
    assert model.tower.action_expert.head.weight.grad is None


def test_failed_episode_masks_action_but_keeps_future_and_value_objectives():
    model = build_model()
    result = model(**inputs())
    losses = fact_backbone_port_losses(
        result,
        action_target=torch.zeros_like(result["action"]),
        future_state_target=torch.zeros_like(result["future_state"]),
        value_target=torch.zeros_like(result["value"]),
        future_representation_target=torch.zeros_like(
            result["future_representation"]
        ),
        action_is_pad=torch.zeros(2, 4, dtype=torch.bool),
        action_loss_mask=torch.zeros(2),
        future_loss_mask=torch.ones(2),
        value_loss_mask=torch.ones(2),
    )
    assert float(losses["action_loss"].detach()) == 0.0
    assert float(losses["future_representation_loss"]) > 0
    assert float(losses["future_state_loss"]) > 0
    assert float(losses["value_loss"]) > 0
    losses["loss"].backward()
    head_grad = model.tower.action_expert.head.weight.grad
    assert head_grad is not None and float(head_grad.norm()) == 0.0
    assert all(block.self_attn.o.weight.grad is not None for block in model.shared_blocks)


def test_strict_restore_reproduces_all_tracks_bit_exact():
    model = build_model().eval()
    sample = inputs()
    with torch.no_grad():
        expected = model(**sample)
    state = copy.deepcopy(model.state_dict())
    restored = build_model().eval()
    restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        actual = restored(**sample)
    for key in ("action", "future_state", "value", "future_representation"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
