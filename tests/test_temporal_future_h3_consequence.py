import torch

from fastwam.models.h3wam.fact_lite_consequence import (
    TemporalFutureH3ConsequenceModel,
)


def make_model():
    return TemporalFutureH3ConsequenceModel(
        state_dim=8, action_dim=7, action_horizon=32,
        actions_per_latent=4, h3_feature_dim=24,
        target_dim=12, hidden_dim=32, num_heads=4,
    )


def test_temporal_consequence_is_finite_and_order_sensitive():
    torch.manual_seed(7)
    model = make_model()
    state = torch.randn(3, 8)
    features = torch.randn(3, 5, 24)
    actions = torch.randn(3, 32, 7)
    direct = model(state, features, actions)
    reversed_actions = model(state, features, actions.flip(1))
    assert direct.shape == (3, 12)
    assert torch.isfinite(direct).all()
    assert not torch.allclose(direct, reversed_actions)


def test_temporal_consequence_detaches_candidate_actions_but_trains_adapter():
    torch.manual_seed(11)
    model = make_model()
    state = torch.randn(2, 8)
    features = torch.randn(2, 5, 24)
    actions = torch.randn(2, 32, 7, requires_grad=True)
    loss = model(state, features, actions).square().mean()
    loss.backward()
    assert actions.grad is None
    grad = model.action_encoder[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_temporal_consequence_rejects_non_aligned_horizon():
    try:
        TemporalFutureH3ConsequenceModel(
            action_horizon=30, actions_per_latent=4,
        )
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("non-aligned action horizon was accepted")
