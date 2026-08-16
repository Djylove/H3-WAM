import pytest
import torch

from fastwam.models.h3wam.c58_online_training import (
    collate_c58_online,
    materialize_frozen_kv,
)


def sample(sample_id: str = "s0"):
    return {
        "sample_id": sample_id,
        "actions": torch.randn(32, 7),
        "proprio": torch.randn(8),
        "action_is_pad": torch.zeros(32, dtype=torch.bool),
        "text_context": torch.randn(9, 5120),
        "text_token_tags": torch.arange(9),
        "current_h3_input": torch.randn(24, 1, 14, 28),
    }


def test_online_collate_preserves_single_rank_h3_layout():
    batch = collate_c58_online([sample()])
    assert batch["sample_ids"] == ["s0"]
    assert batch["current_h3_input"].shape == (1, 24, 1, 14, 28)
    assert batch["text_context"].shape == (1, 9, 5120)
    assert batch["text_token_tags"].shape == (1, 9)
    assert batch["actions"].shape == (1, 32, 7)
    assert batch["text_mask"].all()
    with pytest.raises(ValueError, match="batch size 1"):
        collate_c58_online([sample("a"), sample("b")])


def test_materialized_frozen_kv_is_exact_and_backward_compatible():
    with torch.inference_mode():
        source = {0: {"k": torch.randn(1, 4), "v": torch.randn(1, 4)}}
    result = materialize_frozen_kv(source)
    for name in ("k", "v"):
        assert not torch.is_inference(result[0][name])
        torch.testing.assert_close(result[0][name], source[0][name], rtol=0, atol=0)
    action = torch.randn(1, 4, requires_grad=True)
    (action * result[0]["k"]).sum().backward()
    assert action.grad is not None
