import unittest

import torch

from fastwam.models.h3wam.int8_backbone import (
    HEADS,
    HEAD_DIM,
    HIDDEN_SIZE,
    FrozenLinear,
    FrozenRMSNorm,
    H3Int8Attention,
    _apply_rotary,
    _prepare_text_hidden_states,
)


class _FailIfCalled(torch.nn.Module):
    def forward(self, _source):
        raise AssertionError("refined context must bypass projection and refiner")


class H3Int8BackbonePrimitiveTest(unittest.TestCase):
    def test_frozen_linear_has_no_parameters(self):
        layer = FrozenLinear(torch.eye(4), torch.arange(4, dtype=torch.float32))
        self.assertEqual(dict(layer.named_parameters()), {})
        expected = (torch.arange(4, dtype=torch.float32) + 1).expand(2, -1)
        torch.testing.assert_close(layer(torch.ones(2, 4)), expected)

    def test_frozen_linear_swiglu_contract(self):
        layer = FrozenLinear(torch.eye(3))
        source = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        expected = torch.nn.functional.silu(source[:, :3]) * source[:, 3:]
        torch.testing.assert_close(layer(source, input_act="swiglu"), expected)

    def test_rms_norm_preserves_shape(self):
        layer = FrozenRMSNorm(torch.ones(8, dtype=torch.bfloat16))
        output = layer(torch.randn(2, 5, 8, dtype=torch.bfloat16))
        self.assertEqual(output.shape, (2, 5, 8))
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_rotary_leaves_tail_unchanged(self):
        source = torch.randn(1, 3, 2, 8)
        output = _apply_rotary(source, torch.ones(3, 4), torch.zeros(3, 4))
        torch.testing.assert_close(output, source)

    def test_raw_context_runs_projection_and_refiner(self):
        projection = FrozenLinear(torch.ones(HIDDEN_SIZE, 4))
        refiner = torch.nn.Identity()
        source = torch.ones(1, 3, 4)
        output = _prepare_text_hidden_states(source, projection, refiner)
        self.assertEqual(tuple(output.shape), (1, 3, HIDDEN_SIZE))
        torch.testing.assert_close(output, torch.full_like(output, 4.0))

    def test_refined_context_bypasses_projection_and_refiner(self):
        projection = FrozenLinear(torch.ones(HIDDEN_SIZE, 4))
        source = torch.randn(1, 3, HIDDEN_SIZE)
        output = _prepare_text_hidden_states(source, projection, _FailIfCalled())
        self.assertIs(output, source)

    def test_context_rejects_unknown_width(self):
        projection = FrozenLinear(torch.ones(HIDDEN_SIZE, 4))
        with self.assertRaisesRegex(ValueError, "raw Qwen.*refined H3"):
            _prepare_text_hidden_states(
                torch.zeros(1, 3, 8), projection, torch.nn.Identity()
            )

    def test_attention_can_return_true_projected_kv_without_second_qkv(self):
        attention = H3Int8Attention.__new__(H3Int8Attention)
        torch.nn.Module.__init__(attention)
        attention.qkv_proj = torch.nn.Linear(4, 3 * HEADS * HEAD_DIM, bias=False)
        attention.out_proj = torch.nn.Linear(HEADS * HEAD_DIM, 4, bias=False)
        attention.q_norm = torch.nn.Identity()
        attention.k_norm = torch.nn.Identity()
        source = torch.randn(1, 2, 4)

        default_output = attention(source, None)
        output, cache = attention(source, None, return_kv=True)

        torch.testing.assert_close(output, default_output, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(output.shape), (1, 2, 4))
        self.assertEqual(tuple(cache["k"].shape), (1, 2, HEADS, HEAD_DIM))
        self.assertEqual(tuple(cache["v"].shape), (1, 2, HEADS, HEAD_DIM))
        self.assertEqual(set(cache), {"k", "v"})


if __name__ == "__main__":
    unittest.main()
