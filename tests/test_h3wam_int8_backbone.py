import unittest

import torch

from fastwam.models.h3wam.int8_backbone import FrozenLinear, FrozenRMSNorm, _apply_rotary


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


if __name__ == "__main__":
    unittest.main()
