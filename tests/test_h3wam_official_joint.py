import unittest

import torch
from torch import nn

from fastwam.models.h3wam import (
    H3BlockAttentionMask,
    H3OfficialFeatureCapture,
    build_h3_observation_attention_mask,
)


class FakeH3Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.last_mask = None

    def forward(self, hidden, attention_mask=None):
        self.last_mask = attention_mask
        return hidden * self.scale + 1.0


class OfficialH3JointTest(unittest.TestCase):
    def test_observation_rows_cannot_read_target_rows(self):
        mask = build_h3_observation_attention_mask(
            sequence_length=8,
            text_indices=torch.tensor([0, 1]),
            condition_video_indices=torch.tensor([2, 3]),
            device="cpu",
        )
        self.assertEqual(tuple(mask.shape), (1, 1, 8, 8))
        self.assertTrue(bool(mask[0, 0, 2, 1]))
        self.assertFalse(bool(mask[0, 0, 2, 4]))
        self.assertFalse(bool(mask[0, 0, 0, 7]))
        self.assertTrue(bool(mask[0, 0, 7, 4]))

    def test_hooks_apply_mask_and_preserve_action_gradient(self):
        blocks = nn.ModuleList([FakeH3Block() for _ in range(3)])
        masker = H3BlockAttentionMask(blocks)
        mask = torch.ones(1, 1, 6, 6, dtype=torch.bool)
        masker.set(mask)
        capture = H3OfficialFeatureCapture(
            blocks, [0, 2], torch.tensor([1, 2])
        )
        capture.clear()
        hidden = torch.randn(2, 6, 4, requires_grad=True)
        for block in blocks:
            hidden = block(hidden)
        features = capture.stacked()
        features.square().mean().backward()

        self.assertEqual(tuple(features.shape), (2, 2, 2, 4))
        self.assertIsNotNone(hidden.grad_fn)
        self.assertTrue(all(block.last_mask is mask for block in blocks))
        self.assertGreater(float(blocks[0].scale.grad.abs()), 0.0)
        capture.close()
        masker.close()


if __name__ == "__main__":
    unittest.main()
