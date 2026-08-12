import unittest

import torch

from fastwam.models.h3wam import SmallActionFlowTransformer


class SmallActionFlowTransformerTest(unittest.TestCase):
    def test_shape_and_gradients(self):
        model = SmallActionFlowTransformer(
            action_dim=7,
            state_dim=8,
            context_dim=16,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            ffn_dim=64,
        )
        prediction = model(
            torch.randn(2, 32, 7),
            state=torch.randn(2, 8),
            context=torch.randn(2, 12, 16),
            video_sigma=torch.tensor([0.2, 0.8]),
        )
        self.assertEqual(prediction.shape, (2, 32, 7))
        prediction.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_rejects_too_long_horizon(self):
        model = SmallActionFlowTransformer(
            action_dim=7,
            state_dim=8,
            context_dim=16,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
            max_horizon=8,
        )
        with self.assertRaisesRegex(ValueError, "exceeds maximum"):
            model(
                torch.randn(1, 9, 7),
                state=torch.randn(1, 8),
                context=torch.randn(1, 4, 16),
                video_sigma=torch.tensor([0.5]),
            )


if __name__ == "__main__":
    unittest.main()
