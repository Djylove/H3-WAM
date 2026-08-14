import unittest

import torch

from fastwam.models.h3wam.starwam_feature_action import H3StarWAMFeatureActionPolicy


class H3StarWAMFeatureActionPolicyTest(unittest.TestCase):
    def build_policy(self):
        return H3StarWAMFeatureActionPolicy(
            action_dim=2,
            proprio_dim=3,
            h3_feature_dim=8,
            context_dim=6,
            hidden_dim=8,
            ffn_dim=16,
            num_heads=2,
            attn_head_dim=4,
            num_layers=2,
            freq_dim=8,
            max_seq_len=8,
            use_gradient_checkpointing=False,
        )

    def test_context_order_and_shape(self):
        policy = self.build_policy()
        context, mask = policy.compose_context(
            torch.randn(2, 5, 6),
            torch.randn(2, 1, 4, 8),
            torch.randn(2, 3),
        )
        self.assertEqual(tuple(context.shape), (2, 10, 6))
        self.assertEqual(tuple(mask.shape), (2, 10))
        self.assertTrue(mask.all())

    def test_action_loss_reaches_expert_and_feature_projector(self):
        torch.manual_seed(7)
        policy = self.build_policy()
        prediction = policy(
            torch.randn(2, 4, 2),
            torch.tensor([100.0, 700.0]),
            text_context=torch.randn(2, 5, 6),
            h3_features=torch.randn(2, 1, 4, 8),
            proprio=torch.randn(2, 3),
        )
        self.assertEqual(tuple(prediction.shape), (2, 4, 2))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.square().mean().backward()
        self.assertIsNotNone(policy.feature_projector.weight.grad)
        self.assertIsNotNone(policy.action_expert.blocks[0].self_attn.q.weight.grad)
        self.assertTrue(torch.isfinite(policy.feature_projector.weight.grad).all())

    def test_rejects_multilayer_cache(self):
        policy = self.build_policy()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            policy.compose_context(
                torch.randn(1, 2, 6), torch.randn(1, 5, 4, 8), torch.randn(1, 3)
            )


if __name__ == "__main__":
    unittest.main()
