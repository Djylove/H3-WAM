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
        self.assertEqual(tuple(context.shape), (2, 11, 6))
        self.assertEqual(tuple(mask.shape), (2, 11))
        self.assertTrue(mask.all())

    def test_clean_feature_timestep_and_input_scale_are_explicit(self):
        torch.manual_seed(3)
        raw = self.build_policy()
        torch.manual_seed(3)
        scaled = H3StarWAMFeatureActionPolicy(
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
            feature_timestep=0.0,
            feature_input_scale=0.5,
        )
        scaled.load_state_dict(raw.state_dict(), strict=False)
        scaled.feature_input_scale.fill_(0.5)
        text = torch.randn(1, 2, 6)
        features = torch.randn(1, 1, 4, 8)
        proprio = torch.randn(1, 3)
        raw_context, _ = raw.compose_context(text, features, proprio)
        scaled_context, _ = scaled.compose_context(text, features, proprio)
        # [text(2), proprio(1), clean-feature-timestep(1), features(4)]
        torch.testing.assert_close(raw_context[:, :4], scaled_context[:, :4])
        expected_scaled = scaled.feature_projector(features[:, 0] * 0.5)
        torch.testing.assert_close(scaled_context[:, 4:], expected_scaled)
        self.assertEqual(raw.feature_timestep, 0.0)

    def test_disabled_feature_timestep_embedding_is_frozen(self):
        policy = H3StarWAMFeatureActionPolicy(
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
            include_feature_timestep=False,
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in policy.feature_timestep_embedding.parameters()
            )
        )
        context, _ = policy.compose_context(
            torch.randn(1, 2, 6), torch.randn(1, 1, 4, 8), torch.randn(1, 3)
        )
        self.assertEqual(tuple(context.shape), (1, 7, 6))

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
