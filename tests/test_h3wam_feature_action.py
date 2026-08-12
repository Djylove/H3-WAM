import unittest

import torch

from fastwam.models.h3wam import (
    H3BlockFeatureCapture,
    H3FeatureActionTransformer,
    H3FeatureSwitchGate,
    H3MultiLayerActionTransformer,
    H3MixtureActionOutput,
)


class H3FeatureActionTest(unittest.TestCase):
    def test_feature_capture_uses_block_replacement_contract(self):
        capture = H3BlockFeatureCapture([2], token_start=3, token_stop=6)
        replacement = capture.transformer_options()["patches_replace"]["dit"][
            ("double_block", 2)
        ]

        def original(args):
            return {"img": args["img"] + 1.0}

        hidden = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        result = replacement({"img": hidden}, {"original_block": original})

        self.assertTrue(torch.equal(result["img"], hidden + 1.0))
        self.assertTrue(torch.equal(capture.stacked()[0], (hidden + 1.0)[3:6]))

    def test_differentiable_feature_capture_preserves_gradient(self):
        capture = H3BlockFeatureCapture(
            [2], token_start=3, token_stop=6, detach=False
        )
        replacement = capture.transformer_options()["patches_replace"]["dit"][
            ("double_block", 2)
        ]

        def original(args):
            return {"img": args["img"] * 2.0}

        hidden = torch.randn(10, 4, requires_grad=True)
        replacement({"img": hidden}, {"original_block": original})
        capture.stacked().square().mean().backward()

        self.assertIsNotNone(hidden.grad)
        self.assertGreater(float(hidden.grad.abs().sum()), 0.0)

    def test_action_expert_cross_attends_h3_tokens(self):
        model = H3FeatureActionTransformer(
            action_dim=7,
            state_dim=8,
            h3_feature_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
        )
        actions = torch.randn(2, 16, 7)
        state = torch.randn(2, 16, 8)
        features = torch.randn(2, 2, 12, 32, requires_grad=True)

        output = model(
            actions,
            state=state,
            h3_features=features,
            video_sigma=torch.tensor([0.2, 0.8]),
        )
        output.square().mean().backward()

        self.assertEqual(tuple(output.shape), (2, 16, 7))
        self.assertIsNotNone(features.grad)
        self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_mixture_action_expert_returns_modes_and_gate(self):
        model = H3FeatureActionTransformer(
            action_dim=7,
            state_dim=9,
            h3_feature_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            num_action_modes=2,
        )
        output = model(
            torch.zeros(3, 16, 7),
            state=torch.randn(3, 16, 9),
            h3_features=torch.randn(3, 2, 12, 32),
            video_sigma=torch.zeros(3),
        )

        self.assertIsInstance(output, H3MixtureActionOutput)
        self.assertEqual(tuple(output.actions.shape), (3, 2, 16, 7))
        self.assertEqual(tuple(output.mode_logits.shape), (3, 2))

    def test_multilayer_action_head_learns_backbone_depth_mixing(self):
        model = H3MultiLayerActionTransformer(
            action_dim=7,
            state_dim=8,
            num_h3_layers=6,
            h3_feature_dim=32,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            ffn_dim=128,
        )
        features = torch.randn(2, 6, 12, 32, requires_grad=True)
        output = model(
            torch.randn(2, 16, 7),
            state=torch.randn(2, 16, 8),
            h3_features=features,
            action_timestep=torch.tensor([0.2, 0.8]),
        )
        output.square().mean().backward()

        self.assertEqual(tuple(output.shape), (2, 16, 7))
        self.assertIsNotNone(features.grad)
        self.assertGreater(float(features.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.layer_mix_logits.grad)
        self.assertGreater(float(model.layer_mix_logits.grad.abs().sum()), 0.0)

    def test_multilayer_action_head_supports_uniform_depth_initialization(self):
        model = H3MultiLayerActionTransformer(
            action_dim=7,
            state_dim=8,
            num_h3_layers=5,
            h3_feature_dim=32,
            hidden_dim=64,
            num_layers=4,
            num_heads=4,
            ffn_dim=128,
            layer_mix_initialization="uniform",
        )

        expected = torch.full((4, 5), 0.2)
        self.assertTrue(
            torch.allclose(model.layer_mix_logits.softmax(dim=-1), expected)
        )

    def test_multilayer_action_head_explicitly_cross_attends_language_tokens(self):
        model = H3MultiLayerActionTransformer(
            action_dim=7,
            state_dim=8,
            num_h3_layers=3,
            h3_feature_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            language_feature_dim=24,
        )
        language = torch.randn(2, 5, 24, requires_grad=True)
        output = model(
            torch.zeros(2, 8, 7),
            state=torch.randn(2, 8, 8),
            h3_features=torch.randn(2, 3, 12, 32),
            action_timestep=torch.zeros(2),
            language_features=language,
        )
        output.square().mean().backward()

        self.assertEqual(tuple(output.shape), (2, 8, 7))
        self.assertIsNotNone(language.grad)
        self.assertGreater(float(language.grad.abs().sum()), 0.0)

    def test_multilayer_history_gate_is_zero_initialized_and_learns_difference(self):
        model = H3MultiLayerActionTransformer(
            action_dim=7,
            state_dim=8,
            num_h3_layers=3,
            h3_feature_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            history_conditioning=True,
        )
        current = torch.randn(2, 3, 12, 32)
        history = torch.randn(2, 3, 12, 32)
        inputs = {
            "state": torch.randn(2, 8, 8),
            "h3_features": current,
            "action_timestep": torch.zeros(2),
        }

        without_history_effect = model(
            torch.zeros(2, 8, 7),
            history_h3_features=current,
            **inputs,
        )
        different_history_at_zero_gate = model(
            torch.zeros(2, 8, 7),
            history_h3_features=history,
            **inputs,
        )
        self.assertTrue(
            torch.equal(without_history_effect, different_history_at_zero_gate)
        )

        assert model.history_gate is not None
        with torch.no_grad():
            model.history_gate.fill_(0.25)
        output = model(
            torch.zeros(2, 8, 7),
            history_h3_features=history,
            **inputs,
        )
        self.assertFalse(torch.equal(output, without_history_effect))
        output.square().mean().backward()
        self.assertIsNotNone(model.history_gate.grad)
        self.assertGreater(float(model.history_gate.grad.abs().sum()), 0.0)

    def test_zero_initialized_history_adapter_preserves_parent_then_gets_gradient(self):
        model = H3MultiLayerActionTransformer(
            action_dim=7,
            state_dim=8,
            num_h3_layers=3,
            h3_feature_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            history_conditioning=True,
            history_adapter_rank=8,
        )
        current = torch.randn(2, 3, 12, 32)
        history = torch.randn(2, 3, 12, 32)
        arguments = {
            "state": torch.randn(2, 8, 8),
            "h3_features": current,
            "action_timestep": torch.zeros(2),
        }
        parent = model(
            torch.zeros(2, 8, 7), history_h3_features=current, **arguments
        )
        adapted = model(
            torch.zeros(2, 8, 7), history_h3_features=history, **arguments
        )

        self.assertTrue(torch.equal(parent, adapted))
        adapted.square().mean().backward()
        assert model.history_up is not None
        self.assertGreater(
            sum(float(layer.weight.grad.abs().sum()) for layer in model.history_up),
            0.0,
        )

    def test_switch_gate_pools_h3_features_without_phase(self):
        gate = H3FeatureSwitchGate(
            h3_feature_dim=32,
            state_dim=8,
            hidden_dim=16,
        )
        features = torch.randn(3, 5, 12, 32, requires_grad=True)
        logits = gate(features, torch.randn(3, 8))
        logits.square().mean().backward()

        self.assertEqual(tuple(logits.shape), (3,))
        self.assertIsNotNone(features.grad)
        self.assertGreater(float(features.grad.abs().sum()), 0.0)
        pooled_logits = gate(features.detach().mean(dim=(1, 2)), torch.randn(3, 8))
        self.assertEqual(tuple(pooled_logits.shape), (3,))


if __name__ == "__main__":
    unittest.main()
