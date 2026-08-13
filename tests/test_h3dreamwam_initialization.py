import unittest
import math

import torch

from fastwam.models.h3dreamwam import (
    H3DoTActionHead,
    H3DreamActionExpert,
    initialize_action_expert_from_h3,
    initialize_dot_action_head_from_h3,
    resize_tensor,
)
from tests.test_h3dreamwam_joint_attention import TinyH3Block


class TinyTimeEmbedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(6, 8)
        self.linear_2 = torch.nn.Linear(8, 4)


class TinyH3(torch.nn.Module):
    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        self.context_embedder = torch.nn.Linear(10, 8)
        self.time_embedder = TinyTimeEmbedding()
        blocks = [TinyH3Block() for _ in range(layers)]
        for block in blocks:
            swiglu = torch.nn.Module()
            swiglu.proj = torch.nn.Linear(8, 32, bias=False)
            block.ff = torch.nn.Module()
            block.ff.net = torch.nn.ModuleList(
                [swiglu, torch.nn.Identity(), torch.nn.Linear(16, 8, bias=False)]
            )
        self.transformer_blocks = torch.nn.ModuleList(blocks)


class H3ActionInitializationTest(unittest.TestCase):
    def test_h3_backbone_initializes_action_expert(self) -> None:
        torch.manual_seed(3)
        h3 = TinyH3()
        action = H3DreamActionExpert(
            action_dim=3,
            state_dim=2,
            text_dim=10,
            hidden_dim=6,
            ffn_dim=12,
            num_heads=2,
            head_dim=4,
            num_layers=2,
            frequency_dim=6,
        )
        report = initialize_action_expert_from_h3(action, h3)
        self.assertEqual(report.layers, 2)
        self.assertGreater(report.resized_tensors, 0)
        expected_q = resize_tensor(
            h3.transformer_blocks[0].attn.to_q.weight,
            tuple(action.blocks[0].attn.to_q.weight.shape),
        )
        torch.testing.assert_close(
            action.blocks[0].attn.to_q.weight.float(), expected_q
        )
        torch.testing.assert_close(
            action.blocks[0].cross_attn.to_q.weight,
            action.blocks[0].attn.to_q.weight,
        )
        self.assertEqual(torch.count_nonzero(action.blocks[0].attn.to_q.bias), 0)
        self.assertEqual(
            torch.count_nonzero(action.blocks[0].cross_attn.to_out.weight), 0
        )
        self.assertEqual(torch.count_nonzero(action.time_projection[-1].weight), 0)
        self.assertEqual(
            torch.count_nonzero(action.blocks[0].video_residual_gate),
            0,
        )
        torch.testing.assert_close(
            action.blocks[0].modulation[:, 2],
            torch.ones_like(action.blocks[0].modulation[:, 2]),
        )
        expected_out = resize_tensor(
            h3.transformer_blocks[0].attn.to_out[0].weight,
            tuple(action.blocks[0].attn.to_out.weight.shape),
        ) * 0.01
        torch.testing.assert_close(
            action.blocks[0].attn.to_out.weight.float(), expected_out
        )
        self.assertEqual(report.residual_output_scale, 0.01)

    def test_fastwam_alpha_scaling_is_available_for_h3_interpolation(self) -> None:
        torch.manual_seed(3)
        h3 = TinyH3()
        action = H3DreamActionExpert(
            action_dim=3,
            state_dim=2,
            text_dim=10,
            hidden_dim=6,
            ffn_dim=12,
            num_heads=2,
            head_dim=4,
            num_layers=2,
            frequency_dim=6,
        )
        initialize_action_expert_from_h3(action, h3, alpha_scaling=True)
        source = h3.transformer_blocks[0].attn.to_q.weight
        expected = resize_tensor(
            source,
            tuple(action.blocks[0].attn.to_q.weight.shape),
        ) * math.sqrt(source.shape[-1] / action.blocks[0].attn.to_q.weight.shape[-1])
        torch.testing.assert_close(
            action.blocks[0].attn.to_q.weight.float(), expected
        )

    def test_depth_sampled_h3_initializes_deeper_dot_carrier(self) -> None:
        torch.manual_seed(7)
        h3 = TinyH3(layers=5)
        for index, block in enumerate(h3.transformer_blocks):
            block.attn.to_q.weight.data.fill_(float(index + 1))
        action = H3DoTActionHead(
            action_dim=3,
            hidden_dim=6,
            ffn_dim=12,
            num_heads=2,
            head_dim=4,
            num_layers=3,
            frequency_dim=6,
        )
        original_action_embedding = action.action_embedding.weight.detach().clone()
        original_output = action.output.weight.detach().clone()
        report = initialize_dot_action_head_from_h3(action, h3)
        self.assertEqual(report.source_layer_indices, (0, 2, 4))
        self.assertTrue(report.alpha_scaling)
        self.assertGreater(report.resized_tensors, 0)
        for target_index, source_index in enumerate(report.source_layer_indices):
            source = h3.transformer_blocks[source_index].attn.to_q.weight
            expected = resize_tensor(
                source,
                tuple(action.layers[target_index].attn.to_q.weight.shape),
            ) * math.sqrt(
                source.shape[-1]
                / action.layers[target_index].attn.to_q.weight.shape[-1]
            )
            torch.testing.assert_close(
                action.layers[target_index].attn.to_q.weight.float(), expected
            )
            torch.testing.assert_close(
                action.layers[target_index].modulation[:, 2],
                torch.ones_like(action.layers[target_index].modulation[:, 2]),
            )
        self.assertEqual(
            torch.count_nonzero(action.time_projection[-1].weight), 0
        )
        torch.testing.assert_close(action.action_embedding.weight, original_action_embedding)
        torch.testing.assert_close(action.output.weight, original_output)
        docked_keys = torch.randn(3, 2, 7, 2, 4)
        docked_values = torch.randn_like(docked_keys)
        output = action(
            noisy_actions=torch.randn(2, 5, 3),
            timestep=torch.tensor([100.0, 700.0]),
            docked_keys=docked_keys,
            docked_values=docked_values,
        )
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        for layer in action.layers:
            self.assertTrue(torch.isfinite(layer.attn.to_q.weight.grad).all())
            self.assertGreater(float(layer.attn.to_q.weight.grad.abs().sum()), 0.0)

    def test_single_dot_layer_uses_last_h3_block(self) -> None:
        h3 = TinyH3(layers=5)
        action = H3DoTActionHead(
            action_dim=3,
            hidden_dim=6,
            ffn_dim=12,
            num_heads=2,
            head_dim=4,
            num_layers=1,
            frequency_dim=6,
        )
        report = initialize_dot_action_head_from_h3(action, h3)
        self.assertEqual(report.source_layer_indices, (4,))

    def test_dot_initialization_requires_source_blocks_before_docking(self) -> None:
        h3 = TinyH3(layers=2)
        h3.transformer_blocks = torch.nn.ModuleList()
        action = H3DoTActionHead(
            action_dim=3,
            hidden_dim=6,
            ffn_dim=12,
            num_heads=2,
            head_dim=4,
            num_layers=1,
            frequency_dim=6,
        )
        with self.assertRaisesRegex(ValueError, "layer counts must be positive"):
            initialize_dot_action_head_from_h3(action, h3)


if __name__ == "__main__":
    unittest.main()
