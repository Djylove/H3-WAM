import unittest
import math

import torch

from fastwam.models.h3dreamwam import (
    H3DreamActionExpert,
    initialize_action_expert_from_h3,
    resize_tensor,
)
from tests.test_h3dreamwam_joint_attention import TinyH3Block


class TinyTimeEmbedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(6, 8)
        self.linear_2 = torch.nn.Linear(8, 4)


class TinyH3(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_embedder = torch.nn.Linear(10, 8)
        self.time_embedder = TinyTimeEmbedding()
        blocks = [TinyH3Block() for _ in range(2)]
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


if __name__ == "__main__":
    unittest.main()
