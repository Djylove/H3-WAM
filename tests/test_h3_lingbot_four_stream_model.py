import unittest

import torch
from torch import nn

from fastwam.models.h3dreamwam import H3DreamActionExpert, H3LingBotWAM
from tests.test_h3dreamwam_joint_attention import TinyH3Block


class TinyTimeEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(4, 4)
        self.linear_2 = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear_2(torch.nn.functional.silu(self.linear_1(value)))


class TinyNormOut(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(8)

    def forward(
        self,
        hidden: torch.Tensor,
        _temb: torch.Tensor,
        _indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.norm(hidden)


class TinyRoPE(nn.Module):
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        empty = torch.empty(position_ids.shape[0], 0, device=position_ids.device)
        return empty, empty


class TinyH3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = nn.Linear(6, 8)
        self.proj_out = nn.Linear(8, 6)
        self.context_embedder = nn.Linear(6, 8)
        self.token_refiner = nn.Identity()
        self.time_proj = nn.Linear(1, 4)
        self.time_embedder = TinyTimeEmbedding()
        self.rope = TinyRoPE()
        self.norm_out = TinyNormOut()
        self.transformer_blocks = nn.ModuleList([TinyH3Block()])


class H3LingBotWAMTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.h3 = TinyH3()
        self.action_expert = H3DreamActionExpert(
            action_dim=2,
            state_dim=3,
            text_dim=6,
            hidden_dim=8,
            ffn_dim=16,
            num_heads=2,
            head_dim=4,
            num_layers=1,
            frequency_dim=4,
        )
        self.model = H3LingBotWAM(
            self.h3,
            self.action_expert,
            use_gradient_checkpointing=False,
        )
        self.arguments = {
            "noisy_video_rows": torch.randn(1, 4, 6),
            "clean_video_rows": torch.randn(1, 4, 6),
            "video_position_ids": torch.zeros(4, 3),
            "video_chunk_ids": torch.tensor([0, 0, 1, 1]),
            "noisy_video_timestep": torch.tensor([0.5]),
            "clean_video_timestep": torch.tensor([0.0]),
            "noisy_actions": torch.randn(1, 4, 2),
            "clean_actions": torch.randn(1, 4, 2),
            "action_chunk_ids": torch.tensor([0, 0, 1, 1]),
            "noisy_action_timestep": torch.tensor([500.0]),
            "context": torch.randn(1, 2, 6),
            "context_position_ids": torch.zeros(3, 3),
            "state": torch.randn(1, 3),
            "context_mask": torch.ones(1, 2, dtype=torch.bool),
        }

    def test_full_model_shapes_and_block_ownership(self) -> None:
        output = self.model(**self.arguments)
        self.assertEqual(output.video_velocity_rows.shape, (1, 4, 6))
        self.assertEqual(output.action_velocity.shape, (1, 4, 2))
        self.assertEqual(len(self.model.paired_layers), 1)
        self.assertEqual(len(self.model.h3.transformer_blocks), 0)
        self.assertEqual(len(self.model.action_expert.blocks), 0)

    def test_both_velocity_objectives_cross_the_expert_boundary(self) -> None:
        output = self.model(**self.arguments)
        output.video_velocity_rows[:, 2:].square().mean().backward(retain_graph=True)
        action_grad = self.model.paired_layers[0].action_block.attn.to_v.weight.grad
        self.assertIsNotNone(action_grad)
        self.assertGreater(float(action_grad.abs().sum()), 0.0)

        self.model.zero_grad(set_to_none=True)
        output.action_velocity.square().mean().backward()
        h3_grad = self.model.paired_layers[0].h3_block.attn.to_v.weight.grad
        self.assertIsNotNone(h3_grad)
        self.assertGreater(float(h3_grad.abs().sum()), 0.0)

    def test_context_layout_must_reserve_proprio_row(self) -> None:
        arguments = dict(self.arguments)
        arguments["context_position_ids"] = torch.zeros(2, 3)
        with self.assertRaisesRegex(ValueError, "reserve one proprio"):
            self.model(**arguments)


if __name__ == "__main__":
    unittest.main()
