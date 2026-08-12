import unittest

import torch

from fastwam.models.h3dreamwam import H3LingBotSharedWAM
from tests.test_h3_lingbot_four_stream_model import (
    TinyH3,
)


class H3LingBotSharedWAMTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(37)
        self.model = H3LingBotSharedWAM(
            TinyH3(),
            action_dim=2,
            state_dim=3,
            text_dim=6,
            use_gradient_checkpointing=False,
        )
        self.arguments = {
            "noisy_video_rows": torch.randn(1, 4, 6),
            "clean_video_rows": torch.randn(1, 4, 6),
            "video_position_ids": torch.zeros(4, 3),
            "video_chunk_ids": torch.tensor([0, 0, 1, 1]),
            "noisy_video_timestep": torch.tensor([0.5]),
            "clean_video_timestep": torch.tensor([1.0]),
            "noisy_actions": torch.randn(1, 4, 2),
            "clean_actions": torch.randn(1, 4, 2),
            "action_position_ids": torch.tensor(
                [[0.0, -1.0, -1.0], [0.25, -1.0, -1.0],
                 [1.0, -1.0, -1.0], [1.25, -1.0, -1.0]]
            ),
            "action_chunk_ids": torch.tensor([0, 0, 1, 1]),
            "noisy_action_timestep": torch.tensor([0.5]),
            "clean_action_timestep": torch.tensor([1.0]),
            "context": torch.randn(1, 2, 6),
            "context_position_ids": torch.zeros(3, 3),
            "state": torch.randn(1, 3),
            "context_mask": torch.ones(1, 2, dtype=torch.bool),
        }

    def test_shapes_and_single_backbone_ownership(self) -> None:
        output = self.model(**self.arguments)
        self.assertEqual(output.video_velocity_rows.shape, (1, 4, 6))
        self.assertEqual(output.action_velocity.shape, (1, 4, 2))
        self.assertEqual(len(self.model.shared_layers), 1)
        self.assertEqual(len(self.model.h3.transformer_blocks), 0)
        self.assertFalse(
            any("action_block" in name for name, _ in self.model.named_modules())
        )

    def test_action_objective_updates_shared_h3_block(self) -> None:
        output = self.model(**self.arguments)
        output.action_velocity.square().mean().backward()
        block = self.model.shared_layers[0].h3_block
        self.assertIsNotNone(block.attn.to_v.weight.grad)
        self.assertGreater(float(block.attn.to_v.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(self.model.action_adapters.action_embedding.weight.grad)

    def test_video_objective_updates_action_input_through_shared_attention(self) -> None:
        output = self.model(**self.arguments)
        output.video_velocity_rows[:, 2:].square().mean().backward()
        gradient = self.model.action_adapters.action_embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_action_uses_separate_time_embedding_copy(self) -> None:
        self.assertIsNot(
            self.model.h3.time_embedder,
            self.model.action_adapters.time_embedder,
        )
        video_parameter = next(self.model.h3.time_embedder.parameters())
        action_parameter = next(self.model.action_adapters.time_embedder.parameters())
        torch.testing.assert_close(video_parameter, action_parameter)
        self.assertIsNot(video_parameter, action_parameter)

    def test_action_accepts_per_chunk_timestep_rows(self) -> None:
        arguments = dict(self.arguments)
        arguments["noisy_action_timestep"] = torch.tensor([0.25, 0.75])
        arguments["noisy_action_timestep_indices"] = torch.tensor([0, 0, 1, 1])
        output = self.model(**arguments)
        self.assertEqual(output.action_velocity.shape, (1, 4, 2))

    def test_action_timestep_indices_are_checked(self) -> None:
        arguments = dict(self.arguments)
        arguments["noisy_action_timestep_indices"] = torch.tensor([0, 0, 0])
        with self.assertRaisesRegex(ValueError, "action timestep indices"):
            self.model(**arguments)

    def test_action_position_layout_is_checked(self) -> None:
        arguments = dict(self.arguments)
        arguments["action_position_ids"] = torch.zeros(3, 3)
        with self.assertRaisesRegex(ValueError, "action position ids"):
            self.model(**arguments)


if __name__ == "__main__":
    unittest.main()
