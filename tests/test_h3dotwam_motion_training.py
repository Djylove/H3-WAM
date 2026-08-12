import unittest
from pathlib import Path

import torch

from scripts.h3dreamwam.train_h3dotwam_fsdp import (
    _is_joint_h3_parameter,
    normalize_world_latents,
    resolve_action_stage,
    resolve_h3_io_hyperparameters,
)


class H3DoTWAMMotionTrainingTests(unittest.TestCase):
    def test_motion_latents_are_normalized_per_sample(self) -> None:
        latents = torch.randn(2, 4, 3, 5, 7) * 3.0 + 11.0
        normalized = normalize_world_latents(latents)
        dimensions = tuple(range(1, normalized.ndim))
        torch.testing.assert_close(
            normalized.mean(dim=dimensions),
            torch.zeros(2),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            normalized.std(dim=dimensions, unbiased=False),
            torch.ones(2),
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_joint_checkpoint_includes_motion_io_and_h3_blocks(self) -> None:
        self.assertTrue(
            _is_joint_h3_parameter("_fsdp_wrapped_module.h3.proj_in.weight")
        )
        self.assertTrue(
            _is_joint_h3_parameter(
                "_fsdp_wrapped_module.hub_layers.49.h3_block.attn.to_q.weight"
            )
        )
        self.assertFalse(
            _is_joint_h3_parameter("_fsdp_wrapped_module.action_head.output.weight")
        )

    def test_motion_defaults_match_dreamwam_new_channel_recipe(self) -> None:
        learning_rate, init_scale = resolve_h3_io_hyperparameters(
            motion_enabled=True,
            h3_learning_rate=1.0e-6,
            h3_io_learning_rate=None,
            flow_channel_init_scale=None,
        )
        self.assertEqual(learning_rate, 1.0e-4)
        self.assertEqual(init_scale, 0.1)

    def test_rgb_defaults_preserve_the_pretrained_path(self) -> None:
        learning_rate, init_scale = resolve_h3_io_hyperparameters(
            motion_enabled=False,
            h3_learning_rate=2.0e-6,
            h3_io_learning_rate=None,
            flow_channel_init_scale=None,
        )
        self.assertEqual(learning_rate, 2.0e-6)
        self.assertEqual(init_scale, 0.0)

    def test_joint_stage_supplies_default_action_weights(self) -> None:
        resolved = resolve_action_stage(
            load_stage=None,
            load_joint_stage=Path("/tmp/joint"),
        )
        self.assertEqual(resolved, Path("/tmp/joint/action_stage.pt"))

    def test_explicit_action_stage_can_pair_with_joint_h3(self) -> None:
        resolved = resolve_action_stage(
            load_stage=Path("/tmp/action-only.pt"),
            load_joint_stage=Path("/tmp/joint"),
        )
        self.assertEqual(resolved, Path("/tmp/action-only.pt"))


if __name__ == "__main__":
    unittest.main()
