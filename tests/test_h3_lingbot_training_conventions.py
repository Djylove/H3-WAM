import unittest
from unittest.mock import patch

import torch

from scripts.h3dreamwam.verify_h3_lingbot_four_stream_fsdp import (
    flow_match_training_weight,
    is_checkpoint_milestone,
    normalize_action,
    prepend_initial_action_history,
    video_clean_from_velocity,
    weighted_video_action_losses,
    shifted_noise_sigma,
)


class H3LingBotTrainingConventionsTest(unittest.TestCase):
    def test_checkpoint_cadence_uses_cumulative_steps(self) -> None:
        saved = [
            step
            for step in range(1, 7501)
            if is_checkpoint_milestone(
                step,
                base_completed_steps=2500,
                checkpoint_every=1000,
                total_steps=7500,
            )
        ]
        self.assertEqual(saved, [500, 1500, 2500, 3500, 4500, 5500, 6500])
        self.assertEqual([2500 + step for step in saved], list(range(3000, 10000, 1000)))

    def test_official_optimizer_overrides_are_explicit(self) -> None:
        from scripts.h3dreamwam.verify_h3_lingbot_four_stream_fsdp import parse_args

        with patch(
            "sys.argv",
            [
                "verify",
                "--model",
                "/tmp/model",
                "--output",
                "/tmp/out.json",
                "--learning-rate",
                "1e-5",
                "--weight-decay",
                "0.1",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.learning_rate, 1.0e-5)
        self.assertEqual(args.weight_decay, 0.1)

    def test_action_train_and_infer_shifts_are_independent(self) -> None:
        from scripts.h3dreamwam.verify_h3_lingbot_four_stream_fsdp import parse_args

        with patch(
            "sys.argv",
            [
                "verify",
                "--model", "/tmp/model",
                "--output", "/tmp/out.json",
                "--action-train-shift", "5.0",
                "--action-infer-shift", "1.0",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.action_train_shift, 5.0)
        self.assertEqual(args.action_infer_shift, 1.0)

    def test_action_shift_controls_high_noise_training_coverage(self) -> None:
        raw = torch.linspace(0.0, 1.0, 10001)[:-1]
        low = shifted_noise_sigma(raw, 0.05)
        neutral = shifted_noise_sigma(raw, 1.0)
        high = shifted_noise_sigma(raw, 5.0)
        self.assertLess(float(low.mean()), float(neutral.mean()))
        self.assertLess(float(neutral.mean()), float(high.mean()))
        self.assertLess(float((low > 0.5).float().mean()), 0.06)
        self.assertGreater(float((high > 0.5).float().mean()), 0.8)

    def test_long_run_checkpoint_contract_is_explicit(self) -> None:
        from scripts.h3dreamwam.verify_h3_lingbot_four_stream_fsdp import parse_args

        with patch(
            "sys.argv",
            [
                "verify",
                "--model", "/tmp/model",
                "--output", "/tmp/out.json",
                "--save-stage", "/tmp/stage.pt",
                "--checkpoint-every", "500",
                "--base-completed-steps", "500",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.checkpoint_every, 500)
        self.assertEqual(args.base_completed_steps, 500)

    def test_initial_action_history_is_zero_before_quantile_normalization(self) -> None:
        action = torch.arange(42, dtype=torch.float32).reshape(6, 7)
        raw = prepend_initial_action_history(action, history_steps=4, horizon=8)
        torch.testing.assert_close(raw[:4], torch.zeros(4, 7))
        torch.testing.assert_close(raw[4:], action[:4])
        stats = {"q01": [-1.0] * 6 + [0.0], "q99": [1.0] * 7}
        normalized = normalize_action(
            raw, mode="quantile", stats={}, quantile_stats=stats
        )
        torch.testing.assert_close(normalized[:4, :6], torch.zeros(4, 6))
        torch.testing.assert_close(normalized[:4, 6], -torch.ones(4))

    def test_clean_reconstruction_matches_clean_time_flow(self) -> None:
        clean = torch.tensor([[[2.0], [-1.0]]])
        noise = torch.tensor([[[6.0], [3.0]]])
        clean_time = torch.tensor([0.75, 0.25])
        sigma = 1.0 - clean_time
        noisy = clean_time[None, :, None] * clean
        noisy += sigma[None, :, None] * noise
        velocity = clean - noise
        reconstructed = video_clean_from_velocity(noisy, clean_time, velocity)
        torch.testing.assert_close(reconstructed, clean)

    def test_flow_weight_matches_released_scheduler_formula(self) -> None:
        raw = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.999])
        shift = 0.05
        sigma = shift * raw / (1.0 + (shift - 1.0) * raw)
        actual = flow_match_training_weight(sigma, shift=shift)
        grid = torch.linspace(1.0, 0.0, 1001, dtype=torch.float64)[:-1]
        shifted = shift * grid / (1.0 + (shift - 1.0) * grid)
        grid_y = torch.exp(-2.0 * (shifted - 0.5).square())
        expected_y = torch.exp(-2.0 * (sigma.double() - 0.5).square())
        expected = (expected_y - grid_y.min()) / (grid_y - grid_y.min()).mean()
        torch.testing.assert_close(actual.double(), expected)

    def test_weighted_losses_apply_frame_and_action_weights(self) -> None:
        video_prediction = torch.tensor([[[0.0], [2.0], [4.0]]])
        video_target = torch.zeros_like(video_prediction)
        future = torch.tensor([False, True, True])
        video_times = torch.tensor([0.9, 0.25, 0.75])
        video_indices = torch.tensor([0, 1, 2])
        action_prediction = torch.tensor([[[1.0], [3.0]]])
        action_target = torch.zeros_like(action_prediction)
        action_time = torch.tensor([0.2, 0.8])
        action_indices = torch.tensor([0, 1])
        video_loss, action_loss = weighted_video_action_losses(
            video_prediction=video_prediction,
            video_target=video_target,
            future=future,
            noisy_video_timesteps=video_times,
            noisy_video_timestep_indices=video_indices,
            action_prediction=action_prediction,
            action_target=action_target,
            action_time=action_time,
            action_timestep_indices=action_indices,
            action_shift=5.0,
        )
        video_weights = flow_match_training_weight(
            1.0 - video_times[1:], shift=12.0
        )
        action_weights = flow_match_training_weight(
            1.0 - action_time, shift=5.0
        )
        torch.testing.assert_close(
            video_loss, (torch.tensor([4.0, 16.0]) * video_weights).mean()
        )
        torch.testing.assert_close(
            action_loss, (torch.tensor([1.0, 9.0]) * action_weights).mean()
        )


if __name__ == "__main__":
    unittest.main()
