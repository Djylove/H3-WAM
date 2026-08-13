import unittest

import torch

from scripts.h3dreamwam.verify_h3_lingbot_four_stream_fsdp import (
    flow_match_training_weight,
    normalize_action,
    prepend_initial_action_history,
    video_clean_from_velocity,
    weighted_video_action_losses,
)


class H3LingBotTrainingConventionsTest(unittest.TestCase):
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
        )
        video_weights = flow_match_training_weight(
            1.0 - video_times[1:], shift=12.0
        )
        action_weights = flow_match_training_weight(
            1.0 - action_time, shift=0.05
        )
        torch.testing.assert_close(
            video_loss, (torch.tensor([4.0, 16.0]) * video_weights).mean()
        )
        torch.testing.assert_close(
            action_loss, (torch.tensor([1.0, 9.0]) * action_weights).mean()
        )


if __name__ == "__main__":
    unittest.main()
