import unittest

import torch

from fastwam.models.h3dreamwam import (
    build_h3dream_inference_schedule,
    h3dream_flow_training_weight,
    sample_h3_lingbot_chunk_causal,
    sample_h3dream_joint_rows,
)


class H3DreamSamplingTest(unittest.TestCase):
    def test_training_weight_matches_dreamwam_normalization(self) -> None:
        progress = torch.linspace(1.0, 0.0, 1001)[:-1]
        sigma = 5.0 * progress / (1.0 + 4.0 * progress)
        weight = h3dream_flow_training_weight(sigma * 1000.0)
        self.assertAlmostEqual(float(weight.mean()), 1.0, places=5)
        self.assertLess(float(weight[0]), 1.0e-6)
        middle = int((sigma - 0.5).abs().argmin())
        self.assertGreater(float(weight[middle]), 1.0)

    def test_shifted_schedule_has_correct_endpoints_and_directions(self) -> None:
        schedule = build_h3dream_inference_schedule(10, device="cpu")
        self.assertEqual(float(schedule.video_clean_times[0]), 0.0)
        self.assertEqual(float(schedule.action_sigmas[0]), 1.0)
        self.assertTrue(torch.all(schedule.video_clean_deltas > 0))
        self.assertTrue(torch.all(schedule.action_sigma_deltas < 0))
        torch.testing.assert_close(schedule.video_clean_deltas.sum(), torch.tensor(1.0))
        torch.testing.assert_close(schedule.action_sigma_deltas.sum(), torch.tensor(-1.0))

    def test_oracle_velocity_recovers_clean_targets_and_freezes_condition(self) -> None:
        torch.manual_seed(4)
        condition = torch.randn(1, 2, 5)
        clean_future = torch.randn(1, 3, 5)
        video_noise = torch.randn_like(clean_future)
        clean_actions = torch.randn(1, 4, 3)
        action_noise = torch.randn_like(clean_actions)
        initial_video = torch.cat((condition, video_noise), dim=1)

        def oracle(video, actions, video_time, action_sigma):
            del video, actions, video_time, action_sigma
            video_velocity = torch.cat(
                (torch.zeros_like(condition), clean_future - video_noise), dim=1
            )
            return video_velocity, action_noise - clean_actions

        sample = sample_h3dream_joint_rows(
            oracle,
            initial_video_rows=initial_video,
            condition_video_rows=condition.shape[1],
            initial_actions=action_noise,
            schedule=build_h3dream_inference_schedule(4, device="cpu"),
        )
        torch.testing.assert_close(sample.video_rows[:, :2], condition)
        torch.testing.assert_close(sample.video_rows[:, 2:], clean_future)
        torch.testing.assert_close(sample.actions, clean_actions)

    def test_lingbot_sampler_commits_only_generated_chunk_history(self) -> None:
        condition = torch.tensor([[[9.0]]])
        clean_video = torch.tensor([[[9.0], [2.0], [3.0]]])
        video_noise = torch.tensor([[[9.0], [12.0], [13.0]]])
        clean_actions = torch.tensor([[[4.0], [5.0]]])
        action_noise = torch.tensor([[[14.0], [15.0]]])
        calls = []

        def oracle(
            video,
            video_history,
            actions,
            action_history,
            video_time,
            action_sigma,
            clean_video_valid,
            clean_action_valid,
        ):
            calls.append(
                (
                    video_history.clone(),
                    action_history.clone(),
                    float(video_time),
                    float(action_sigma),
                    clean_video_valid.clone(),
                    clean_action_valid.clone(),
                )
            )
            return clean_video - video_noise, action_noise - clean_actions

        two_steps = build_h3dream_inference_schedule(
            2, device="cpu", action_shift=0.05
        )
        sample = sample_h3_lingbot_chunk_causal(
            oracle,
            initial_video_rows=torch.cat((condition, video_noise[:, 1:]), dim=1),
            observed_video_mask=torch.tensor([True, False, False]),
            video_chunk_ids=torch.tensor([0, 0, 1]),
            initial_actions=action_noise,
            action_chunk_ids=torch.tensor([0, 1]),
            video_schedule=two_steps,
            action_schedule=two_steps,
        )
        torch.testing.assert_close(sample.video_rows, clean_video)
        torch.testing.assert_close(sample.actions, clean_actions)
        torch.testing.assert_close(sample.clean_video_rows, clean_video)
        torch.testing.assert_close(sample.clean_actions, clean_actions)
        # Before chunk 0 video is generated, no future clean row is exposed.
        torch.testing.assert_close(calls[0][0], torch.tensor([[[9.0], [0.0], [0.0]]]))
        torch.testing.assert_close(calls[0][1], torch.zeros_like(clean_actions))
        # The first action call sees generated chunk-0 video but no clean action.
        first_action_call = next(call for call in calls if call[2] == 1.0)
        torch.testing.assert_close(
            first_action_call[0], torch.tensor([[[9.0], [2.0], [0.0]]])
        )
        torch.testing.assert_close(first_action_call[1], torch.zeros_like(clean_actions))
        # The last chunk may see committed prior action, never the future one.
        self.assertAlmostEqual(float(calls[-1][1][0, 0, 0]), 4.0, places=5)
        self.assertEqual(float(calls[-1][1][0, 1, 0]), 0.0)
        torch.testing.assert_close(calls[0][4], torch.tensor([True, False, False]))
        torch.testing.assert_close(calls[0][5], torch.tensor([False, False]))

    def test_lingbot_sampler_preserves_observed_action_anchor(self) -> None:
        initial_video = torch.tensor([[[7.0], [8.0]]])
        initial_actions = torch.tensor([[[0.0], [9.0]]])

        def oracle(
            video,
            video_history,
            actions,
            action_history,
            video_time,
            action_sigma,
            clean_video_valid,
            clean_action_valid,
        ):
            del video_history, video_time, action_sigma, clean_video_valid
            self.assertTrue(bool(clean_action_valid[0]))
            self.assertEqual(float(action_history[0, 0, 0]), 0.0)
            return torch.zeros_like(video), torch.tensor([[[99.0], [9.0]]])

        schedule = build_h3dream_inference_schedule(1, device="cpu")
        sample = sample_h3_lingbot_chunk_causal(
            oracle,
            initial_video_rows=initial_video,
            observed_video_mask=torch.tensor([True, False]),
            video_chunk_ids=torch.tensor([0, 1]),
            initial_actions=initial_actions,
            action_chunk_ids=torch.tensor([0, 1]),
            observed_action_mask=torch.tensor([True, False]),
            video_schedule=schedule,
            action_schedule=schedule,
        )
        self.assertEqual(float(sample.actions[0, 0, 0]), 0.0)
        self.assertEqual(float(sample.actions[0, 1, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
