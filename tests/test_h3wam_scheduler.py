import unittest

import torch

from fastwam.models.h3wam import (
    H3ActionAdapter,
    H3ActionBridge,
    H3ActionFlowScheduler,
    h3wam_action_training_step,
    h3wam_joint_training_step,
    prepare_h3wam_flow_batch,
)


class FakeH3(torch.nn.Module):
    def forward(self, x, timestep, context, **kwargs):
        del timestep, context, kwargs
        return [torch.zeros_like(x[0]), x[1] * 0.5]


class H3ActionFlowSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = H3ActionFlowScheduler()

    def test_video_to_action_sigma_matches_shared_base_grid(self):
        base = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
        video = self.scheduler.shift(base, self.scheduler.video_shift)
        expected_action = self.scheduler.shift(base, self.scheduler.action_shift)
        torch.testing.assert_close(self.scheduler.action_sigma(video), expected_action)

    def test_action_slope_matches_autograd(self):
        video = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, requires_grad=True)
        action = self.scheduler.action_sigma(video)
        derivative = torch.autograd.grad(action.sum(), video)[0]
        torch.testing.assert_close(self.scheduler.action_slope(video), derivative)

    def test_noise_and_velocity_use_action_schedule(self):
        actions = torch.tensor([[[1.0, -1.0]], [[2.0, 0.0]]])
        noise = torch.zeros_like(actions)
        video_sigma = torch.tensor([0.0, 1.0])
        noisy = self.scheduler.add_action_noise(actions, noise, video_sigma)
        target = self.scheduler.training_target(actions, noise, video_sigma)
        torch.testing.assert_close(noisy[0], actions[0])
        torch.testing.assert_close(noisy[1], noise[1])
        self.assertEqual(target.shape, actions.shape)

    def test_sampled_sigmas_share_batch_and_range(self):
        video, action = self.scheduler.sample_training_sigmas(32, device="cpu")
        self.assertEqual(video.shape, (32,))
        self.assertEqual(action.shape, (32,))
        self.assertTrue(bool(((video >= 0) & (video <= 1)).all()))
        self.assertTrue(bool(((action >= 0) & (action <= 1)).all()))

    def test_action_inference_deltas_reach_clean_endpoint(self):
        sigmas, video_deltas = self.scheduler.inference_schedule(4, device="cpu")
        action_deltas = torch.stack(
            [
                self.scheduler.action_inference_delta(sigma, delta)
                for sigma, delta in zip(sigmas, video_deltas)
            ]
        )
        self.assertAlmostEqual(float(action_deltas.sum()), -1.0, places=6)

    def test_rejects_bad_sigma_batch(self):
        actions = torch.zeros(2, 4, 7)
        with self.assertRaisesRegex(ValueError, "sigma must be scalar or"):
            self.scheduler.add_action_noise(actions, torch.zeros_like(actions), torch.ones(3))

    def test_prepare_batch_noises_both_modalities(self):
        video = torch.ones(1, 24, 2, 4, 4)
        actions = torch.ones(1, 8, 7)
        batch = prepare_h3wam_flow_batch(
            video_latents=video,
            actions=actions,
            scheduler=self.scheduler,
            video_sigma=torch.tensor([1.0]),
            video_noise=torch.zeros_like(video),
            action_noise=torch.zeros_like(actions),
        )
        torch.testing.assert_close(batch.noisy_video_latents, torch.zeros_like(video))
        torch.testing.assert_close(batch.noisy_actions, torch.zeros_like(actions))
        self.assertEqual(batch.action_target.shape, actions.shape)

    def test_training_step_masks_padding_and_backpropagates(self):
        adapter = H3ActionAdapter(action_dim=7)
        bridge = H3ActionBridge(FakeH3(), adapter)
        actions = torch.randn(1, 8, 7)
        loss, output, flow_batch = h3wam_action_training_step(
            bridge,
            video_latents=torch.randn(1, 24, 2, 4, 4),
            actions=actions,
            context=torch.randn(1, 4, 16),
            scheduler=self.scheduler,
            action_is_pad=torch.tensor([[False] * 6 + [True] * 2]),
            video_sigma=torch.tensor([0.5]),
        )
        loss.backward()
        self.assertEqual(output.action_velocity.shape, actions.shape)
        self.assertEqual(flow_batch.timestep.shape, (1,))
        self.assertTrue(any(parameter.grad is not None for parameter in adapter.parameters()))

    def test_joint_step_reports_action_and_video_losses(self):
        adapter = H3ActionAdapter(action_dim=7)
        bridge = H3ActionBridge(FakeH3(), adapter)
        losses, output, _ = h3wam_joint_training_step(
            bridge,
            video_latents=torch.randn(1, 24, 2, 4, 4),
            actions=torch.randn(1, 8, 7),
            context=torch.randn(1, 4, 16),
            scheduler=self.scheduler,
            video_loss_weight=0.2,
            video_sigma=torch.tensor([0.5]),
        )
        torch.testing.assert_close(losses.total, losses.action + 0.2 * losses.video)
        self.assertEqual(output.video_velocity.shape, (1, 24, 2, 4, 4))
        losses.total.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in adapter.parameters()))


if __name__ == "__main__":
    unittest.main()
