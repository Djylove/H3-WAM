import unittest

import torch
from torch import nn

from fastwam.models.h3wam import (
    H3ActionAdapter,
    H3ActionBridge,
    H3ActionFlowScheduler,
    sample_h3wam_actions,
)


class ConstantVelocityH3(nn.Module):
    def __init__(self):
        super().__init__()
        self.scheduler = H3ActionFlowScheduler()

    def forward(self, x, timestep, context, **kwargs):
        del context, kwargs
        video_sigma = timestep.float() / self.scheduler.timestep_scale
        slope = self.scheduler.action_slope(video_sigma).reshape(-1, 1, 1, 1)
        return [torch.ones_like(x[0]), torch.ones_like(x[1]) * slope]


class IdentityVelocityAdapter(H3ActionAdapter):
    def __init__(self):
        nn.Module.__init__(self)
        self.action_dim = 1
        self.state_dim = 0
        self.latent_dim = 1
        self.num_streams = 1

    def encode_actions(self, actions, state=None):
        del state
        return actions.permute(0, 2, 1).unsqueeze(2)

    def decode_velocity(self, latent_velocity, state=None, context=None):
        del state, context
        return latent_velocity.squeeze(2).permute(0, 2, 1)


class H3InferenceTest(unittest.TestCase):
    def test_euler_schedule_reaches_clean_endpoint(self):
        bridge = H3ActionBridge(ConstantVelocityH3(), IdentityVelocityAdapter())
        sample = sample_h3wam_actions(
            bridge,
            context=torch.zeros(1, 2, 4),
            state=None,
            scheduler=H3ActionFlowScheduler(),
            action_shape=(1, 3, 1),
            video_shape=(1, 1, 1, 1, 1),
            model_evaluations=4,
            initial_action_noise=torch.ones(1, 3, 1),
            initial_video_noise=torch.ones(1, 1, 1, 1, 1),
        )
        torch.testing.assert_close(sample.actions, torch.zeros_like(sample.actions), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            sample.video_latents.float(),
            torch.zeros_like(sample.video_latents.float()),
            atol=5e-3,
            rtol=0,
        )

    def test_schedule_has_requested_number_of_evaluations(self):
        sigmas, deltas = H3ActionFlowScheduler().inference_schedule(2, device="cpu")
        self.assertEqual(sigmas.shape, (2,))
        self.assertEqual(deltas.shape, (2,))
        self.assertAlmostEqual(float(deltas.sum()), -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
