import unittest

import torch

from scripts.h3dreamwam.verify_h3_lingbot_four_stream_fsdp import (
    video_clean_from_velocity,
)


class H3LingBotTrainingConventionsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
