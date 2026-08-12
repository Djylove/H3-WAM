import unittest

import torch
from torch import nn

from scripts.h3dreamwam.precompute_h3_motion_latents import (
    flow_to_rgb,
    raft_motion_video,
)


class ZeroRAFT(nn.Module):
    def forward(self, image1, image2, *, iters, test_mode):
        del image2, iters, test_mode
        flow = image1.new_zeros(image1.shape[0], 2, *image1.shape[-2:])
        return flow, flow


class H3DreamMotionTest(unittest.TestCase):
    def test_zero_flow_is_white_in_dreamwam_colorwheel(self) -> None:
        image = flow_to_rgb(torch.zeros(2, 7, 11, 2))
        self.assertEqual(tuple(image.shape), (2, 7, 11, 3))
        self.assertTrue(torch.all(image == 255))

    def test_motion_video_duplicates_first_transition(self) -> None:
        video = torch.randint(0, 256, (5, 3, 9, 13), dtype=torch.uint8)
        motion = raft_motion_video(
            ZeroRAFT(),
            video,
            device=torch.device("cpu"),
            iterations=3,
            batch_size=2,
        )
        self.assertEqual(tuple(motion.shape), tuple(video.shape))
        torch.testing.assert_close(motion[0], motion[1], rtol=0, atol=0)
        self.assertTrue(torch.all(motion == 255))


if __name__ == "__main__":
    unittest.main()
