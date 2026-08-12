import unittest

import torch

from fastwam.models.h3wam import (
    align_h3_frame_count,
    h3_latent_is_pad,
    h3_video_latent_frames,
    plan_h3_window,
    resample_video_nearest,
)


class H3TemporalAlignmentTest(unittest.TestCase):
    def test_libero_32_step_window_maps_to_39_h3_frames(self):
        plan = plan_h3_window(action_horizon=32, source_fps=20)
        self.assertEqual(plan.source_frame_count, 33)
        self.assertEqual(plan.h3_frame_count, 39)
        self.assertEqual(plan.h3_latent_frames, 12)
        self.assertAlmostEqual(plan.action_duration_seconds, 1.6)

    def test_h3_frame_grids(self):
        self.assertEqual(align_h3_frame_count(1), 5)
        self.assertEqual(align_h3_frame_count(22), 22)
        self.assertEqual(align_h3_frame_count(23), 39)
        self.assertEqual(h3_video_latent_frames(5), 2)
        self.assertEqual(h3_video_latent_frames(39), 12)

    def test_resampling_preserves_endpoints(self):
        video = torch.arange(33)
        resampled = resample_video_nearest(video, 39)
        self.assertEqual(resampled.shape, (39,))
        self.assertEqual(resampled[0].item(), 0)
        self.assertEqual(resampled[-1].item(), 32)

    def test_padding_mask_matches_released_h3_chunk_geometry(self):
        no_tail = h3_latent_is_pad(torch.zeros(39, dtype=torch.bool))
        self.assertEqual(tuple(no_tail.shape), (12,))
        self.assertFalse(no_tail.any())

        tail = h3_latent_is_pad(torch.arange(39) >= 18)
        self.assertEqual(
            tail.tolist(),
            [False, False, False, False, False, False, True, True, True, True, True, True],
        )


if __name__ == "__main__":
    unittest.main()
