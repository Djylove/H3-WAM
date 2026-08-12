from types import SimpleNamespace
import unittest

import torch
from torch import nn

from fastwam.models.h3dreamwam import expand_h3_rgb_flow_projections


class TinyH3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(in_channels=3, patch_size=(1, 2, 2))
        self.proj_in = nn.Linear(12, 8)
        self.proj_out = nn.Linear(8, 12)


class ProjectionExpansionTest(unittest.TestCase):
    def test_projection_expansion_preserves_zero_flow_rgb_path(self) -> None:
        torch.manual_seed(7)
        model = TinyH3()
        rgb_rows = torch.randn(2, 5, 12)
        hidden = torch.randn(2, 5, 8)
        original_input = model.proj_in(rgb_rows)
        original_output = model.proj_out(hidden)

        report = expand_h3_rgb_flow_projections(
            model, generator=torch.Generator().manual_seed(9)
        )
        joint_rows = torch.cat((rgb_rows, torch.zeros_like(rgb_rows)), dim=-1)
        expanded_input = model.proj_in(joint_rows)
        expanded_output = model.proj_out(hidden)

        self.assertEqual(report.old_patch_width, 12)
        self.assertEqual(report.new_patch_width, 24)
        self.assertEqual(report.flow_output_init_scale, 0.1)
        self.assertEqual(model.config.in_channels, 6)
        torch.testing.assert_close(expanded_input, original_input, rtol=0, atol=0)
        torch.testing.assert_close(
            expanded_output[..., :12], original_output, rtol=2e-6, atol=1e-7
        )
        self.assertGreater(torch.count_nonzero(expanded_output[..., 12:]), 0)

    def test_projection_expansion_can_disable_flow_output_initialization(self) -> None:
        model = TinyH3()
        expand_h3_rgb_flow_projections(model, flow_output_init_scale=0.0)
        hidden = torch.randn(2, 5, 8)
        self.assertEqual(torch.count_nonzero(model.proj_out(hidden)[..., 12:]), 0)

    def test_projection_expansion_rejects_wrong_h3_shape(self) -> None:
        model = TinyH3()
        model.proj_in = nn.Linear(13, 8)
        with self.assertRaisesRegex(ValueError, "proj_in width"):
            expand_h3_rgb_flow_projections(model)


if __name__ == "__main__":
    unittest.main()
