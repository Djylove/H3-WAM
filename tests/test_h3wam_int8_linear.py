import json
import unittest
from unittest import mock

import torch

from fastwam.models.h3wam.int8_linear import (
    ConvRotInt8Linear,
    parse_int8_marker,
)


def marker(**overrides) -> torch.Tensor:
    payload = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 256,
        **overrides,
    }
    return torch.tensor(list(json.dumps(payload).encode()), dtype=torch.uint8)


class ConvRotInt8LinearContractTest(unittest.TestCase):
    def test_parses_official_marker(self):
        self.assertEqual(
            parse_int8_marker(marker()),
            {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 256,
            },
        )

    def test_rejects_unknown_format(self):
        with self.assertRaisesRegex(ValueError, "unsupported quantized format"):
            parse_int8_marker(marker(format="int8_w8a8"))

    def test_builds_frozen_checkpoint_layer(self):
        weight = torch.zeros(8, 256, dtype=torch.int8)
        scale = torch.ones(8, 1, dtype=torch.float32)
        layer = ConvRotInt8Linear.from_checkpoint_tensors(
            weight=weight,
            weight_scale=scale,
            marker=marker(),
        )
        self.assertEqual(layer.in_features, 256)
        self.assertEqual(layer.out_features, 8)
        self.assertEqual(dict(layer.named_parameters()), {})
        self.assertEqual(
            set(dict(layer.named_buffers())), {"weight", "weight_scale"}
        )

    def test_rejects_nondivisible_convrot_width(self):
        with self.assertRaisesRegex(ValueError, "divide the group size"):
            ConvRotInt8Linear(
                torch.zeros(8, 255, dtype=torch.int8),
                torch.ones(8, 1, dtype=torch.float32),
            )

    def test_swiglu_accepts_double_width_input_before_fused_projection(self):
        layer = ConvRotInt8Linear(
            torch.zeros(4, 8, dtype=torch.int8),
            torch.ones(4, 1, dtype=torch.float32),
            convrot=False,
        )
        self.assertEqual(layer.in_features, 8)
        # Kernel invocation is covered by the real cloud smoke test.  This
        # contract check makes sure the fused fc2(SwiGLU(fc1(x))) width is
        # accepted before importing the optional CUDA package.
        real_import = __import__

        def import_without_kernel(name, *args, **kwargs):
            if name == "comfy_kitchen":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_without_kernel):
            with self.assertRaisesRegex(RuntimeError, "comfy-kitchen"):
                layer(torch.zeros(2, 16), input_act="swiglu")


if __name__ == "__main__":
    unittest.main()
