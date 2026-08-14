import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/h3wam/precompute_h3_int8_features.py"
SPEC = importlib.util.spec_from_file_location("precompute_h3_int8_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StarWAMFeaturePoolingTest(unittest.TestCase):
    def test_matches_upstream_adaptive_average_pool(self):
        features = torch.arange(2 * 98 * 4, dtype=torch.float32).reshape(2, 98, 4)
        actual = MODULE.pool_feature_tokens(features, 32)
        expected = F.adaptive_avg_pool1d(features.transpose(1, 2), 32).transpose(1, 2)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (2, 32, 4))

    def test_nonpositive_count_preserves_all_tokens(self):
        features = torch.randn(1, 7, 3)
        self.assertIs(MODULE.pool_feature_tokens(features, 0), features)

    def test_historical_comfy_alias_repeats_last_layer_without_aliasing(self):
        features = torch.randn(5, 7, 3)
        output = MODULE.apply_capture_compatibility(features, "comfy_alias_v1")
        for index in range(output.shape[0]):
            torch.testing.assert_close(output[index], features[-1])
        output[0, 0, 0] += 1
        self.assertNotEqual(float(output[0, 0, 0]), float(output[1, 0, 0]))

    def test_default_capture_compatibility_preserves_features(self):
        features = torch.randn(5, 7, 3)
        self.assertIs(MODULE.apply_capture_compatibility(features, "none"), features)

    def test_dreamwam_kv_pooling_preserves_heads_and_clones_every_layer(self):
        captured = {
            layer: {
                "k": torch.randn(1, 11, 2, 4),
                "v": torch.randn(1, 11, 2, 4),
            }
            for layer in (9, 19, 29, 39, 49)
        }
        output = MODULE.prepare_dreamwam_kv_cache(
            captured,
            layers=(9, 19, 29, 39, 49),
            token_count=5,
        )
        pointers = []
        for layer in (9, 19, 29, 39, 49):
            for name in ("k", "v"):
                self.assertEqual(tuple(output[layer][name].shape), (5, 2, 4))
                self.assertEqual(output[layer][name].dtype, torch.bfloat16)
                pointers.append(output[layer][name].untyped_storage().data_ptr())
        self.assertEqual(len(pointers), len(set(pointers)))
        before = output[19]["k"].clone()
        output[9]["k"].add_(1)
        torch.testing.assert_close(output[19]["k"], before)

    def test_dreamwam_kv_capture_is_default_off(self):
        import sys
        from unittest.mock import patch

        required = [
            "precompute",
            "manifest.jsonl",
            "--cache-root",
            "cache",
            "--h3-checkpoint",
            "h3.safetensors",
        ]
        with patch.object(sys, "argv", required):
            args = MODULE.parse_args()
        self.assertFalse(args.dreamwam_kv_carrier)
        self.assertFalse(args.also_starwam_feature_cache)
        self.assertEqual(args.output_subdir, "h3_int8_last32_features")
        self.assertEqual(args.layers, (49,))
        self.assertEqual(
            args.dreamwam_kv_output_subdir, "h3_int8_dreamwam_kv_5x32"
        )

    def test_dual_cache_flag_is_explicit(self):
        import sys
        from unittest.mock import patch

        argv = [
            "precompute",
            "manifest.jsonl",
            "--cache-root",
            "cache",
            "--h3-checkpoint",
            "h3.safetensors",
            "--dreamwam-kv-carrier",
            "--also-starwam-feature-cache",
            "--output-subdir",
            "starwam_dense",
        ]
        with patch.object(sys, "argv", argv):
            args = MODULE.parse_args()
        self.assertTrue(args.dreamwam_kv_carrier)
        self.assertTrue(args.also_starwam_feature_cache)
        self.assertEqual(args.layers, (49,))
        self.assertEqual(args.output_subdir, "starwam_dense")


if __name__ == "__main__":
    unittest.main()
