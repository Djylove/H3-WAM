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


if __name__ == "__main__":
    unittest.main()
