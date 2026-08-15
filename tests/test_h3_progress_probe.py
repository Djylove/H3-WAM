import unittest

import torch

from scripts.h3wam import evaluate_h3_progress_probe as probe
from scripts.h3wam import precompute_h3_progress_probe_features as precompute


class H3ProgressProbeTest(unittest.TestCase):
    def test_compact_layer49(self):
        tensor = torch.arange(32 * 56 * 128, dtype=torch.float32).reshape(32, 56, 128)
        feature = precompute.compact_layer49({"video_kv_cache": {49: {"k": tensor, "v": tensor + 1}}})
        self.assertEqual(tuple(feature.shape), (512,))
        self.assertTrue(torch.isfinite(feature.float()).all())

    def test_ridge_recovers_linear_signal(self):
        x = torch.arange(20, dtype=torch.double).unsqueeze(1)
        y = x[:, 0] / 20
        prediction = probe.ridge_predict(x, y, x, 1e-6)
        self.assertLess(float((prediction - y).abs().mean()), 1e-5)

    def test_fitted_state_restores_prediction(self):
        x = torch.arange(20, dtype=torch.double).unsqueeze(1)
        y = x[:, 0] / 20
        prediction, state = probe.ridge_fit_predict(x, y, x, 1e-6)
        restored_x = (x - state["mean"]) / state["std"]
        restored_x = torch.cat((torch.ones((20, 1), dtype=torch.double), restored_x), dim=1)
        restored = (restored_x @ state["weights"]).clamp(0, 1)
        self.assertTrue(torch.equal(prediction, restored))


if __name__ == "__main__":
    unittest.main()
