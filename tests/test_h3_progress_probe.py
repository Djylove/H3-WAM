import unittest

import torch

from scripts.h3wam import evaluate_h3_progress_probe as probe
from scripts.h3wam import evaluate_c17_progress_shadow as shadow
from fastwam.models.h3wam import (
    FrozenH3ProgressProbe,
    PROGRESS_DESIGN_CONTRACT,
    PROGRESS_FEATURE_CONTRACT,
    TIMEBLIND_PROGRESS_DESIGN_CONTRACT,
    compact_h3_kv_progress_feature,
)


class H3ProgressProbeTest(unittest.TestCase):
    def test_compact_layer49(self):
        tensor = torch.arange(32 * 56 * 128, dtype=torch.float32).reshape(32, 56, 128)
        feature = compact_h3_kv_progress_feature({"k": tensor, "v": tensor + 1})
        self.assertEqual(tuple(feature.shape), (512,))
        self.assertTrue(torch.isfinite(feature.float()).all())

    def test_validated_probe_predicts_without_mutating_live_cache(self):
        contexts = [f"context-{index:02d}" for index in range(40)]
        payload = {
            "format": "h3wam-frozen-h3-progress-ridge-v1",
            "validation_status": "PASS_PROGRESS_FEATURE_GATE",
            "design_contract": PROGRESS_DESIGN_CONTRACT,
            "feature_contract": PROGRESS_FEATURE_CONTRACT,
            "contexts": contexts,
            "mean": torch.zeros(553, dtype=torch.float64),
            "std": torch.ones(553, dtype=torch.float64),
            "weights": torch.zeros(554, dtype=torch.float64),
        }
        payload["weights"][0] = 0.25
        model = FrozenH3ProgressProbe(payload)
        layer_cache = {
            "k": torch.randn(1, 32, 56, 128, dtype=torch.bfloat16),
            "v": torch.randn(1, 32, 56, 128, dtype=torch.bfloat16),
        }
        original = {name: value.clone() for name, value in layer_cache.items()}
        prediction = model.predict(
            context_id=contexts[3], absolute_step=128, layer_cache=layer_cache
        )
        self.assertEqual(prediction, 0.25)
        for name in original:
            torch.testing.assert_close(layer_cache[name], original[name], atol=0, rtol=0)

    def test_probe_rejects_unvalidated_payload(self):
        with self.assertRaisesRegex(ValueError, "validation gate"):
            FrozenH3ProgressProbe(
                {
                    "format": "h3wam-frozen-h3-progress-ridge-v1",
                    "validation_status": "FAIL_PROGRESS_FEATURE_GATE",
                }
            )

    def test_timeblind_probe_ignores_absolute_step(self):
        contexts = [f"context-{index:02d}" for index in range(40)]
        payload = {
            "format": "h3wam-frozen-h3-timeblind-progress-ridge-v1",
            "validation_status": "PASS_TIMEBLIND_PROGRESS_FEATURE_GATE",
            "design_contract": TIMEBLIND_PROGRESS_DESIGN_CONTRACT,
            "feature_contract": PROGRESS_FEATURE_CONTRACT,
            "contexts": contexts,
            "mean": torch.zeros(552, dtype=torch.float64),
            "std": torch.ones(552, dtype=torch.float64),
            "weights": torch.zeros(553, dtype=torch.float64),
        }
        payload["weights"][0] = 0.4
        model = FrozenH3ProgressProbe(payload)
        layer_cache = {
            "k": torch.zeros(32, 56, 128),
            "v": torch.zeros(32, 56, 128),
        }
        first = model.predict(
            context_id=contexts[0], absolute_step=0, layer_cache=layer_cache
        )
        last = model.predict(
            context_id=contexts[0], absolute_step=400, layer_cache=layer_cache
        )
        self.assertEqual(first, last)

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

    def test_shadow_auc_treats_lower_remaining_progress_as_success(self):
        self.assertEqual(
            shadow.pairwise_auc(
                [True, True, False, False], [-0.1, -0.2, -0.8, -0.9]
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
