import unittest

import torch

from scripts.h3dreamwam.train_h3dotwam_fsdp import (
    action_language_ranking_loss,
    select_negative_context,
)


class H3DoTWAMLanguageRankingTest(unittest.TestCase):
    def test_negative_context_has_different_task_and_context(self) -> None:
        rows = [
            {"id": "a0", "context_id": "task_a", "task": "open drawer"},
            {"id": "a1", "context_id": "task_a", "task": "open drawer"},
            {"id": "b0", "context_id": "task_b", "task": "turn on stove"},
        ]
        self.assertEqual(select_negative_context(rows, 0), ("task_b", "turn on stove"))

    def test_ranking_pushes_only_wrong_context_away_from_target(self) -> None:
        correct = torch.tensor(0.20, requires_grad=True)
        wrong = torch.tensor(0.21, requires_grad=True)
        loss = action_language_ranking_loss(correct, wrong, margin=0.05)
        self.assertAlmostEqual(float(loss), 0.04, places=6)
        loss.backward()
        self.assertIsNone(correct.grad)
        self.assertEqual(float(wrong.grad), -1.0)

    def test_satisfied_margin_has_zero_loss(self) -> None:
        correct = torch.tensor(0.20)
        wrong = torch.tensor(0.30, requires_grad=True)
        loss = action_language_ranking_loss(correct, wrong, margin=0.05)
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
