import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.h3dreamwam.serve_h3dreamwam_fsdp import _task_contexts
from fastwam.models.h3dreamwam import H3DreamActionExpert


class H3DreamWAMContextRoutingTest(unittest.TestCase):
    def test_proprio_token_can_be_shared_without_double_append(self):
        expert = H3DreamActionExpert(
            action_dim=2,
            state_dim=3,
            text_dim=6,
            hidden_dim=8,
            ffn_dim=16,
            num_heads=2,
            head_dim=4,
            num_layers=1,
            frequency_dim=4,
        )
        context = torch.randn(1, 2, 6)
        mask = torch.ones(1, 2, dtype=torch.bool)
        state = torch.randn(1, 3)
        shared_context, shared_mask = expert.append_state_to_context(
            context=context,
            context_mask=mask,
            state=state,
        )
        prepared = expert.prepare(
            noisy_actions=torch.randn(1, 4, 2),
            timestep=torch.tensor([500.0]),
            context=shared_context,
            context_mask=shared_mask,
            state=state,
            append_state=False,
        )
        self.assertEqual(shared_context.shape, (1, 3, 6))
        self.assertEqual(prepared["context_mask"].shape, (1, 3))

    def test_shared_text_context_id_is_preferred(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            rows = [
                {
                    "id": "window_0",
                    "context_id": "task_context",
                    "task": "pick the bowl",
                },
                {
                    "id": "window_1",
                    "context_id": "task_context",
                    "task": "pick the bowl",
                },
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertEqual(
                _task_contexts(manifest),
                {"pick the bowl": "task_context"},
            )

    def test_legacy_window_context_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"id": "window_0", "task": "pick the bowl"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _task_contexts(manifest),
                {"pick the bowl": "window_0"},
            )


if __name__ == "__main__":
    unittest.main()
