import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "h3wam"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODEL = load_module(
    "fact_lite_consequence_future_h3_test_module",
    ROOT / "src/fastwam/models/h3wam/fact_lite_consequence.py",
)
TRAIN = load_module(
    "train_h3_fact_future_h3_test_module",
    ROOT / "scripts/h3wam/train_h3_fact_future_h3.py",
)


class FactLiteFutureH3Test(unittest.TestCase):
    def test_preprojected_forward_matches_raw_feature_forward(self):
        torch.manual_seed(3)
        inputs = {
            "current_proprio": torch.randn(2, 3),
            "h3_features": torch.randn(2, 5, 6),
            "candidate_actions": torch.randn(2, 4, 2),
        }
        for model in (
            MODEL.FutureH3ConsequenceModel(
                state_dim=3, action_dim=2, action_horizon=4,
                h3_feature_dim=6, target_dim=4, hidden_dim=8,
                feature_input_scale=1.0, projection_seed=9,
            ),
            MODEL.TemporalFutureH3ConsequenceModel(
                state_dim=3, action_dim=2, action_horizon=4,
                actions_per_latent=2, h3_feature_dim=6, target_dim=4,
                hidden_dim=8, num_heads=2, feature_input_scale=1.0,
                projection_seed=9,
            ),
        ):
            projected = model.project_features(inputs["h3_features"])
            expected = model(**inputs)
            actual = model.forward_projected(
                inputs["current_proprio"], projected,
                inputs["candidate_actions"],
            )
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_target_is_absent_from_forward_and_action_generator_is_detached(self):
        torch.manual_seed(7)
        generator = torch.nn.Linear(5, 8)
        model = MODEL.FutureH3ConsequenceModel(
            state_dim=3,
            action_dim=2,
            action_horizon=4,
            h3_feature_dim=6,
            target_dim=4,
            hidden_dim=8,
            feature_input_scale=1.0,
            projection_seed=9,
        )
        actions = generator(torch.randn(3, 5)).reshape(3, 4, 2)
        _, loss = MODEL.future_h3_mse(
            model,
            current_proprio=torch.randn(3, 3),
            h3_features=torch.randn(3, 2, 6),
            candidate_actions=actions,
            future_h3_features=torch.randn(3, 2, 6, requires_grad=True),
        )
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in generator.parameters()))
        self.assertGreater(float(model.action_encoder[0].weight.grad.abs().sum()), 0.0)

    def test_temporal_join_requires_exact_start_plus_horizon(self):
        root = Path("/dataset")
        current = {
            "id": "current",
            "dataset_root": str(root),
            "suite": "libero_goal",
            "episode": 3,
            "start": 4,
            "length": 80,
            "context_id": "task-a",
            "padded_tail": False,
        }
        future = {**current, "id": "future", "start": 36}
        index = TRAIN.build_future_index([current, future])
        self.assertEqual(
            TRAIN.eligible_rows([current], index, action_horizon=32), [current]
        )
        self.assertEqual(
            TRAIN.eligible_rows([current], TRAIN.build_future_index([current]), action_horizon=32),
            [],
        )

    def test_two_step_cpu_main_writes_restore_checked_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            features = cache / "features"
            windows = cache / "windows"
            features.mkdir(parents=True)
            windows.mkdir()
            torch.save(
                {
                    "action_min": -torch.ones(7),
                    "action_max": torch.ones(7),
                    "state_min": -torch.ones(8),
                    "state_max": torch.ones(8),
                },
                cache / "stats.pt",
            )
            currents = []
            source = []
            for episode in range(4):
                split = "train" if episode < 2 else "validation"
                current = {
                    "id": f"libero_goal_ep{episode:06d}_s000000",
                    "dataset_root": str(root / "raw"),
                    "suite": "libero_goal",
                    "episode": episode,
                    "start": 0,
                    "length": 80,
                    "task": "open drawer",
                    "context_id": "task-drawer",
                    "split": split,
                    "padded_tail": False,
                }
                future = {
                    **current,
                    "id": f"libero_goal_ep{episode:06d}_s000032",
                    "start": 32,
                }
                currents.append(current)
                source.extend((current, future))
            checkpoint = root / "h3-int8.safetensors"
            for index, row in enumerate(source):
                torch.save(
                    {
                        "features": torch.full(
                            (1, 32, 5376), float(index + 1), dtype=torch.bfloat16
                        ),
                        "layers": (49,),
                        "context_id": row["context_id"],
                        "action_horizon": 32,
                        "capture_token_count": 32,
                        "capture_token_strategy": TRAIN.CACHE_STRATEGY,
                        "backbone": TRAIN.CACHE_BACKBONE,
                        "quantization": TRAIN.CACHE_QUANTIZATION,
                        "manifest_items": len(source),
                        "timestep": 1.0,
                        "checkpoint": str(checkpoint),
                    },
                    features / f"{row['id']}.pt",
                )
            for row in currents:
                torch.save(
                    {
                        "actions": torch.linspace(-1, 1, 32 * 7).reshape(32, 7),
                        "state": torch.zeros(8),
                        "action_is_pad": torch.zeros(32, dtype=torch.bool),
                    },
                    windows / f"{row['id']}.pt",
                )

            def write_jsonl(path: Path, rows: list[dict]):
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

            train_manifest = root / "train.jsonl"
            val_manifest = root / "val.jsonl"
            source_manifest = root / "source.jsonl"
            write_jsonl(train_manifest, currents[:2])
            write_jsonl(val_manifest, currents[2:])
            write_jsonl(source_manifest, source)
            output = root / "report.json"
            saved = root / "checkpoint.pt"
            argv = [
                "train_h3_fact_future_h3.py",
                "--train-manifest", str(train_manifest),
                "--val-manifest", str(val_manifest),
                "--source-manifest", str(source_manifest),
                "--cache-root", str(cache),
                "--feature-subdir", "features",
                "--steps", "2",
                "--train-limit", "2",
                "--val-limit", "2",
                "--batch-size", "2",
                "--target-dim", "4",
                "--hidden-dim", "8",
                "--feature-input-scale", "0.01",
                "--device", "cpu",
                "--output", str(output),
                "--checkpoint", str(saved),
            ]
            with patch.object(sys, "argv", argv):
                TRAIN.main()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["mechanism"]["fresh_restore_max_abs"], 0.0)
            self.assertEqual(report["data"]["episode_overlap"], 0)
            self.assertEqual(report["data"]["train_selected"], 2)
            self.assertEqual(torch.load(saved, weights_only=False)["completed_steps"], 2)


if __name__ == "__main__":
    unittest.main()
