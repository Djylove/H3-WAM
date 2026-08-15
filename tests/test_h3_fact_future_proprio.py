import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODEL = load_module(
    "fact_lite_consequence_test_module",
    ROOT / "src/fastwam/models/h3wam/fact_lite_consequence.py",
)
TRAIN = load_module(
    "train_h3_fact_future_proprio_test_module",
    ROOT / "scripts/h3wam/train_h3_fact_future_proprio.py",
)


class FakeStateReader:
    def __init__(self, values: dict[int, torch.Tensor]) -> None:
        self.values = values

    def states(self, root: Path, episode: int) -> torch.Tensor:
        del root
        return self.values[episode]


class FactLiteFutureProprioTest(unittest.TestCase):
    def build_cache(self, root: Path, rows: list[dict]) -> None:
        (root / "features").mkdir()
        (root / "windows").mkdir()
        torch.save(
            {
                "action_min": -torch.ones(7),
                "action_max": torch.ones(7),
                "state_min": -torch.ones(8),
                "state_max": torch.ones(8),
            },
            root / "stats.pt",
        )
        for row in rows:
            sample_id = row["id"]
            torch.save(
                {
                    "features": torch.ones(1, 32, 5376, dtype=torch.bfloat16),
                    "layers": (49,),
                    "context_id": row["context_id"],
                    "action_horizon": 32,
                    "capture_token_count": 32,
                    "capture_token_strategy": TRAIN.CACHE_STRATEGY,
                    "backbone": TRAIN.CACHE_BACKBONE,
                    "quantization": TRAIN.CACHE_QUANTIZATION,
                    "manifest_items": len(rows),
                    "timestep": 1.0,
                    "checkpoint": str(root / "h3-int8.safetensors"),
                },
                root / "features" / f"{sample_id}.pt",
            )
            current = torch.full((8,), float(row["episode"]) / 10.0)
            torch.save(
                {
                    "actions": torch.linspace(-1, 1, 32 * 7).reshape(32, 7),
                    "state": current,
                    "action_is_pad": torch.zeros(32, dtype=torch.bool),
                },
                root / "windows" / f"{sample_id}.pt",
            )

    @staticmethod
    def rows(dataset_root: Path) -> list[dict]:
        return [
            {
                "id": f"libero_goal_ep{episode:06d}_s000000",
                "dataset_root": str(dataset_root),
                "suite": "libero_goal",
                "episode": episode,
                "start": 0,
                "length": 40,
                "task": "open drawer",
                "context_id": "task_drawer",
                "split": "train" if episode < 2 else "validation",
                "padded_tail": False,
            }
            for episode in range(4)
        ]

    def test_future_state_is_raw_start_plus_horizon_and_current_matches_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.rows(root)
            self.build_cache(root, rows)
            raw = {}
            for row in rows:
                states = torch.zeros(40, 8)
                states[0] = float(row["episode"]) / 10.0
                states[32] = 0.5
                raw[int(row["episode"])] = states
            dataset = TRAIN.FutureProprioDataset(
                rows[:1],
                cache_root=root,
                feature_subdir="features",
                source_manifest_items=len(rows),
                action_horizon=32,
                state_reader=FakeStateReader(raw),
            )
            item = dataset[0]
            torch.testing.assert_close(
                item["future_proprio"], torch.full((8,), 0.5)
            )
            self.assertEqual(item["current_parity_max_abs"], 0.0)

            raw[0][0, 0] = 0.25
            with self.assertRaisesRegex(ValueError, "cache/parquet current-state mismatch"):
                dataset[0]

    def test_episode_split_and_source_provenance_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.rows(Path(directory))
            audit = TRAIN.audit_manifests(rows[:2], rows[2:], rows)
            self.assertEqual(audit["episode_overlap"], 0)

            leaked = dict(rows[2])
            leaked["episode"] = rows[0]["episode"]
            with self.assertRaisesRegex(ValueError, "source provenance"):
                TRAIN.audit_manifests(rows[:2], [leaked], rows)

            duplicate_episode = dict(rows[2])
            duplicate_episode["id"] = "different-id"
            duplicate_episode["episode"] = rows[0]["episode"]
            source = [*rows, duplicate_episode]
            with self.assertRaisesRegex(ValueError, "episode leakage"):
                TRAIN.audit_manifests(rows[:2], [duplicate_episode], source)

    def test_consequence_loss_cannot_update_upstream_action_generator(self):
        torch.manual_seed(4)
        generator = torch.nn.Linear(5, 4 * 2)
        model = MODEL.FutureProprioConsequenceModel(
            state_dim=3,
            action_dim=2,
            action_horizon=4,
            h3_feature_dim=6,
            hidden_dim=8,
            feature_input_scale=1.0,
        )
        actions = generator(torch.randn(3, 5)).reshape(3, 4, 2)
        _, loss = MODEL.future_proprio_mse(
            model,
            current_proprio=torch.randn(3, 3),
            h3_features=torch.randn(3, 2, 6),
            candidate_actions=actions,
            future_proprio=torch.randn(3, 3),
        )
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in generator.parameters()))
        self.assertIsNotNone(model.action_encoder[0].weight.grad)
        self.assertGreater(float(model.action_encoder[0].weight.grad.abs().sum()), 0.0)

    def test_shuffled_control_has_no_self_map_and_independent_is_zero(self):
        actions = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float()
        shuffled = MODEL.actions_for_arm(actions, "shuffled")
        for index in range(4):
            self.assertFalse(torch.equal(actions[index], shuffled[index]))
        self.assertTrue(
            torch.equal(MODEL.actions_for_arm(actions, "independent"), torch.zeros_like(actions))
        )
        with self.assertRaisesRegex(ValueError, "batch_size >= 2"):
            MODEL.actions_for_arm(actions[:1], "shuffled")

    def test_lerobot_parquet_path_uses_metadata_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text(
                json.dumps(
                    {
                        "chunks_size": 100,
                        "data_path": (
                            "data/chunk-{chunk_index:03d}/"
                            "episode_{file_index:06d}.parquet"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            (root / "meta/episodes.jsonl").write_text(
                json.dumps(
                    {
                        "episode_index": 123,
                        "data/chunk_index": 7,
                        "data/file_index": 9,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            expected = root / "data/chunk-007/episode_000009.parquet"
            expected.parent.mkdir(parents=True)
            expected.touch()
            reader = TRAIN.LeRobotParquetStateReader()
            self.assertEqual(reader.parquet_path(root, 123), expected)

    def test_two_step_cpu_main_writes_restore_checked_report(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.rows(root)
            # Add four more episodes so each split can use batch2 x two batches.
            rows.extend(
                {
                    **row,
                    "id": row["id"].replace(
                        f"ep{row['episode']:06d}", f"ep{row['episode'] + 4:06d}"
                    ),
                    "episode": row["episode"] + 4,
                }
                for row in list(rows)
            )
            for index, row in enumerate(rows):
                row["split"] = "train" if index < 4 else "validation"
            self.build_cache(root, rows)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text(
                json.dumps(
                    {
                        "chunks_size": 1000,
                        "data_path": (
                            "data/chunk-{episode_chunk:03d}/"
                            "episode_{episode_index:06d}.parquet"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            (root / "meta/episodes.jsonl").write_text(
                "".join(
                    json.dumps({"episode_index": row["episode"], "length": 40})
                    + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            parquet_dir = root / "data/chunk-000"
            parquet_dir.mkdir(parents=True)
            for row in rows:
                states = []
                for step in range(40):
                    value = float(row["episode"]) / 10.0 if step == 0 else 0.5
                    states.append(torch.full((8,), value).numpy())
                pd.DataFrame({"observation.state": states}).to_parquet(
                    parquet_dir / f"episode_{row['episode']:06d}.parquet"
                )
            train_manifest = root / "train.jsonl"
            val_manifest = root / "val.jsonl"
            source_manifest = root / "source.jsonl"
            train_manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows[:4]),
                encoding="utf-8",
            )
            val_manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows[4:]),
                encoding="utf-8",
            )
            source_manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "report.json"
            checkpoint = root / "checkpoint.pt"
            argv = [
                "trainer",
                "--train-manifest",
                str(train_manifest),
                "--val-manifest",
                str(val_manifest),
                "--source-manifest",
                str(source_manifest),
                "--cache-root",
                str(root),
                "--feature-subdir",
                "features",
                "--output",
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--steps",
                "2",
                "--train-limit",
                "4",
                "--val-limit",
                "4",
                "--batch-size",
                "2",
                "--hidden-dim",
                "8",
                "--device",
                "cpu",
            ]
            with patch.object(sys, "argv", argv):
                TRAIN.main()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn(
                report["status"], {"PASS_MECHANISM_GATE", "FAIL_MECHANISM_GATE"}
            )
            self.assertEqual(report["mechanism"]["fresh_restore_max_abs"], 0.0)
            self.assertEqual(report["data"]["episode_overlap"], 0)
            self.assertTrue(checkpoint.is_file())


if __name__ == "__main__":
    unittest.main()
