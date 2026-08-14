import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/h3wam/evaluate_h3_int8_starwam_action.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_h3_int8_starwam_action", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MetricTest(unittest.TestCase):
    def test_domain_metrics_respect_padding_and_endpoint(self):
        accumulator = MODULE.DomainMetricAccumulator(action_dim=2)
        prediction = torch.zeros(1, 3, 2)
        target = torch.ones(1, 3, 2)
        is_pad = torch.tensor([[False, False, True]])
        accumulator.update(prediction, target, is_pad)
        metrics = accumulator.finalize()
        self.assertEqual(metrics["valid_steps"], 2)
        self.assertEqual(metrics["valid_elements"], 4)
        self.assertEqual(metrics["action_mse"], 1.0)
        self.assertEqual(metrics["action_mae"], 1.0)
        self.assertAlmostEqual(metrics["chunk_ade_l2"], math.sqrt(2.0))
        self.assertAlmostEqual(metrics["chunk_endpoint_l2"], math.sqrt(2.0))
        self.assertEqual(metrics["prediction_std"], 0.0)
        self.assertEqual(metrics["prediction_outside_unit_count"], 0)
        self.assertEqual(metrics["prediction_outside_unit_fraction"], 0.0)
        self.assertEqual(metrics["prediction_max_abs"], 0.0)

    def test_domain_metrics_report_generated_action_range_before_clamp(self):
        accumulator = MODULE.DomainMetricAccumulator(action_dim=2)
        prediction = torch.tensor([[[2.0, 0.5], [-3.0, 1.0], [99.0, 99.0]]])
        target = torch.zeros_like(prediction)
        is_pad = torch.tensor([[False, False, True]])
        accumulator.update(prediction, target, is_pad)
        metrics = accumulator.finalize()
        self.assertEqual(metrics["valid_elements"], 4)
        self.assertEqual(metrics["prediction_outside_unit_count"], 2)
        self.assertEqual(metrics["prediction_outside_unit_fraction"], 0.5)
        self.assertEqual(metrics["prediction_max_abs"], 3.0)

    def test_gripper_sign_accuracy_and_f1(self):
        accumulator = MODULE.GripperSignAccumulator(gripper_dim=1)
        prediction = torch.tensor([[[0.0, 1.0], [0.0, -1.0], [0.0, 1.0]]])
        target = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, -1.0]]])
        accumulator.update(
            prediction, target, torch.tensor([[False, False, False]])
        )
        metrics = accumulator.finalize()
        self.assertAlmostEqual(metrics["accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_shift5_ten_step_schedule_matches_upstream_formula(self):
        scheduler = MODULE.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        timesteps, deltas = scheduler.build_inference_schedule(
            10, torch.device("cpu"), torch.float32
        )
        u = torch.linspace(1.0, 0.0, 11)
        sigma = 5.0 * u / (1.0 + 4.0 * u)
        torch.testing.assert_close(timesteps, sigma[:-1] * 1000.0)
        torch.testing.assert_close(deltas, sigma[1:] - sigma[:-1])
        self.assertAlmostEqual(float(deltas.sum()), -1.0, places=6)


class BalancedValidationSelectionTest(unittest.TestCase):
    @staticmethod
    def _rows(samples_per_task: int = 3) -> list[dict]:
        return [
            {
                "id": f"task_{task_index:02d}_sample_{sample_index:02d}",
                "task": f"task_{task_index:02d}",
            }
            for task_index in range(MODULE.EXPECTED_BALANCED_VAL_TASKS)
            for sample_index in range(samples_per_task)
        ]

    def test_selection_is_order_independent_and_covers_all_tasks(self):
        rows = self._rows()
        selected, evidence = MODULE.select_validation_rows(
            rows, samples_per_task=2
        )
        reversed_selected, reversed_evidence = MODULE.select_validation_rows(
            list(reversed(rows)), samples_per_task=2
        )
        selected_ids = [row["id"] for row in selected]
        self.assertEqual(selected_ids, [row["id"] for row in reversed_selected])
        self.assertEqual(evidence, reversed_evidence)
        self.assertEqual(evidence["salt"], MODULE.BALANCED_VAL_SELECTION_SALT)
        self.assertEqual(evidence["selected_task_count"], 40)
        self.assertEqual(evidence["selected_items"], 80)
        self.assertEqual(set(evidence["task_counts"].values()), {2})
        self.assertEqual(
            evidence["selected_ids_sha256"], MODULE.sha256_strings(selected_ids)
        )

    def test_rejects_incomplete_task_coverage(self):
        rows = [
            row for row in self._rows() if row["task"] != "task_39"
        ]
        with self.assertRaisesRegex(ValueError, "requires exactly 40 tasks"):
            MODULE.select_validation_rows(rows, samples_per_task=1)

    def test_rejects_task_with_fewer_than_requested_samples(self):
        rows = [
            row
            for row in self._rows(samples_per_task=2)
            if not (
                row["task"] == "task_00"
                and row["id"].endswith("sample_01")
            )
        ]
        with self.assertRaisesRegex(ValueError, "fewer than 2 samples"):
            MODULE.select_validation_rows(rows, samples_per_task=2)

    def test_balanced_selection_is_mutually_exclusive_with_slice_controls(self):
        rows = self._rows()
        for kwargs in ({"limit": 1}, {"sample_offset": 1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                    MODULE.select_validation_rows(
                        rows, samples_per_task=1, **kwargs
                    )


class VisualFeatureShuffleContractTest(unittest.TestCase):
    @staticmethod
    def _rows() -> list[dict]:
        return [
            {"id": f"sample_{index:03d}", "task": f"task_{index // 2:02d}"}
            for index in range(MODULE.VISUAL_FEATURE_SHUFFLE_ITEMS)
        ]

    def test_fixed_salt_right_shift_is_order_independent_and_has_no_self_map(self):
        rows = self._rows()
        mapping, evidence = MODULE.build_visual_feature_shuffle(rows)
        reversed_mapping, reversed_evidence = MODULE.build_visual_feature_shuffle(
            list(reversed(rows))
        )
        self.assertEqual(mapping, reversed_mapping)
        self.assertEqual(evidence["salt"], MODULE.VISUAL_FEATURE_SHUFFLE_SALT)
        self.assertEqual(evidence["self_map_count"], 0)
        self.assertEqual(set(mapping), set(mapping.values()))
        self.assertTrue(all(target != source for target, source in mapping.items()))
        ranked = sorted(
            mapping,
            key=lambda sample_id: MODULE._salted_sample_rank(
                sample_id, MODULE.VISUAL_FEATURE_SHUFFLE_SALT
            ),
        )
        self.assertEqual(mapping[ranked[0]], ranked[-1])
        for index in range(1, len(ranked)):
            self.assertEqual(mapping[ranked[index]], ranked[index - 1])
        self.assertEqual(
            evidence["mapping_sha256"],
            MODULE.sha256_strings(
                f"{sample_id}\0{mapping[sample_id]}"
                for sample_id in sorted(mapping)
            ),
        )
        self.assertEqual(
            evidence["mapping_sha256"], reversed_evidence["mapping_sha256"]
        )

    def test_requires_exactly_frozen_eighty_unique_ids(self):
        with self.assertRaisesRegex(ValueError, "exactly 80"):
            MODULE.build_visual_feature_shuffle(self._rows()[:-1])
        duplicated = self._rows()
        duplicated[-1] = dict(duplicated[0])
        with self.assertRaisesRegex(ValueError, "unique sample ids"):
            MODULE.build_visual_feature_shuffle(duplicated)

    def test_replacement_visual_changes_only_features_with_same_noise(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[dict[str, torch.Tensor]] = []

            def forward(
                self,
                actions: torch.Tensor,
                timestep: torch.Tensor,
                *,
                text_context: torch.Tensor,
                h3_features: torch.Tensor,
                proprio: torch.Tensor,
                text_mask: torch.Tensor,
            ) -> torch.Tensor:
                self.calls.append(
                    {
                        "actions": actions.clone(),
                        "timestep": timestep.clone(),
                        "text_context": text_context.clone(),
                        "h3_features": h3_features.clone(),
                        "proprio": proprio.clone(),
                        "text_mask": text_mask.clone(),
                    }
                )
                return torch.zeros_like(actions)

        batch = {
            "features": torch.zeros(2, 1, 32, 4),
            "shuffled_features": torch.ones(2, 1, 32, 4),
            "text_context": torch.randn(2, 3, 6),
            "text_mask": torch.ones(2, 3, dtype=torch.bool),
            "proprio": torch.randn(2, 2),
        }
        initial_noise = torch.randn(2, 4, 3)
        scheduler = MODULE.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        baseline_model = RecordingModel()
        shuffled_model = RecordingModel()
        baseline = MODULE.sample_action_flow(
            baseline_model,
            batch,
            scheduler,
            inference_steps=2,
            initial_noise=initial_noise,
        )
        shuffled = MODULE.sample_action_flow(
            shuffled_model,
            batch,
            scheduler,
            inference_steps=2,
            initial_noise=initial_noise,
            replacement_visual=True,
        )
        torch.testing.assert_close(baseline, initial_noise)
        torch.testing.assert_close(shuffled, initial_noise)
        self.assertEqual(len(baseline_model.calls), len(shuffled_model.calls))
        for baseline_call, shuffled_call in zip(
            baseline_model.calls, shuffled_model.calls, strict=True
        ):
            for key in (
                "actions",
                "timestep",
                "text_context",
                "proprio",
                "text_mask",
            ):
                torch.testing.assert_close(baseline_call[key], shuffled_call[key])
            torch.testing.assert_close(
                baseline_call["h3_features"], batch["features"]
            )
            torch.testing.assert_close(
                shuffled_call["h3_features"], batch["shuffled_features"]
            )


class SyntheticEvaluatorFixture:
    def __init__(self, root: Path, *, balanced_visual: bool = False) -> None:
        self.root = root
        self.cache_root = root / "cache"
        self.feature_subdir = "last32"
        (self.cache_root / self.feature_subdir).mkdir(parents=True)
        (self.cache_root / "windows").mkdir()
        (self.cache_root / "contexts").mkdir()
        self.source_manifest = root / "source.jsonl"
        self.train_manifest = root / "train.jsonl"
        self.val_manifest = root / "val.jsonl"
        self.checkpoint = root / "checkpoint.pt"
        self.output = root / "report.json"
        self.model_spec = {
            "action_dim": 3,
            "proprio_dim": 2,
            "h3_feature_dim": 12,
            "context_dim": 6,
            "hidden_dim": 8,
            "ffn_dim": 16,
            "num_heads": 2,
            "attn_head_dim": 4,
            "action_layers": 1,
            "freq_dim": 8,
            "max_seq_len": 8,
            "gradient_checkpointing": False,
            "include_feature_timestep": False,
            "feature_timestep": 0.0,
            "feature_input_scale": 1.0,
        }
        self.action_horizon = 4
        if balanced_visual:
            self.train_rows = [
                self._row(
                    f"train_{task_index:02d}",
                    task_index,
                    f"task_{task_index:02d}",
                    f"context_{task_index:02d}",
                )
                for task_index in range(MODULE.EXPECTED_BALANCED_VAL_TASKS)
            ]
            self.val_rows = [
                self._row(
                    f"val_{task_index:02d}_{sample_index}",
                    100 + task_index * 2 + sample_index,
                    f"task_{task_index:02d}",
                    f"context_{task_index:02d}",
                )
                for task_index in range(MODULE.EXPECTED_BALANCED_VAL_TASKS)
                for sample_index in range(2)
            ]
        else:
            self.train_rows = [
                self._row("train_a", 1, "task_a", "context_a"),
                self._row("train_b", 2, "task_b", "context_b"),
            ]
            self.val_rows = [
                self._row("val_a", 3, "task_a", "context_a"),
                self._row("val_b", 4, "task_b", "context_b"),
            ]
        self.rows = [*self.train_rows, *self.val_rows]
        self._write_jsonl(self.source_manifest, self.rows)
        self._write_jsonl(self.train_manifest, self.train_rows)
        self._write_jsonl(self.val_manifest, self.val_rows)
        torch.save(
            {
                "action_min": -torch.ones(3),
                "action_max": torch.ones(3),
                "state_min": -torch.ones(2),
                "state_max": torch.ones(2),
            },
            self.cache_root / "stats.pt",
        )
        for index, row in enumerate(self.rows):
            self._write_cache(row, index)
        self.contract = self._contract()
        torch.manual_seed(17)
        model = MODULE.build_model_from_spec(
            self.model_spec, device=torch.device("cpu"), dtype=torch.float32
        )
        torch.save(
            {
                "schema_version": 2,
                "completed_steps": 12,
                "model": model.state_dict(),
                "optimizer": {},
                "lr_scheduler": {},
                "contract": self.contract,
                "probe_prediction": torch.zeros(1),
                "probe_sample_ids": ["train_a"],
            },
            self.checkpoint,
        )

    @staticmethod
    def _row(
        sample_id: str, episode: int, task: str, context_id: str
    ) -> dict:
        return {
            "id": sample_id,
            "dataset_root": "/synthetic",
            "suite": "libero_synthetic",
            "episode": episode,
            "start": 0,
            "task": task,
            "context_id": context_id,
        }

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _write_cache(self, row: dict, index: int) -> None:
        sample_id = row["id"]
        context_id = row["context_id"]
        torch.save(
            {
                "features": torch.linspace(
                    -1.0 + index * 0.1, 1.0 + index * 0.1, 32 * 12
                ).reshape(1, 32, 12),
                "layers": (49,),
                "context_id": context_id,
                "timestep": 1.0,
                "action_horizon": self.action_horizon,
                "capture_token_count": 32,
                "capture_token_strategy": MODULE.FEATURE_STRATEGY,
                "backbone": MODULE.FEATURE_BACKBONE,
                "quantization": MODULE.FEATURE_QUANTIZATION,
                "manifest_items": len(self.rows),
            },
            self.cache_root / self.feature_subdir / f"{sample_id}.pt",
        )
        action = torch.tensor(
            [
                [-0.5, 0.1, -1.0],
                [-0.2, 0.2, -1.0],
                [0.2, -0.1, 1.0],
                [0.5, -0.2, 1.0],
            ],
            dtype=torch.float32,
        )
        torch.save(
            {
                "actions": action + torch.tensor([index * 0.01, 0.0, 0.0]),
                "state": torch.tensor([index * 0.1, -index * 0.1]),
                "action_is_pad": torch.tensor([False, False, False, index % 2 == 0]),
            },
            self.cache_root / "windows" / f"{sample_id}.pt",
        )
        context_path = self.cache_root / "contexts" / f"{context_id}.pt"
        if not context_path.exists():
            sign = 1.0 if context_id == "context_a" else -1.0
            torch.save(
                {
                    "context": torch.full((1, 3, 6), sign),
                    "token_tags": torch.ones(3, dtype=torch.long),
                    "text_only": True,
                },
                context_path,
            )

    def _contract(self) -> dict:
        return {
            "starwam_commit": MODULE.STARWAM_COMMIT,
            "starwam_action_dit_sha256": MODULE.STARWAM_ACTION_DIT_SHA256,
            "starwam_wan_block_sha256": MODULE.STARWAM_WAN_BLOCK_SHA256,
            "h3_checkpoint_sha256": "synthetic",
            "feature_subdir": self.feature_subdir,
            "feature_strategy": MODULE.FEATURE_STRATEGY,
            "feature_layers": [49],
            "feature_tokens": 32,
            "feature_timestep": 1.0,
            "feature_timestep_token": {
                "value": 0.0,
                "semantics": "starwam_clean_observation_flow_timestep",
                "h3_curve_timestep": 1.0,
                "enabled": False,
                "embedding_trainable": False,
            },
            "feature_input_scale": 1.0,
            "feature_scale_audit": {},
            "source_manifest_sha256": MODULE.sha256_file(self.source_manifest),
            "source_manifest_items": len(self.rows),
            "split_manifest_sha256": MODULE.sha256_file(self.train_manifest),
            "split_manifest_items": len(self.train_rows),
            "stats_sha256": MODULE.sha256_file(self.cache_root / "stats.pt"),
            "action_normalization": "starwam_minmax_clip5",
            "state_normalization": "starwam_minmax_clip5",
            "action_horizon": self.action_horizon,
            "action_shift": 5.0,
            "lr_schedule": {},
            "model_spec": self.model_spec,
        }

    def config(self, **overrides) -> MODULE.EvalConfig:
        values = {
            "checkpoint": self.checkpoint,
            "source_manifest": self.source_manifest,
            "train_manifest": self.train_manifest,
            "val_manifest": self.val_manifest,
            "cache_root": self.cache_root,
            "output": self.output,
            "feature_subdir": self.feature_subdir,
            "batch_size": 2,
            "device": "cpu",
            "seed": 7,
            "language_sensitivity": True,
        }
        values.update(overrides)
        return MODULE.EvalConfig(**values)


class EvaluatorIntegrationTest(unittest.TestCase):
    def test_strict_restore_episode_disjoint_eval_and_evidence_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticEvaluatorFixture(Path(directory))
            report = MODULE.run_evaluation(fixture.config())
            self.assertEqual(report["checkpoint"]["schema_version"], 2)
            self.assertEqual(report["checkpoint"]["completed_steps"], 12)
            self.assertEqual(
                report["checkpoint"]["fresh_restore"]["max_abs"], 0.0
            )
            self.assertEqual(report["data"]["selected_validation_items"], 2)
            self.assertEqual(
                report["data"]["selection"]["selected_ids_sha256"],
                report["data"]["selected_sample_ids_sha256"],
            )
            self.assertIsNone(report["data"]["selection"]["salt"])
            self.assertEqual(report["data"]["split_audit"]["episode_overlap"], 0)
            self.assertEqual(
                report["data"]["source_manifest_sha256"],
                MODULE.sha256_file(fixture.source_manifest),
            )
            normalized = report["metrics"]["normalized_clip5_model_domain"]
            self.assertTrue(math.isfinite(normalized["action_mse"]))
            self.assertTrue(math.isfinite(normalized["action_mae"]))
            self.assertEqual(normalized["valid_steps"], 7)
            self.assertIn("chunk_ade_l2", normalized)
            self.assertIn("chunk_endpoint_l2", normalized)
            self.assertIn("f1", report["metrics"]["gripper_sign"])
            sensitivity = report["metrics"]["language_replacement_sensitivity"]
            self.assertIsNotNone(sensitivity)
            self.assertTrue(sensitivity["same_noise"])
            self.assertTrue(fixture.output.is_file())
            self.assertEqual(json.loads(fixture.output.read_text()), report)

    def test_visual_shuffle_runs_paired_same_noise_and_reports_metric_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticEvaluatorFixture(
                Path(directory), balanced_visual=True
            )
            report = MODULE.run_evaluation(
                fixture.config(
                    samples_per_task=2,
                    batch_size=80,
                    language_sensitivity=False,
                    visual_feature_shuffle=True,
                )
            )
            visual = report["metrics"]["visual_feature_shuffle"]
            self.assertIsNotNone(visual)
            contract = visual["contract"]
            self.assertEqual(contract["salt"], MODULE.VISUAL_FEATURE_SHUFFLE_SALT)
            self.assertEqual(contract["selected_items"], 80)
            self.assertEqual(contract["self_map_count"], 0)
            self.assertEqual(
                visual["evaluated_mapping_sha256"],
                contract["ordered_mapping_sha256"],
            )
            self.assertTrue(visual["same_initial_action_noise"])
            self.assertEqual(
                contract["fixed_conditioning"]["text"], "unchanged"
            )
            self.assertEqual(
                contract["fixed_conditioning"]["proprio"], "unchanged"
            )
            mapping = contract["mapping"]
            self.assertEqual(len(mapping), 80)
            self.assertTrue(
                all(
                    item["target_sample_id"]
                    != item["feature_source_sample_id"]
                    for item in mapping
                )
            )
            normalized_delta = visual["baseline_vs_shuffle_action_delta"][
                "normalized_model_domain"
            ]
            self.assertEqual(normalized_delta["valid_steps"], 280)
            self.assertGreater(normalized_delta["action_mse"], 0.0)
            self.assertTrue(math.isfinite(normalized_delta["chunk_ade_l2"]))
            metric_change = visual["metric_change_shuffle_minus_baseline"][
                "normalized_clip5_model_domain"
            ]
            self.assertIn("action_mse", metric_change)
            self.assertIn("action_mse_per_dim", metric_change)
            self.assertEqual(len(metric_change["action_mse_per_dim"]), 3)
            self.assertTrue(report["inference"]["visual_feature_shuffle"])

    def test_visual_shuffle_rejects_non_balanced_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticEvaluatorFixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "samples-per-task 2"):
                MODULE.run_evaluation(
                    fixture.config(
                        language_sensitivity=False,
                        visual_feature_shuffle=True,
                    )
                )

    def test_rejects_non_schema2_checkpoint_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticEvaluatorFixture(Path(directory))
            payload = torch.load(fixture.checkpoint, weights_only=True)
            payload["schema_version"] = 1
            torch.save(payload, fixture.checkpoint)
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                MODULE.run_evaluation(fixture.config(language_sensitivity=False))

    def test_strict_model_restore_rejects_missing_parameter(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticEvaluatorFixture(Path(directory))
            payload = torch.load(fixture.checkpoint, weights_only=True)
            model_state = dict(payload["model"])
            model_state.pop("action_expert.head.weight")
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                MODULE.restore_model_strict(
                    fixture.model_spec,
                    model_state,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )

    def test_rejects_episode_overlap_even_without_window_overlap(self):
        source = [
            {
                "id": "a",
                "dataset_root": "/data",
                "suite": "suite",
                "episode": 1,
                "task": "task",
            },
            {
                "id": "b",
                "dataset_root": "/data",
                "suite": "suite",
                "episode": 1,
                "task": "task",
            },
        ]
        with self.assertRaisesRegex(ValueError, "episode overlap"):
            MODULE.validate_episode_disjoint_manifests(
                source, [source[0]], [source[1]]
            )


if __name__ == "__main__":
    unittest.main()
