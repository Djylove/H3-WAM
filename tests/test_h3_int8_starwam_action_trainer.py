import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/h3wam/train_h3_int8_starwam_action.py"
)
SPEC = importlib.util.spec_from_file_location("train_h3_int8_starwam_action", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CachedLast32DatasetTest(unittest.TestCase):
    def build_cache(self, root: Path) -> tuple[Path, str]:
        sample_id = "libero_spatial_ep000001_s000000"
        context_id = "task_test"
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "id": sample_id,
                    "context_id": context_id,
                    "episode": 1,
                    "start": 0,
                    "suite": "libero_spatial",
                }
            )
            + "\n"
        )
        (root / "features").mkdir()
        (root / "windows").mkdir()
        (root / "contexts").mkdir()
        torch.save(
            {
                "features": torch.ones(1, 32, 5376, dtype=torch.bfloat16),
                "layers": (49,),
                "episode": 1,
                "start": 0,
                "suite": "libero_spatial",
                "context_id": context_id,
                "timestep": 1.0,
                "action_horizon": 32,
                "capture_token_count": 32,
                "capture_token_strategy": MODULE.CACHE_STRATEGY,
                "backbone": MODULE.CACHE_BACKBONE,
                "quantization": MODULE.CACHE_QUANTIZATION,
                "checkpoint": "/tmp/h3-int8.safetensors",
                "manifest_items": 1,
            },
            root / "features" / f"{sample_id}.pt",
        )
        torch.save(
            {
                "actions": torch.linspace(-1, 1, 32 * 7).reshape(32, 7),
                "state": torch.linspace(-1, 1, 8),
                "action_is_pad": torch.zeros(32, dtype=torch.bool),
            },
            root / "windows" / f"{sample_id}.pt",
        )
        torch.save(
            {
                "context": torch.randn(1, 3, 6),
                "token_tags": torch.ones(3, dtype=torch.long),
                "text_only": True,
            },
            root / "contexts" / f"{context_id}.pt",
        )
        torch.save(
            {
                "action_min": -torch.ones(7),
                "action_max": torch.ones(7),
                "state_min": -torch.ones(8),
                "state_max": torch.ones(8),
            },
            root / "stats.pt",
        )
        return manifest, sample_id

    def test_dataset_checks_contract_and_normalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, sample_id = self.build_cache(root)
            dataset = MODULE.CachedLast32Dataset(
                manifest, root, "features", action_horizon=32
            )
            item = dataset[0]
            self.assertEqual(item["sample_id"], sample_id)
            self.assertEqual(tuple(item["features"].shape), (1, 32, 5376))
            self.assertEqual(tuple(item["actions"].shape), (32, 7))
            torch.testing.assert_close(item["proprio"], torch.linspace(-1, 1, 8))

    def test_collate_pads_variable_text_context(self):
        features = torch.randn(1, 32, 5376)
        base = {
            "features": features,
            "actions": torch.randn(32, 7),
            "proprio": torch.randn(8),
            "action_is_pad": torch.zeros(32, dtype=torch.bool),
        }
        batch = MODULE.collate_cached_batch(
            [
                {**base, "sample_id": "a", "text_context": torch.randn(2, 6)},
                {**base, "sample_id": "b", "text_context": torch.randn(4, 6)},
            ]
        )
        self.assertEqual(tuple(batch["text_context"].shape), (2, 4, 6))
        self.assertEqual(batch["text_mask"].sum(dim=1).tolist(), [2, 4])

    def test_split_manifest_preserves_full_source_cache_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_manifest, sample_id = self.build_cache(root)
            split_row = json.loads(split_manifest.read_text())
            second_row = dict(split_row)
            second_row["id"] = "libero_spatial_ep000002_s000000"
            second_row["episode"] = 2
            source_manifest = root / "source.jsonl"
            source_manifest.write_text(
                json.dumps(split_row) + "\n" + json.dumps(second_row) + "\n"
            )
            feature_path = root / "features" / f"{sample_id}.pt"
            payload = torch.load(feature_path, weights_only=False)
            payload["manifest_items"] = 2
            torch.save(payload, feature_path)
            dataset = MODULE.CachedLast32Dataset(
                split_manifest,
                root,
                "features",
                source_manifest=source_manifest,
            )
            self.assertEqual(dataset.manifest_items, 1)
            self.assertEqual(dataset.source_manifest_items, 2)
            self.assertEqual(dataset[0]["sample_id"], sample_id)

            corrupt = dict(split_row)
            corrupt["task"] = "not source provenance"
            split_manifest.write_text(json.dumps(corrupt) + "\n")
            with self.assertRaisesRegex(ValueError, "not byte-equivalent"):
                MODULE.CachedLast32Dataset(
                    split_manifest,
                    root,
                    "features",
                    source_manifest=source_manifest,
                )


class FlowRegressionComplementTest(unittest.TestCase):
    @staticmethod
    def checkpoint_args(weight: float) -> Namespace:
        return Namespace(
            expected_h3_checkpoint_sha256="h3",
            feature_subdir="features",
            action_horizon=32,
            action_shift=5.0,
            learning_rate=1.0e-4,
            min_learning_rate=1.0e-6,
            warmup_steps=1000,
            scheduler_horizon=21700,
            clean_action_regression_weight=weight,
        )

    @staticmethod
    def checkpoint_dataset() -> Namespace:
        return Namespace(
            source_manifest_sha256="source",
            source_manifest_items=2,
            manifest_sha256="split",
            manifest_items=1,
            stats_sha256="stats",
        )

    def test_cli_is_default_off_and_weight_is_explicit(self):
        required = [
            "trainer",
            "manifest.jsonl",
            "--cache-root",
            "cache",
            "--output",
            "report.json",
        ]
        with patch.object(sys, "argv", required):
            self.assertEqual(MODULE.parse_args().clean_action_regression_weight, 0.0)
        with patch.object(
            sys,
            "argv",
            [*required, "--clean-action-regression-weight", "1.0"],
        ):
            self.assertEqual(MODULE.parse_args().clean_action_regression_weight, 1.0)

    def test_clean_reconstruction_is_exact_for_fastwam_velocity_target(self):
        scheduler = MODULE.FlowMatchScheduler(shift=5.0)
        clean = torch.tensor(
            [
                [[-0.5, 0.25], [0.1, 0.8]],
                [[0.3, -0.2], [0.7, -0.9]],
            ],
            dtype=torch.float32,
        )
        noise = torch.tensor(
            [
                [[0.9, -0.1], [-0.4, 0.2]],
                [[-0.6, 0.5], [0.2, 0.4]],
            ],
            dtype=torch.float32,
        )
        timesteps = torch.tensor([250.0, 750.0])
        noisy = scheduler.add_noise(clean, noise, timesteps)
        velocity = scheduler.training_target(clean, noise, timesteps)
        reconstructed = MODULE.reconstruct_clean_action_from_flow(
            noisy, velocity, timesteps, scheduler
        )
        torch.testing.assert_close(reconstructed, clean)
        loss = MODULE.masked_chunk_regression_loss(
            reconstructed, clean, torch.zeros(2, 2, dtype=torch.bool)
        )
        self.assertLess(float(loss), 1.0e-12)

    def test_clean_regression_is_sigma_squared_velocity_error_and_masks_padding(self):
        scheduler = MODULE.FlowMatchScheduler(shift=5.0)
        clean = torch.zeros(2, 2, 2)
        noise = torch.ones_like(clean)
        timesteps = torch.tensor([250.0, 750.0])
        noisy = scheduler.add_noise(clean, noise, timesteps)
        target_velocity = scheduler.training_target(clean, noise, timesteps)
        velocity_error = torch.tensor(
            [
                [[1.0, -2.0], [1000.0, 1000.0]],
                [[0.5, -0.5], [2.0, -1.0]],
            ]
        )
        prediction = target_velocity + velocity_error
        reconstructed = MODULE.reconstruct_clean_action_from_flow(
            noisy, prediction, timesteps, scheduler
        )
        is_pad = torch.tensor([[False, True], [False, False]])
        actual = MODULE.masked_chunk_regression_loss(
            reconstructed, clean, is_pad
        )
        sigma = timesteps / 1000.0
        expected_elements = sigma[:, None, None].square() * velocity_error.square()
        valid = (~is_pad).unsqueeze(-1).expand_as(expected_elements)
        expected = torch.stack(
            [expected_elements[index][valid[index]].mean() for index in range(2)]
        ).mean()
        torch.testing.assert_close(actual, expected)

    def test_default_off_preserves_flow_and_enabled_reports_additive_loss(self):
        class TinyPolicy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.25))

            def forward(self, noisy_actions, timestep, **conditioning):
                del timestep, conditioning
                return noisy_actions * self.scale

        batch = {
            "actions": torch.linspace(-1.0, 1.0, 2 * 4 * 3).reshape(2, 4, 3),
            "action_is_pad": torch.tensor(
                [[False, False, False, True], [False, False, False, False]]
            ),
            "text_context": torch.zeros(2, 1, 2),
            "text_mask": torch.ones(2, 1, dtype=torch.bool),
            "features": torch.zeros(2, 1, 2, 2),
            "proprio": torch.zeros(2, 2),
        }
        scheduler = MODULE.FlowMatchScheduler(shift=5.0)
        default_model = TinyPolicy()
        default_optimizer = torch.optim.AdamW(default_model.parameters(), lr=1e-4)
        default_metrics = MODULE.optimizer_step(
            default_model,
            batch,
            default_optimizer,
            scheduler,
            seed=123,
            max_grad_norm=1.0,
        )
        self.assertEqual(default_metrics["loss"], default_metrics["flow_loss"])
        self.assertEqual(default_metrics["clean_action_regression_loss"], 0.0)
        self.assertEqual(
            default_metrics["weighted_clean_action_regression_loss"], 0.0
        )

        enabled_model = TinyPolicy()
        enabled_optimizer = torch.optim.AdamW(enabled_model.parameters(), lr=1e-4)
        enabled_metrics = MODULE.optimizer_step(
            enabled_model,
            batch,
            enabled_optimizer,
            scheduler,
            seed=123,
            max_grad_norm=1.0,
            clean_action_regression_weight=1.0,
        )
        self.assertGreater(enabled_metrics["clean_action_regression_loss"], 0.0)
        self.assertAlmostEqual(
            enabled_metrics["loss"],
            enabled_metrics["flow_loss"]
            + enabled_metrics["weighted_clean_action_regression_loss"],
            places=6,
        )
        self.assertNotEqual(
            float(default_model.scale.grad), float(enabled_model.scale.grad)
        )

    def test_negative_complement_weight_is_rejected(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            MODULE.optimizer_step(
                model,
                {"actions": torch.zeros(1, 1, 1)},
                optimizer,
                MODULE.FlowMatchScheduler(shift=5.0),
                seed=1,
                max_grad_norm=1.0,
                clean_action_regression_weight=-1.0,
            )

    def test_checkpoint_contract_is_unchanged_off_and_records_enabled_weight(self):
        spec = MODULE.ModelSpec()
        default_contract = MODULE.checkpoint_contract(
            self.checkpoint_args(0.0), spec, self.checkpoint_dataset()
        )
        self.assertNotIn("clean_action_regression_complement", default_contract)

        enabled_contract = MODULE.checkpoint_contract(
            self.checkpoint_args(1.0), spec, self.checkpoint_dataset()
        )
        complement = enabled_contract["clean_action_regression_complement"]
        self.assertEqual(complement["candidate"], "F")
        self.assertEqual(complement["weight"], 1.0)
        self.assertEqual(complement["extra_parameters"], 0)

    def test_fp32_timestep_avoids_bfloat16_zero_weight_endpoint(self):
        scheduler = MODULE.FlowMatchScheduler(shift=5.0)
        # Exact CUDA draw observed at the former deterministic step245 failure.
        uniform = torch.tensor([0.9919270873], dtype=torch.float32)
        sigma = scheduler._phi(uniform, scheduler.shift)
        timestep = sigma * float(scheduler.num_train_timesteps)
        self.assertEqual(timestep.dtype, torch.float32)
        self.assertLess(float(timestep), 1000.0)
        self.assertEqual(float(timestep.to(torch.bfloat16)), 1000.0)
        self.assertGreater(float(scheduler.training_weight(timestep)), 0.0)
        self.assertEqual(
            float(scheduler.training_weight(timestep.to(torch.bfloat16))), 0.0
        )

        actions = torch.zeros(1, 2, 2, dtype=torch.bfloat16)
        _, _, sampled_timestep = MODULE.deterministic_flow_batch(
            actions, scheduler, seed=245_000_777
        )
        self.assertEqual(sampled_timestep.dtype, torch.float32)

    def test_distributed_flow_seed_is_reproducible_and_rank_distinct(self):
        rank0 = MODULE.distributed_flow_seed(
            base_seed=42, completed_step=245, accumulation_index=0, rank=0
        )
        rank1 = MODULE.distributed_flow_seed(
            base_seed=42, completed_step=245, accumulation_index=0, rank=1
        )
        self.assertNotEqual(rank0, rank1)
        self.assertEqual(
            rank0,
            MODULE.distributed_flow_seed(
                base_seed=42, completed_step=245, accumulation_index=0, rank=0
            ),
        )

    def test_checkpoint_records_fp32_rank_distinct_flow_contract(self):
        contract = MODULE.checkpoint_contract(
            self.checkpoint_args(0.0), MODULE.ModelSpec(), self.checkpoint_dataset()
        )
        self.assertEqual(
            contract["flow_timestep_contract"],
            "continuous_fp32_no_bf16_endpoint_rounding_v2",
        )
        self.assertEqual(
            contract["flow_rng_contract"],
            "base_plus_step1000003_plus_rank10000019_v2",
        )


class StarWAMTrainerCheckpointTest(unittest.TestCase):
    def build_model(self):
        spec = MODULE.ModelSpec(
            context_dim=6,
            hidden_dim=8,
            ffn_dim=16,
            num_heads=2,
            attn_head_dim=4,
            action_layers=2,
            freq_dim=8,
            max_seq_len=32,
            gradient_checkpointing=False,
        )
        return spec, MODULE.build_model(
            spec, device=torch.device("cpu"), dtype=torch.float32
        )

    def build_batch(self):
        return {
            "sample_ids": ["probe"],
            "features": torch.randn(1, 1, 32, 5376),
            "actions": torch.randn(1, 32, 7),
            "proprio": torch.randn(1, 8),
            "action_is_pad": torch.zeros(1, 32, dtype=torch.bool),
            "text_context": torch.randn(1, 3, 6),
            "text_mask": torch.ones(1, 3, dtype=torch.bool),
        }

    def test_optimizer_update_and_exact_checkpoint_restore(self):
        torch.manual_seed(11)
        spec, model = self.build_model()
        batch = self.build_batch()
        scheduler = MODULE.FlowMatchScheduler(shift=5.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lr_scheduler = MODULE.build_lr_scheduler(
            optimizer,
            warmup_steps=2,
            scheduler_horizon=10,
            min_learning_rate=1e-6,
        )
        before = model.action_expert.head.weight.detach().clone()
        metrics = MODULE.optimizer_step(
            model,
            batch,
            optimizer,
            scheduler,
            seed=42,
            max_grad_norm=1.0,
        )
        self.assertTrue(metrics["loss"] > 0)
        self.assertTrue(MODULE.module_grad_norm(model.action_expert) > 0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        self.assertFalse(torch.equal(before, model.action_expert.head.weight))

        model.eval()
        noisy, _, timesteps = MODULE.deterministic_flow_batch(
            batch["actions"], scheduler, seed=99
        )
        with torch.inference_mode():
            prediction = MODULE.forward_policy(model, batch, noisy, timesteps)
        contract = {"model_spec": MODULE.asdict(spec), "test": True}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            MODULE.save_checkpoint_atomic(
                checkpoint,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                completed_steps=1,
                contract=contract,
                probe_prediction=prediction,
                probe_sample_ids=["probe"],
            )
            _, restored = self.build_model()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
            restored_lr_scheduler = MODULE.build_lr_scheduler(
                restored_optimizer,
                warmup_steps=2,
                scheduler_horizon=10,
                min_learning_rate=1e-6,
            )
            payload = MODULE.load_checkpoint_strict(
                checkpoint,
                model=restored,
                optimizer=restored_optimizer,
                lr_scheduler=restored_lr_scheduler,
                expected_contract=contract,
            )
            restored.eval()
            # Match the resumable restore probe: inference_mode would retain
            # an inference-only RoPE cache inside ActionDiT and poison the
            # next checkpointed backward.
            with torch.no_grad():
                restored_prediction = MODULE.forward_policy(
                    restored, batch, noisy, timesteps
                )
            torch.testing.assert_close(restored_prediction, prediction, rtol=0, atol=0)
            self.assertEqual(payload["completed_steps"], 1)
            self.assertEqual(
                restored_lr_scheduler.state_dict(), lr_scheduler.state_dict()
            )
            restored.train()
            restored_optimizer.zero_grad(set_to_none=True)
            resumed_metrics = MODULE.optimizer_step(
                restored,
                batch,
                restored_optimizer,
                scheduler,
                seed=100,
                max_grad_norm=1.0,
            )
            self.assertGreater(resumed_metrics["loss"], 0)
            self.assertGreater(MODULE.module_grad_norm(restored.action_expert), 0)

    def test_checkpoint_contract_rejects_change(self):
        _, model = self.build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lr_scheduler = MODULE.build_lr_scheduler(
            optimizer, warmup_steps=2, scheduler_horizon=10, min_learning_rate=1e-6
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            MODULE.save_checkpoint_atomic(
                checkpoint,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                completed_steps=0,
                contract={"action_shift": 5.0},
                probe_prediction=torch.zeros(1),
                probe_sample_ids=["probe"],
            )
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                MODULE.load_checkpoint_strict(
                    checkpoint,
                    model=model,
                    optimizer=None,
                    lr_scheduler=None,
                    expected_contract={"action_shift": 1.0},
                )

    def test_checkpoint_contract_rejects_feature_scale_change(self):
        _, model = self.build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lr_scheduler = MODULE.build_lr_scheduler(
            optimizer, warmup_steps=2, scheduler_horizon=10, min_learning_rate=1e-6
        )
        original = {"model_spec": {"feature_input_scale": 1.0}}
        changed = {
            "model_spec": {
                "feature_input_scale": MODULE.RMS_MATCH_FEATURE_INPUT_SCALE
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            MODULE.save_checkpoint_atomic(
                checkpoint,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                completed_steps=0,
                contract=original,
                probe_prediction=torch.zeros(1),
                probe_sample_ids=["probe"],
            )
            with self.assertRaisesRegex(ValueError, "model_spec"):
                MODULE.load_checkpoint_strict(
                    checkpoint,
                    model=model,
                    optimizer=None,
                    lr_scheduler=None,
                    expected_contract=changed,
                )

    def test_official_warmup_schedule_starts_at_one_over_warmup(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = MODULE.build_lr_scheduler(
            optimizer,
            warmup_steps=1000,
            scheduler_horizon=21700,
            min_learning_rate=1e-6,
        )
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-7, places=14)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.999e-7, places=14)

    def test_gradient_checkpointed_restore_probe_allows_next_backward(self):
        torch.manual_seed(17)
        spec = MODULE.ModelSpec(
            context_dim=6,
            hidden_dim=8,
            ffn_dim=16,
            num_heads=2,
            attn_head_dim=4,
            action_layers=2,
            freq_dim=8,
            max_seq_len=32,
            gradient_checkpointing=True,
            include_feature_timestep=False,
        )
        model = MODULE.build_model(spec, device=torch.device("cpu"), dtype=torch.float32)
        batch = self.build_batch()
        scheduler = MODULE.FlowMatchScheduler(shift=5.0)
        flags = MODULE.probe_action_state_inference_flags(
            model, batch, scheduler, seed=123
        )
        self.assertEqual(
            flags,
            {"tokens": False, "ctx": False, "t_mod": False, "freqs": False, "mask": False},
        )
        model.train()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-4,
        )
        metrics = MODULE.optimizer_step(
            model, batch, optimizer, scheduler, seed=124, max_grad_norm=1.0
        )
        self.assertGreater(metrics["loss"], 0)
        self.assertGreater(MODULE.module_grad_norm(model.action_expert), 0)


if __name__ == "__main__":
    unittest.main()
