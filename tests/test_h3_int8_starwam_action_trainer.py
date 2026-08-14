import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

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
