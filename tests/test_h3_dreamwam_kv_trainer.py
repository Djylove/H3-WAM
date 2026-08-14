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
    / "scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py"
)
SPEC = importlib.util.spec_from_file_location(
    "train_h3_int8_dreamwam_kv_carrier", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DreamWAMKVDatasetTest(unittest.TestCase):
    layers = (9, 19, 29, 39, 49)

    def build_cache(self, root: Path, *, alias: bool = False) -> tuple[Path, str]:
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
        (root / "kv").mkdir()
        (root / "windows").mkdir()
        (root / "contexts").mkdir()
        video_kv_cache = {
            layer: {
                "k": torch.randn(5, 2, 4),
                "v": torch.randn(5, 2, 4),
            }
            for layer in self.layers
        }
        if alias:
            video_kv_cache[19]["k"] = video_kv_cache[9]["k"]
        torch.save(
            {
                "schema": MODULE.DREAMWAM_KV_SCHEMA,
                "video_kv_cache": video_kv_cache,
                "layers": self.layers,
                "capture_token_count": 5,
                "num_heads": 2,
                "attn_head_dim": 4,
                "capture_token_strategy": MODULE.DREAMWAM_KV_STRATEGY,
                "dreamwam_commit": MODULE.DREAMWAM_COMMIT,
                "episode": 1,
                "start": 0,
                "suite": "libero_spatial",
                "context_id": context_id,
                "timestep": 1.0,
                "action_horizon": 4,
                "backbone": MODULE.CACHE_BACKBONE,
                "quantization": MODULE.CACHE_QUANTIZATION,
                "checkpoint": "/tmp/h3-int8.safetensors",
                "manifest_items": 1,
            },
            root / "kv" / f"{sample_id}.pt",
        )
        torch.save(
            {
                "actions": torch.linspace(-1, 1, 4 * 7).reshape(4, 7),
                "state": torch.linspace(-1, 1, 8),
                "action_is_pad": torch.zeros(4, dtype=torch.bool),
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

    def dataset(self, root: Path, manifest: Path):
        return MODULE.CachedDreamWAMKVDataset(
            manifest,
            root,
            "kv",
            carrier_layers=self.layers,
            capture_token_count=5,
            num_heads=2,
            attn_head_dim=4,
            action_horizon=4,
        )

    def test_schema_maps_five_real_h3_layers_to_five_batch_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, sample_id = self.build_cache(root)
            item = self.dataset(root, manifest)[0]
            self.assertEqual(item["sample_id"], sample_id)
            self.assertEqual(tuple(item["video_kv_cache"]), self.layers)
            batch = MODULE.collate_cached_batch([item])
            for layer in self.layers:
                self.assertEqual(
                    tuple(batch["video_kv_cache"][layer]["k"].shape),
                    (1, 5, 2, 4),
                )

    def test_schema_rejects_storage_alias_after_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.build_cache(root, alias=True)
            with self.assertRaisesRegex(ValueError, "storage aliases"):
                self.dataset(root, manifest)[0]


class DreamWAMKVTrainingContractTest(unittest.TestCase):
    layers = (9, 19, 29, 39, 49)

    def build_model(self):
        spec = MODULE.ModelSpec(
            action_dim=2,
            proprio_dim=3,
            context_dim=6,
            hidden_dim=8,
            ffn_dim=16,
            num_heads=2,
            attn_head_dim=4,
            freq_dim=8,
            carrier_layers=self.layers,
        )
        return spec, MODULE.build_model(
            spec, device=torch.device("cpu"), dtype=torch.float32
        )

    def batch(self):
        return {
            "sample_ids": ["probe"],
            "video_kv_cache": {
                layer: {
                    "k": torch.randn(1, 5, 2, 4),
                    "v": torch.randn(1, 5, 2, 4),
                }
                for layer in self.layers
            },
            "actions": torch.randn(1, 4, 2),
            "proprio": torch.randn(1, 3),
            "action_is_pad": torch.zeros(1, 4, dtype=torch.bool),
            "text_context": torch.randn(1, 3, 6),
            "text_mask": torch.ones(1, 3, dtype=torch.bool),
        }

    def test_five_blocks_receive_five_layers_and_gradients(self):
        torch.manual_seed(11)
        _, model = self.build_model()
        batch = self.batch()
        scheduler = MODULE.FlowMatchScheduler(shift=5.0)
        noisy, target, timesteps = MODULE.PARENT.deterministic_flow_batch(
            batch["actions"], scheduler, seed=42
        )
        prediction = MODULE.forward_policy(model, batch, noisy, timesteps)
        self.assertEqual(tuple(prediction.shape), (1, 4, 2))
        loss = MODULE.flow_matching_loss(
            prediction,
            target,
            timesteps,
            scheduler,
            is_pad_mask=batch["action_is_pad"],
        )
        loss.backward()
        self.assertEqual(len(model.action_expert.blocks), 5)
        for block in model.action_expert.blocks:
            self.assertIsNotNone(block.self_attn.q.weight.grad)
            self.assertGreater(float(block.self_attn.q.weight.grad.norm()), 0.0)
        self.assertGreater(float(model.proprio_encoder.weight.grad.norm()), 0.0)

    def test_checkpoint_round_trip_and_contract_mismatch(self):
        torch.manual_seed(13)
        spec, model = self.build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = MODULE.PARENT.build_lr_scheduler(
            optimizer,
            warmup_steps=1,
            scheduler_horizon=4,
            min_learning_rate=1e-6,
        )
        batch = self.batch()
        flow = MODULE.FlowMatchScheduler(shift=5.0)
        noisy, _, timesteps = MODULE.PARENT.deterministic_flow_batch(
            batch["actions"], flow, seed=99
        )
        with torch.no_grad():
            prediction = MODULE.forward_policy(model, batch, noisy, timesteps)
        contract = {"candidate": "D", "model_spec": MODULE.asdict(spec)}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate_d.pt"
            MODULE.save_checkpoint_atomic(
                checkpoint,
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                completed_steps=0,
                contract=contract,
                probe_prediction=prediction,
                probe_sample_ids=["probe"],
            )
            restored = MODULE.build_model(
                spec, device=torch.device("cpu"), dtype=torch.float32
            )
            payload = MODULE.load_checkpoint_strict(
                checkpoint,
                model=restored,
                optimizer=None,
                lr_scheduler=None,
                expected_contract=contract,
            )
            with torch.no_grad():
                actual = MODULE.forward_policy(restored, batch, noisy, timesteps)
            torch.testing.assert_close(actual, payload["probe_prediction"])
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                MODULE.load_checkpoint_strict(
                    checkpoint,
                    model=restored,
                    optimizer=None,
                    lr_scheduler=None,
                    expected_contract={**contract, "candidate": "not-D"},
                )

    def test_parent_v2_flow_rng_is_rank_distinct(self):
        rank0 = MODULE.PARENT.distributed_flow_seed(
            base_seed=42, completed_step=245, accumulation_index=0, rank=0
        )
        rank1 = MODULE.PARENT.distributed_flow_seed(
            base_seed=42, completed_step=245, accumulation_index=0, rank=1
        )
        self.assertNotEqual(rank0, rank1)
        self.assertEqual(
            rank1 - rank0,
            10_000_019,
        )


if __name__ == "__main__":
    unittest.main()
