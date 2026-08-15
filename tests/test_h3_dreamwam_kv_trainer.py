import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

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

    def test_h32_feature_cache_can_train_a_shorter_h2_action_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.build_cache(root)
            dataset = MODULE.CachedDreamWAMKVDataset(
                manifest,
                root,
                "kv",
                carrier_layers=self.layers,
                capture_token_count=5,
                num_heads=2,
                attn_head_dim=4,
                action_horizon=2,
            )
            item = dataset[0]
            self.assertEqual(tuple(item["actions"].shape), (2, 7))
            self.assertEqual(tuple(item["action_is_pad"].shape), (2,))
            self.assertEqual(dataset.cache_action_horizon, 4)

    def test_schema_rejects_storage_alias_after_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.build_cache(root, alias=True)
            with self.assertRaisesRegex(ValueError, "storage aliases"):
                self.dataset(root, manifest)[0]

    def test_parser_pins_h3_sha_and_requires_explicit_verify_opt_in(self):
        argv = [
            str(SCRIPT),
            "/tmp/manifest.jsonl",
            "--cache-root",
            "/tmp/cache",
            "--output",
            "/tmp/report.json",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = MODULE.parse_args()
        self.assertEqual(
            args.expected_h3_checkpoint_sha256,
            "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
        )
        self.assertFalse(args.verify_h3_checkpoint_sha256)
        self.assertFalse(args.enable_d0_repeat_layer49)
        with mock.patch.object(
            sys, "argv", [*argv, "--verify-h3-checkpoint-sha256"]
        ):
            args = MODULE.parse_args()
        self.assertTrue(args.verify_h3_checkpoint_sha256)
        with mock.patch.object(
            sys, "argv", [*argv, "--enable-d0-repeat-layer49"]
        ):
            args = MODULE.parse_args()
        self.assertTrue(args.enable_d0_repeat_layer49)

    def test_h3_checkpoint_hash_verify_and_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "h3.safetensors"
            checkpoint.write_bytes(b"pinned-h3-test")
            expected = MODULE.sha256_file(checkpoint)
            self.assertEqual(
                MODULE.verify_h3_checkpoint_sha256(
                    checkpoint,
                    expected_sha256=expected,
                    enabled=True,
                    rank=0,
                ),
                expected,
            )
            with self.assertRaisesRegex(ValueError, "H3 checkpoint SHA256 mismatch"):
                MODULE.verify_h3_checkpoint_sha256(
                    checkpoint,
                    expected_sha256="0" * 64,
                    enabled=True,
                    rank=0,
                )
            self.assertIsNone(
                MODULE.verify_h3_checkpoint_sha256(
                    Path(directory) / "missing.safetensors",
                    expected_sha256="0" * 64,
                    enabled=False,
                    rank=0,
                )
            )


class DreamWAMKVTrainingContractTest(unittest.TestCase):
    layers = (9, 19, 29, 39, 49)

    @staticmethod
    def data_state():
        return {
            "resume_mode": "explicit_stage_slice_v1",
            "sample_offset": 0,
            "limit": 1,
            "selected_windows": 1,
            "steps_in_invocation": 1,
            "sample_ids": ["probe"],
            "sampler_cursor_restorable": False,
        }

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

    def test_d0_model_is_parameter_identical_single_variable_ablation(self):
        aligned_spec, aligned = self.build_model()
        d0_spec = MODULE.ModelSpec(
            **{
                **MODULE.asdict(aligned_spec),
                "carrier_source_mode": MODULE.REPEAT_LAYER49_CARRIER_SOURCE,
            }
        )
        torch.manual_seed(313)
        aligned = MODULE.build_model(
            aligned_spec, device=torch.device("cpu"), dtype=torch.float32
        )
        torch.manual_seed(313)
        d0 = MODULE.build_model(
            d0_spec, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertEqual(set(aligned.state_dict()), set(d0.state_dict()))
        for name, tensor in aligned.state_dict().items():
            torch.testing.assert_close(tensor, d0.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(d0.carrier_source_mode, MODULE.REPEAT_LAYER49_CARRIER_SOURCE)

    def test_d0_checkpoint_contract_changes_only_derived_identity_and_mode(self):
        aligned_spec, _ = self.build_model()
        d0_spec = MODULE.ModelSpec(
            **{
                **MODULE.asdict(aligned_spec),
                "carrier_source_mode": MODULE.REPEAT_LAYER49_CARRIER_SOURCE,
            }
        )
        args = Namespace(
            expected_h3_checkpoint_sha256=MODULE.H3_INT8_CHECKPOINT_SHA256,
            verify_h3_checkpoint_sha256=True,
            kv_subdir="same-kv",
            capture_token_count=5,
            action_horizon=4,
            action_shift=5.0,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            num_workers=0,
            seed=42,
            learning_rate=1e-4,
            min_learning_rate=1e-6,
            warmup_steps=10,
            scheduler_horizon=100,
        )
        dataset = Namespace(
            first_checkpoint_path=Path("/models/h3"),
            source_manifest_sha256="source",
            source_manifest_items=100,
            manifest_sha256="train",
            manifest_items=80,
            stats_sha256="stats",
        )
        aligned = MODULE.checkpoint_contract(args, aligned_spec, dataset)
        d0 = MODULE.checkpoint_contract(args, d0_spec, dataset)
        self.assertEqual(aligned["candidate"], "D")
        self.assertEqual(d0["candidate"], "D0")
        differing = {
            key for key in aligned if aligned[key] != d0[key]
        }
        self.assertEqual(differing, {"candidate", "carrier_source_mode", "model_spec"})
        model_spec_differences = {
            key
            for key in aligned["model_spec"]
            if aligned["model_spec"][key] != d0["model_spec"][key]
        }
        self.assertEqual(model_spec_differences, {"carrier_source_mode"})

    def test_grid_adaptation_contract_records_parent_and_restored_optimizer(self):
        spec, _ = self.build_model()
        spec = MODULE.ModelSpec(
            **{
                **MODULE.asdict(spec),
                "carrier_source_mode": MODULE.REPEAT_LAYER49_CARRIER_SOURCE,
            }
        )
        args = Namespace(
            expected_h3_checkpoint_sha256=MODULE.H3_INT8_CHECKPOINT_SHA256,
            verify_h3_checkpoint_sha256=True,
            kv_subdir="grid-kv",
            kv_pool_strategy=MODULE.DUAL_VIEW_GRID_KV_STRATEGY,
            capture_token_count=32,
            action_horizon=32,
            action_shift=5.0,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            num_workers=0,
            seed=42,
            learning_rate=1e-4,
            min_learning_rate=1e-6,
            warmup_steps=1000,
            scheduler_horizon=21700,
            initialization_kind="kv_pool_adaptation",
            initialization_parent_sha256="a" * 64,
            initialization_parent_completed_steps=14000,
        )
        dataset = Namespace(
            first_checkpoint_path=Path("/models/h3"),
            source_manifest_sha256="source",
            source_manifest_items=100,
            manifest_sha256="train",
            manifest_items=80,
            stats_sha256="stats",
        )
        contract = MODULE.checkpoint_contract(args, spec, dataset)
        self.assertEqual(contract["kv_strategy"], MODULE.DUAL_VIEW_GRID_KV_STRATEGY)
        self.assertEqual(contract["initialization_kind"], "kv_pool_adaptation")
        self.assertEqual(contract["initialization_parent_completed_steps"], 14000)
        self.assertTrue(contract["initialization_optimizer_scheduler_restored"])

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

    def test_bf16_forward_autocast_preserves_fp32_flow_timestep(self):
        class TimeEmbeddingProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.time_embedding = torch.nn.Linear(1, 2).to(torch.bfloat16)
                self.seen_timestep_dtype = None

            def forward(
                self,
                noisy,
                timestep,
                *,
                text_context,
                proprio,
                video_kv_cache,
                text_mask,
            ):
                del text_context, proprio, video_kv_cache, text_mask
                self.seen_timestep_dtype = timestep.dtype
                embedded = self.time_embedding(timestep[:, None])
                return embedded[:, None, :].expand_as(noisy)

        model = TimeEmbeddingProbe()
        noisy = torch.randn(1, 4, 2, dtype=torch.bfloat16)
        timesteps = torch.tensor([999.25], dtype=torch.float32)
        batch = {
            "text_context": torch.empty(1, 0, 1, dtype=torch.bfloat16),
            "proprio": torch.empty(1, 0, dtype=torch.bfloat16),
            "video_kv_cache": {},
            "text_mask": torch.empty(1, 0, dtype=torch.bool),
        }
        prediction = MODULE.forward_policy(model, batch, noisy, timesteps)
        self.assertEqual(model.seen_timestep_dtype, torch.float32)
        self.assertEqual(prediction.dtype, torch.bfloat16)
        self.assertEqual(tuple(prediction.shape), tuple(noisy.shape))

    def test_checkpoint_round_trip_and_contract_mismatch(self):
        torch.manual_seed(13)
        MODULE.random.seed(13)
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
        prediction = MODULE.forward_policy(model, batch, noisy, timesteps)
        prediction.square().mean().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            prediction = MODULE.forward_policy(model, batch, noisy, timesteps)
        contract = {
            "candidate": "D",
            "model_spec": MODULE.asdict(spec),
            "training_topology": {"world_size": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate_d.pt"
            rng_states = [MODULE.capture_rng_state(torch.device("cpu"))]
            MODULE.save_checkpoint_atomic(
                checkpoint,
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                completed_steps=1,
                contract=contract,
                probe_prediction=prediction,
                probe_sample_ids=["probe"],
                rng_states=rng_states,
                data_state=self.data_state(),
            )
            expected_python_random = MODULE.random.random()
            expected_torch_random = torch.rand(4)
            restored = MODULE.build_model(
                spec, device=torch.device("cpu"), dtype=torch.float32
            )
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
            restored_scheduler = MODULE.PARENT.build_lr_scheduler(
                restored_optimizer,
                warmup_steps=1,
                scheduler_horizon=4,
                min_learning_rate=1e-6,
            )
            payload = MODULE.load_checkpoint_strict(
                checkpoint,
                model=restored,
                optimizer=restored_optimizer,
                lr_scheduler=restored_scheduler,
                expected_contract=contract,
                restore_rng_rank=0,
                rng_device=torch.device("cpu"),
            )
            with torch.no_grad():
                actual = MODULE.forward_policy(restored, batch, noisy, timesteps)
            torch.testing.assert_close(
                actual, payload["probe_prediction"], rtol=0, atol=0
            )
            self.assertEqual(MODULE.random.random(), expected_python_random)
            torch.testing.assert_close(torch.rand(4), expected_torch_random)
            self.assertEqual(
                restored_scheduler.state_dict(), scheduler.state_dict()
            )
            expected_optimizer = optimizer.state_dict()
            actual_optimizer = restored_optimizer.state_dict()
            self.assertEqual(
                actual_optimizer["param_groups"], expected_optimizer["param_groups"]
            )
            self.assertEqual(
                set(actual_optimizer["state"]), set(expected_optimizer["state"])
            )
            for parameter_id, expected_state in expected_optimizer["state"].items():
                actual_state = actual_optimizer["state"][parameter_id]
                self.assertEqual(set(actual_state), set(expected_state))
                for key, expected_value in expected_state.items():
                    if torch.is_tensor(expected_value):
                        torch.testing.assert_close(
                            actual_state[key], expected_value, rtol=0, atol=0
                        )
                    else:
                        self.assertEqual(actual_state[key], expected_value)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                MODULE.load_checkpoint_strict(
                    checkpoint,
                    model=restored,
                    optimizer=None,
                    lr_scheduler=None,
                    expected_contract={**contract, "candidate": "not-D"},
                )

    def test_atomic_checkpoint_failure_removes_partial_and_preserves_target(self):
        spec, model = self.build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = MODULE.PARENT.build_lr_scheduler(
            optimizer,
            warmup_steps=1,
            scheduler_horizon=4,
            min_learning_rate=1e-6,
        )
        contract = {
            "candidate": "D",
            "model_spec": MODULE.asdict(spec),
            "training_topology": {"world_size": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate_d.pt"
            checkpoint.write_bytes(b"previous-good-checkpoint")

            def fail_after_partial(_payload, path):
                Path(path).write_bytes(b"incomplete")
                raise OSError("injected save failure")

            with mock.patch.object(MODULE.torch, "save", side_effect=fail_after_partial):
                with self.assertRaisesRegex(OSError, "injected save failure"):
                    MODULE.save_checkpoint_atomic(
                        checkpoint,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=scheduler,
                        completed_steps=0,
                        contract=contract,
                        probe_prediction=torch.zeros(1),
                        probe_sample_ids=["probe"],
                        rng_states=[MODULE.capture_rng_state(torch.device("cpu"))],
                        data_state=self.data_state(),
                    )
            self.assertEqual(checkpoint.read_bytes(), b"previous-good-checkpoint")
            self.assertEqual(list(Path(directory).glob("*.partial")), [])

    def test_invalid_rng_schema_is_rejected_before_model_mutation(self):
        spec, model = self.build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = MODULE.PARENT.build_lr_scheduler(
            optimizer,
            warmup_steps=1,
            scheduler_horizon=4,
            min_learning_rate=1e-6,
        )
        contract = {
            "candidate": "D",
            "model_spec": MODULE.asdict(spec),
            "training_topology": {"world_size": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate_d.pt"
            MODULE.save_checkpoint_atomic(
                checkpoint,
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                completed_steps=0,
                contract=contract,
                probe_prediction=torch.zeros(1),
                probe_sample_ids=["probe"],
                rng_states=[MODULE.capture_rng_state(torch.device("cpu"))],
                data_state=self.data_state(),
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["rng_states"] = []
            torch.save(payload, checkpoint)
            restored = MODULE.build_model(
                spec, device=torch.device("cpu"), dtype=torch.float32
            )
            before = {
                name: tensor.detach().clone()
                for name, tensor in restored.state_dict().items()
            }
            with self.assertRaisesRegex(ValueError, "RNG state count"):
                MODULE.load_checkpoint_strict(
                    checkpoint,
                    model=restored,
                    optimizer=None,
                    lr_scheduler=None,
                    expected_contract=contract,
                )
            for name, tensor in restored.state_dict().items():
                torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

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

    def test_consumed_ids_are_flattened_across_ranks_without_duplicates(self):
        flattened = MODULE.flatten_consumed_sample_ids(
            [["r0_a", "r0_b"], ["r1_a", "r1_b"]],
            expected_per_rank=2,
        )
        self.assertEqual(flattened, ["r0_a", "r0_b", "r1_a", "r1_b"])
        with self.assertRaisesRegex(RuntimeError, "duplicate consumed sample IDs"):
            MODULE.flatten_consumed_sample_ids(
                [["shared"], ["shared"]], expected_per_rank=1
            )
        with self.assertRaisesRegex(RuntimeError, "consumed 1 samples, expected 2"):
            MODULE.flatten_consumed_sample_ids(
                [["r0_a", "r0_b"], ["r1_a"]], expected_per_rank=2
            )

    def test_three_stage_consumed_ids_remain_cumulative_and_block_old_overlap(self):
        cumulative = MODULE.merge_cumulative_consumed_sample_ids([], ["stage1"])
        cumulative = MODULE.merge_cumulative_consumed_sample_ids(
            cumulative, ["stage2"]
        )
        cumulative = MODULE.merge_cumulative_consumed_sample_ids(
            cumulative, ["stage3"]
        )
        self.assertEqual(cumulative, ["stage1", "stage2", "stage3"])
        with self.assertRaisesRegex(RuntimeError, "overlaps historical"):
            MODULE.merge_cumulative_consumed_sample_ids(
                cumulative, ["stage1"]
            )


if __name__ == "__main__":
    unittest.main()
