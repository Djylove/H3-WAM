from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/h3wam/evaluate_h3_dreamwam_kv_carrier.py"


def load_module():
    name = "_test_h3_dreamwam_kv_evaluator"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVAL = load_module()


class RecordingPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cache = None

    def forward(
        self,
        noisy_actions,
        timestep,
        *,
        text_context,
        proprio,
        video_kv_cache,
        text_mask,
    ):
        self.cache = video_kv_cache
        return torch.zeros_like(noisy_actions)


class DreamWAMKVEvaluatorTest(unittest.TestCase):
    def config(self, **changes):
        values = dict(
            checkpoint=Path("candidate_d.pt"),
            source_manifest=Path("source.jsonl"),
            train_manifest=Path("train.jsonl"),
            val_manifest=Path("val.jsonl"),
            cache_root=Path("cache"),
        )
        values.update(changes)
        return EVAL.EvalConfig(**values)

    def test_balanced80_contract_is_not_configurable(self):
        EVAL.require_balanced80_protocol(self.config())
        for field, value in (
            ("seed", 43),
            ("batch_size", 2),
            ("samples_per_task", 1),
            ("inference_steps", 9),
            ("action_shift", 1.0),
            ("language_sensitivity", False),
            ("visual_feature_shuffle", False),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                EVAL.require_balanced80_protocol(self.config(**{field: value}))
        with self.assertRaises(ValueError):
            EVAL.require_balanced80_protocol(
                self.config(cache_audit_aggregate_sha256="not-a-sha")
            )
        with self.assertRaises(ValueError):
            EVAL.require_balanced80_protocol(
                self.config(expected_selected_ids_sha256="not-a-sha")
            )

    def test_selected_id_gate_defaults_to_audited_v4_not_v8_parent(self):
        config = self.config()
        self.assertEqual(
            config.expected_selected_ids_sha256,
            EVAL.CANDIDATE_D_V4_SELECTED_IDS_SHA256,
        )
        self.assertNotEqual(
            config.expected_selected_ids_sha256,
            EVAL.R1_G_V8_SELECTED_IDS_SHA256,
        )
        self.assertEqual(
            EVAL.require_selected_ids(
                {"selected_ids_sha256": EVAL.CANDIDATE_D_V4_SELECTED_IDS_SHA256},
                config,
            ),
            EVAL.CANDIDATE_D_V4_SELECTED_IDS_SHA256,
        )
        with self.assertRaises(ValueError):
            EVAL.require_selected_ids(
                {"selected_ids_sha256": EVAL.R1_G_V8_SELECTED_IDS_SHA256}, config
            )
        migrated = self.config(
            expected_selected_ids_sha256=EVAL.R1_G_V8_SELECTED_IDS_SHA256
        )
        self.assertEqual(
            EVAL.require_selected_ids(
                {"selected_ids_sha256": EVAL.R1_G_V8_SELECTED_IDS_SHA256}, migrated
            ),
            EVAL.R1_G_V8_SELECTED_IDS_SHA256,
        )

    def test_protocol_policy_unpacks_complete_distinct_kv_bundle(self):
        carrier = torch.arange(2 * 5 * 2 * 3 * 2 * 4, dtype=torch.float32).reshape(
            2, 5, 2, 3, 2, 4
        )
        recording = RecordingPolicy()
        wrapper = EVAL.DreamWAMKVProtocolPolicy(
            recording, EVAL.CANDIDATE_D.DEFAULT_H3_CARRIER_LAYERS
        )
        output = wrapper(
            torch.ones(2, 4, 7),
            torch.ones(2),
            text_context=torch.ones(2, 3, 5120),
            h3_features=carrier,
            proprio=torch.ones(2, 8),
            text_mask=torch.ones(2, 3, dtype=torch.bool),
        )
        self.assertEqual(tuple(output.shape), (2, 4, 7))
        self.assertEqual(set(recording.cache), {9, 19, 29, 39, 49})
        signatures = []
        for index, layer in enumerate((9, 19, 29, 39, 49)):
            self.assertTrue(torch.equal(recording.cache[layer]["k"], carrier[:, index, 0]))
            self.assertTrue(torch.equal(recording.cache[layer]["v"], carrier[:, index, 1]))
            signatures.extend(
                recording.cache[layer][name].untyped_storage().data_ptr()
                for name in ("k", "v")
            )
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_candidate_d_contract_is_strict(self):
        spec = {
            "action_dim": 7,
            "proprio_dim": 8,
            "context_dim": 5120,
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "num_heads": 56,
            "attn_head_dim": 128,
            "freq_dim": 256,
            "carrier_layers": (9, 19, 29, 39, 49),
            "carrier_source_mode": EVAL.CANDIDATE_D.ALIGNED_5LAYER_CARRIER_SOURCE,
        }
        contract = {
            "candidate": "D",
            "classification": "action-only-on-frozen-features",
            "dreamwam_commit": EVAL.CANDIDATE_D.DREAMWAM_COMMIT,
            "dreamwam_layers_sha256": EVAL.CANDIDATE_D.DREAMWAM_LAYERS_SHA256,
            "dreamwam_experts_sha256": EVAL.CANDIDATE_D.DREAMWAM_EXPERTS_SHA256,
            "dreamwam_mot_sha256": EVAL.CANDIDATE_D.DREAMWAM_MOT_SHA256,
            "parent_shifted_flow_commit": EVAL.CANDIDATE_D.PARENT_OBJECTIVE_COMMIT,
            "carrier_source_mode": EVAL.CANDIDATE_D.ALIGNED_5LAYER_CARRIER_SOURCE,
            "h3_checkpoint_path": "/models/h3",
            "h3_checkpoint_sha256": EVAL.EXPECTED_H3_CHECKPOINT_SHA256,
            "verify_h3_checkpoint_sha256": True,
            "kv_subdir": "kv",
            "kv_schema": EVAL.CANDIDATE_D.DREAMWAM_KV_SCHEMA,
            "kv_strategy": EVAL.CANDIDATE_D.DREAMWAM_KV_STRATEGY,
            "kv_layers": [9, 19, 29, 39, 49],
            "kv_tokens": 32,
            "kv_num_heads": 56,
            "kv_attn_head_dim": 128,
            "kv_bytes_per_sample": EVAL.CANDIDATE_D.h3_kv_cache_bytes(
                layers=5, tokens=32, heads=56, head_dim=128
            ),
            "source_manifest_sha256": "source",
            "source_manifest_items": 100,
            "split_manifest_sha256": "train",
            "split_manifest_items": 80,
            "stats_sha256": "stats",
            "action_normalization": "starwam_minmax_clip5",
            "state_normalization": "starwam_minmax_clip5",
            "action_horizon": 32,
            "action_shift": 5.0,
            "flow_timestep_contract": "continuous_fp32_no_bf16_endpoint_rounding_v2",
            "flow_rng_contract": "base_plus_step1000003_plus_rank10000019_v2",
            "training_topology": {},
            "lr_schedule": {},
            "model_spec": spec,
        }
        payload = {key: None for key in EVAL.CANDIDATE_D.CHECKPOINT_KEYS}
        payload.update(
            schema_version=EVAL.CANDIDATE_D.CHECKPOINT_SCHEMA,
            completed_steps=1,
            model={"weight": torch.ones(1)},
            contract=contract,
        )
        actual, actual_spec, subdir = EVAL._require_contract(
            payload,
            config=self.config(),
            source_manifest_sha256="source",
            source_manifest_items=100,
            train_manifest_sha256="train",
            train_manifest_items=80,
            stats_sha256="stats",
        )
        self.assertIs(actual, contract)
        self.assertIs(actual_spec, spec)
        self.assertEqual(subdir, "kv")
        contract["candidate"] = "D0"
        contract["carrier_source_mode"] = EVAL.CANDIDATE_D.REPEAT_LAYER49_CARRIER_SOURCE
        spec["carrier_source_mode"] = EVAL.CANDIDATE_D.REPEAT_LAYER49_CARRIER_SOURCE
        d0_contract, d0_spec, _ = EVAL._require_contract(
            payload,
            config=self.config(),
            source_manifest_sha256="source",
            source_manifest_items=100,
            train_manifest_sha256="train",
            train_manifest_items=80,
            stats_sha256="stats",
        )
        self.assertEqual(d0_contract["candidate"], "D0")
        self.assertEqual(
            d0_spec["carrier_source_mode"],
            EVAL.CANDIDATE_D.REPEAT_LAYER49_CARRIER_SOURCE,
        )
        contract["candidate"] = "D"
        contract["carrier_source_mode"] = EVAL.CANDIDATE_D.ALIGNED_5LAYER_CARRIER_SOURCE
        spec["carrier_source_mode"] = EVAL.CANDIDATE_D.ALIGNED_5LAYER_CARRIER_SOURCE
        contract["h3_checkpoint_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            EVAL._require_contract(
                payload,
                config=self.config(),
                source_manifest_sha256="source",
                source_manifest_items=100,
                train_manifest_sha256="train",
                train_manifest_items=80,
                stats_sha256="stats",
            )
        contract["h3_checkpoint_sha256"] = EVAL.EXPECTED_H3_CHECKPOINT_SHA256
        contract["verify_h3_checkpoint_sha256"] = False
        with self.assertRaises(ValueError):
            EVAL._require_contract(
                payload,
                config=self.config(),
                source_manifest_sha256="source",
                source_manifest_items=100,
                train_manifest_sha256="train",
                train_manifest_items=80,
                stats_sha256="stats",
            )
        contract["verify_h3_checkpoint_sha256"] = True
        contract["kv_tokens"] = 31
        with self.assertRaises(ValueError):
            EVAL._require_contract(
                payload,
                config=self.config(),
                source_manifest_sha256="source",
                source_manifest_items=100,
                train_manifest_sha256="train",
                train_manifest_items=80,
                stats_sha256="stats",
            )

    def test_dataset_uses_canonical_action_path_and_shuffles_whole_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kv").mkdir()
            (root / "windows").mkdir()
            (root / "contexts").mkdir()
            torch.save(
                {
                    "action_min": torch.zeros(7),
                    "action_max": torch.ones(7) * 2,
                    "state_min": torch.zeros(8),
                    "state_max": torch.ones(8) * 2,
                },
                root / "stats.pt",
            )
            rows = []
            for index, sample_id in enumerate(("a", "b")):
                context_id = f"context-{index}"
                rows.append(
                    {"id": sample_id, "context_id": context_id, "task": f"task-{index}"}
                )
                torch.save(
                    {
                        "text_only": True,
                        "token_tags": torch.ones(2, dtype=torch.long),
                        "context": torch.full((1, 2, 4), float(index)),
                    },
                    root / "contexts" / f"{context_id}.pt",
                )
                torch.save(
                    {
                        "actions": torch.ones(32, 7),
                        "state": torch.ones(8),
                        "action_is_pad": torch.zeros(32, dtype=torch.bool),
                    },
                    root / "windows" / f"{sample_id}.pt",
                )
                cache = {
                    layer: {
                        name: torch.full(
                            (32, 2, 3),
                            float(index * 100 + layer + (name == "v")),
                            dtype=torch.bfloat16,
                        )
                        for name in ("k", "v")
                    }
                    for layer in (9, 19, 29, 39, 49)
                }
                torch.save(
                    {
                        "schema": EVAL.CANDIDATE_D.DREAMWAM_KV_SCHEMA,
                        "layers": (9, 19, 29, 39, 49),
                        "capture_token_count": 32,
                        "num_heads": 2,
                        "attn_head_dim": 3,
                        "capture_token_strategy": EVAL.CANDIDATE_D.DREAMWAM_KV_STRATEGY,
                        "dreamwam_commit": EVAL.CANDIDATE_D.DREAMWAM_COMMIT,
                        "context_id": context_id,
                        "action_horizon": 32,
                        "backbone": EVAL.CANDIDATE_D.CACHE_BACKBONE,
                        "quantization": EVAL.CANDIDATE_D.CACHE_QUANTIZATION,
                        "manifest_items": 2,
                        "timestep": 1.0,
                        "checkpoint": "/models/h3",
                        "video_kv_cache": cache,
                    },
                    root / "kv" / f"{sample_id}.pt",
                )
            model_spec = {
                "action_dim": 7,
                "proprio_dim": 8,
                "context_dim": 4,
                "num_heads": 2,
                "attn_head_dim": 3,
                "carrier_layers": (9, 19, 29, 39, 49),
                "carrier_source_mode": EVAL.CANDIDATE_D.ALIGNED_5LAYER_CARRIER_SOURCE,
            }
            dataset = EVAL.CachedDreamWAMKVValidationDataset(
                rows,
                cache_root=root,
                kv_subdir="kv",
                source_manifest_items=2,
                model_spec=model_spec,
                action_horizon=32,
                h3_checkpoint_path="/models/h3",
                visual_feature_shuffle={"a": "b", "b": "a"},
            )
            item = dataset[0]
            self.assertEqual(tuple(item["features"].shape), (5, 2, 32, 2, 3))
            self.assertTrue(torch.equal(item["actions"], torch.zeros(32, 7)))
            self.assertTrue(torch.equal(item["raw_actions"], torch.ones(32, 7)))
            self.assertEqual(item["visual_shuffle_source_id"], "b")
            self.assertFalse(torch.equal(item["features"], item["shuffled_features"]))

            short_dataset = EVAL.CachedDreamWAMKVValidationDataset(
                rows,
                cache_root=root,
                kv_subdir="kv",
                source_manifest_items=2,
                model_spec=model_spec,
                action_horizon=8,
                h3_checkpoint_path="/models/h3",
                visual_feature_shuffle={"a": "b", "b": "a"},
            )
            short_item = short_dataset[0]
            self.assertEqual(tuple(short_item["actions"].shape), (8, 7))
            self.assertEqual(tuple(short_item["raw_actions"].shape), (8, 7))


if __name__ == "__main__":
    unittest.main()
