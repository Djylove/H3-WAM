from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from fastwam.models.h3wam.dreamwam_kv_carrier import H3DreamWAMKVCarrierPolicy
from fastwam.models.h3wam.fact_backbone_port import (
    C59FailureOverlay,
    C60CausalFailureLabels,
    FACTTokenLayout,
    H3FACTBackbonePort,
    build_fact_teacher_forcing_mask,
    fact_backbone_port_losses,
)


class H3FACTBackbonePortTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260816)

    @staticmethod
    def build_model() -> H3FACTBackbonePort:
        carrier = H3DreamWAMKVCarrierPolicy(
            enabled=True,
            carrier_layers=(1, 3),
            action_dim=2,
            proprio_dim=3,
            context_dim=6,
            hidden_dim=16,
            ffn_dim=32,
            num_heads=2,
            attn_head_dim=4,
            freq_dim=8,
        )
        return H3FACTBackbonePort(
            carrier,
            hidden_dim=16,
            future_representation_dim=5,
            future_state_dim=3,
            causal_layers=2,
            causal_heads=4,
            causal_ffn_dim=32,
            h3_source_layer=3,
        )

    @staticmethod
    def inputs(batch: int = 2) -> dict:
        cache = {}
        for layer in (1, 3):
            cache[layer] = {
                "k": torch.randn(batch, 4, 2, 4).clone(),
                "v": torch.randn(batch, 4, 2, 4).clone(),
            }
        return {
            "noisy_actions": torch.randn(batch, 4, 2),
            "clean_actions": torch.randn(batch, 4, 2),
            "timestep": torch.tensor([800.0, 250.0])[:batch],
            "noisy_future_state": torch.randn(batch, 3),
            "noisy_value": torch.randn(batch),
            "noisy_future_representation": torch.randn(batch, 5),
            "text_context": torch.randn(batch, 3, 6),
            "text_mask": torch.tensor([[True, True, False], [True, True, True]])[:batch],
            "proprio": torch.randn(batch, 3),
            "video_kv_cache": cache,
        }

    def test_teacher_forcing_mask_matches_fact_relations(self) -> None:
        layout = FACTTokenLayout(2, 3, 4, 4, 1, 1, 2)
        mask = build_fact_teacher_forcing_mask(layout)
        allowed = torch.isfinite(mask)
        p, a, g, sv, total = (
            layout.prefix_end,
            layout.pred_action_end,
            layout.clean_action_end,
            layout.value_end,
            layout.total_tokens,
        )
        self.assertTrue(allowed[:p, :p].all())
        self.assertFalse(allowed[:p, p:].any())
        self.assertTrue(allowed[p:a, :a].all())
        self.assertFalse(allowed[p:a, a:].any())
        self.assertTrue(allowed[a:g, :p].all())
        self.assertFalse(allowed[a:g, p:a].any())
        self.assertTrue(allowed[a:g, a:g].all())
        self.assertFalse(allowed[a:g, g:].any())
        self.assertTrue(allowed[g:sv, :p].all())
        self.assertFalse(allowed[g:sv, p:a].any())
        self.assertTrue(allowed[g:sv, a:sv].all())
        self.assertFalse(allowed[g:sv, sv:].any())
        self.assertTrue(allowed[sv:total, :p].all())
        self.assertFalse(allowed[sv:total, p:a].any())
        self.assertTrue(allowed[sv:total, a:].all())

    def test_mask_is_byte_semantic_parity_with_pinned_official_function(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[1]
            / "third_party/FACT/world_action_model/models/transformer_wa_casual.py"
        )
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef)
            and item.name == "_build_teacher_forcing_mask"
        )
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
        official = namespace["_build_teacher_forcing_mask"]
        layout = FACTTokenLayout(2, 3, 4, 4, 1, 1, 2)
        expected = official(
            seq_len=layout.total_tokens,
            num_state_tokens=layout.state_tokens,
            num_ref_tokens=layout.ref_tokens,
            num_pred_action_tokens=layout.pred_action_tokens,
            num_gt_action_tokens=layout.clean_action_tokens,
            num_future_targets_tokens=layout.future_state_tokens + layout.value_tokens,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        actual = build_fact_teacher_forcing_mask(layout)
        self.assertTrue(torch.equal(actual, expected))

    def test_zero_residual_preserves_d0_stage1_exactly(self) -> None:
        model = self.build_model().eval()
        item = self.inputs()
        with torch.no_grad():
            parent = model.carrier(
                item["noisy_actions"],
                item["timestep"],
                text_context=item["text_context"],
                text_mask=item["text_mask"],
                proprio=item["proprio"],
                video_kv_cache=item["video_kv_cache"],
            )
            port = model.forward_action(
                item["noisy_actions"],
                item["timestep"],
                text_context=item["text_context"],
                text_mask=item["text_mask"],
                proprio=item["proprio"],
                video_kv_cache=item["video_kv_cache"],
            )
        torch.testing.assert_close(port, parent, rtol=0, atol=0)

    def test_joint_action_is_causally_invariant_to_clean_and_future_tracks(self) -> None:
        model = self.build_model().eval()
        torch.nn.init.normal_(model.action_residual_decoder.weight, std=0.1)
        item = self.inputs()
        with torch.no_grad():
            expected = model.forward_action(
                item["noisy_actions"],
                item["timestep"],
                text_context=item["text_context"],
                text_mask=item["text_mask"],
                proprio=item["proprio"],
                video_kv_cache=item["video_kv_cache"],
            )
            first = model(**item)["action"]
            changed = dict(item)
            changed["clean_actions"] = item["clean_actions"] + 100.0
            changed["noisy_future_state"] = item["noisy_future_state"] - 50.0
            changed["noisy_value"] = item["noisy_value"] + 25.0
            changed["noisy_future_representation"] = (
                item["noisy_future_representation"] * -20.0
            )
            second = model(**changed)["action"]
        torch.testing.assert_close(first, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(second, expected, rtol=1e-6, atol=1e-6)

    def test_future_loss_reaches_clean_action_and_shared_carrier_not_pred_action(self) -> None:
        model = self.build_model().train()
        item = self.inputs()
        noisy = item["noisy_actions"].detach().requires_grad_(True)
        clean = item["clean_actions"].detach().requires_grad_(True)
        output = model(**{**item, "noisy_actions": noisy, "clean_actions": clean})
        future_loss = (
            output["future_state"].square().mean()
            + output["value"].square().mean()
            + output["future_representation"].square().mean()
        )
        future_loss.backward()
        if noisy.grad is not None:
            self.assertEqual(float(noisy.grad.abs().max()), 0.0)
        self.assertIsNotNone(clean.grad)
        self.assertGreater(float(clean.grad.abs().sum()), 0.0)
        shared_grad = model.causal_backbone.layers[0].self_attn.in_proj_weight.grad
        self.assertIsNotNone(shared_grad)
        self.assertGreater(float(shared_grad.abs().sum()), 0.0)
        carrier_grad = model.carrier.action_expert.blocks[0].self_attn.q.weight.grad
        self.assertIsNotNone(carrier_grad)
        self.assertGreater(float(carrier_grad.abs().sum()), 0.0)

    def test_failure_action_mask_does_not_mask_valid_consequence_losses(self) -> None:
        action_target = torch.zeros(2, 4, 2)
        future_state = torch.zeros(2, 1, 3)
        value = torch.zeros(2, 1, 1)
        representation = torch.zeros(2, 1, 5)
        predictions = {
            "action": torch.zeros_like(action_target),
            "future_state": torch.ones_like(future_state),
            "value": torch.ones_like(value),
            "future_representation": torch.ones_like(representation),
        }
        predictions["action"][1] = 1000.0
        losses = fact_backbone_port_losses(
            predictions,
            action_target=action_target,
            future_state_target=future_state,
            value_target=value,
            future_representation_target=representation,
            action_is_pad=torch.zeros(2, 4, dtype=torch.bool),
            action_loss_mask=torch.tensor([1.0, 0.0]),
        )
        self.assertEqual(float(losses["action_loss"]), 0.0)
        self.assertEqual(float(losses["future_state_loss"]), 1.0)
        self.assertEqual(float(losses["value_loss"]), 1.0)
        self.assertEqual(float(losses["future_representation_loss"]), 1.0)
        self.assertAlmostEqual(float(losses["loss"]), 1.8, places=6)

    def test_c59_loader_keeps_base_value_but_rejects_fabricated_penalty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "c48.pt"
            source.write_bytes(b"immutable-c48")
            failure_rows = [
                {
                    "episode_index": 7,
                    "failure_episode": True,
                    "failure_active_from_frame": 1,
                    "failure_onset_available": False,
                }
            ]
            labels = [
                {
                    "sample_id": 11,
                    "episode_id": 7,
                    "failure_episode_mask": 1,
                    "failure_active_mask": 0,
                    "action_loss_mask": 0,
                    "future_loss_mask": 1,
                    "fact_code_value_raw": 0.5,
                    "fact_paper_progress_target": 0.5,
                    "source_c48_value_target": 1.5,
                }
            ]
            failure_path = root / "failure_rollouts.jsonl"
            labels_path = root / "sample_labels.jsonl"
            failure_path.write_text(
                "".join(json.dumps(row) + "\n" for row in failure_rows),
                encoding="utf-8",
            )
            labels_path.write_text(
                "".join(json.dumps(row) + "\n" for row in labels),
                encoding="utf-8",
            )

            def sha(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            report = {
                "format": "h3wam-c59-fact-failure-active-overlay-v1",
                "status": "PASS_C59_FAILURE_OVERLAY_CONTRACT",
                "source_dataset_sha256": sha(source),
                "samples": 1,
                "files": {
                    "sample_labels": {"path": str(labels_path), "sha256": sha(labels_path)},
                    "failure_rollouts": {
                        "path": str(failure_path),
                        "sha256": sha(failure_path),
                    },
                },
            }
            (root / "COMPLETED.json").write_text(json.dumps(report), encoding="utf-8")
            overlay = C59FailureOverlay(
                root,
                source_dataset=source,
                value_contract="fact_code_remaining_plus_penalty",
            )
            row = overlay.for_sample(11)
            self.assertEqual(row["action_loss_mask"], 0)
            self.assertEqual(row["future_loss_mask"], 1)
            self.assertEqual(row["value_loss_mask"], 1)
            self.assertEqual(row["value_target"], 0.5)
            labels[0]["fact_code_value_raw"] = 1.5
            labels_path.write_text(json.dumps(labels[0]) + "\n", encoding="utf-8")
            report["files"]["sample_labels"]["sha256"] = sha(labels_path)
            (root / "COMPLETED.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fabricated"):
                C59FailureOverlay(
                    root,
                    source_dataset=source,
                    value_contract="fact_code_remaining_plus_penalty",
                )

    def test_c60_loader_requires_active_causal_intervention_and_disjoint_parents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c60.pt"
            episodes = [
                {
                    "episode_index": 0,
                    "split": "train",
                    "source_ordinal": 2,
                    "failure_episode": True,
                    "failure_active_from_frame": 0,
                    "annotation_source": "state_aligned_counterfactual_action_intervention",
                },
                {
                    "episode_index": 1,
                    "split": "validation",
                    "source_ordinal": 5,
                    "failure_episode": True,
                    "failure_active_from_frame": 0,
                    "annotation_source": "state_aligned_counterfactual_action_intervention",
                },
            ]

            def row(sample_id: int, episode_id: int, split: str) -> dict:
                return {
                    "sample_id": sample_id,
                    "episode_id": episode_id,
                    "split": split,
                    "success": False,
                    "action_loss_mask": 0.0,
                    "future_loss_mask": 1.0,
                    "value_loss_mask": 1.0,
                    "failure_active": True,
                    "current_step": 10,
                    "failure_active_from_step": 10,
                    "fact_code_value_raw": 1.5,
                    "fact_paper_progress_target": 0.0,
                }

            payload = {
                "format": "h3wam-c60-counterfactual-failure-dataset-v1",
                "action_contract": "all branch actions masked from imitation",
                "episodes": episodes,
                "samples": [row(0, 0, "train"), row(1, 1, "validation")],
                "counts": {
                    "train": {"samples": 1},
                    "validation": {"samples": 1},
                },
            }
            torch.save(payload, path)
            labels = C60CausalFailureLabels(
                path, value_contract="fact_code_remaining_plus_penalty"
            )
            self.assertEqual(len(labels.split("train")), 1)
            self.assertEqual(labels.target_for(labels.split("train")[0])["value_target"], 1.5)
            payload["episodes"][1]["source_ordinal"] = 2
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "leakage"):
                C60CausalFailureLabels(
                    path, value_contract="fact_code_remaining_plus_penalty"
                )

    def test_strict_save_restore_is_identical(self) -> None:
        model = self.build_model().eval()
        item = self.inputs(batch=1)
        with torch.no_grad():
            expected = model(**item)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c56.pt"
            torch.save({"model": model.state_dict()}, path)
            restored = self.build_model().eval()
            payload = torch.load(path, map_location="cpu", weights_only=True)
            restored.load_state_dict(payload["model"], strict=True)
            with torch.no_grad():
                actual = restored(**item)
        for key in ("action", "future_state", "value", "future_representation"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
