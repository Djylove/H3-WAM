#!/usr/bin/env python3
"""Fixed balanced-80 evaluator for the C71 direct-action Light-WAM H3 port."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam.c58_online_training import (  # noqa: E402
    C58OnlineFrozenH3Provider,
)
from fastwam.models.h3wam.lightwam_state_fusion import (  # noqa: E402
    LIGHTWAM_COMMIT,
    LIGHTWAM_H3_CARRIER_LAYERS,
    LIGHTWAM_STATE_FUSION_SHA256,
    H3LightWAMStateFusionPolicy,
)


def _load_sibling(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_sibling("_c71_balanced80_base", "evaluate_h3_fastwam_full_tower_online.py")
TRAINER = _load_sibling("_c71_checkpoint_contract", "train_c71_lightwam_online.py")
PROTOCOL = BASE.PROTOCOL

FORMAT = "h3wam-c71-lightwam-online-balanced80-v1"
EXPECTED_SELECTED_IDS_SHA256 = BASE.EXPECTED_SELECTED_IDS_SHA256
EXPECTED_H3_SHA256 = TRAINER.EXPECTED_H3_SHA256
EXPECTED_ITEMS = 80
EVALUATION_MILESTONES = frozenset({1000, 5000, 10000})
LAYERS = tuple(LIGHTWAM_H3_CARRIER_LAYERS)


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: Path
    restore_report: Path
    h3_checkpoint: Path
    source_manifest: Path
    train_manifest: Path
    val_manifest: Path
    cache_root: Path
    output: Path
    device: str = "cuda:0"
    num_workers: int = 0
    seed: int = 42
    batch_size: int = 1
    samples_per_task: int = 2
    expected_steps: int = 1000


class C71ValidationDataset(BASE.OnlineC58bValidationDataset):
    """Add replacement H3 token tags to the fixed C58b balanced-80 rows."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = super().__getitem__(index)
        row = self.rows[index]
        context_id = str(row["context_id"])
        replacement_id = self.replacement_context[context_id]
        _, replacement_tags = BASE._load_context(self.cache_root, replacement_id)
        result["replacement_text_token_tags"] = replacement_tags
        return result


def collate_c71(items: list[dict[str, Any]]) -> dict[str, Any]:
    result = BASE.collate_online(items)
    if len(items) != 1:
        raise ValueError("C71 online evaluation requires batch size 1")
    result["replacement_text_token_tags"] = items[0][
        "replacement_text_token_tags"
    ].unsqueeze(0)
    return result


def move_c71_batch(
    batch: dict[str, Any], device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    result = BASE.move_batch(batch, device, dtype)
    result["replacement_text_token_tags"] = batch[
        "replacement_text_token_tags"
    ].to(device=device, dtype=torch.long)
    return result


def provider_input(batch: dict[str, Any], mode: str) -> dict[str, torch.Tensor]:
    if mode == "baseline":
        return {
            "current_h3_input": batch["current_h3_input"],
            "text_context": batch["text_context"],
            "text_token_tags": batch["text_token_tags"],
        }
    if mode == "visual_shuffle":
        return {
            "current_h3_input": batch["shuffled_h3_input"],
            "text_context": batch["shuffled_h3_text_context"],
            "text_token_tags": batch["shuffled_h3_text_token_tags"],
        }
    if mode == "replacement_language":
        return {
            "current_h3_input": batch["current_h3_input"],
            "text_context": batch["replacement_text_context"],
            "text_token_tags": batch["replacement_text_token_tags"],
        }
    raise ValueError(f"unsupported C71 provider mode: {mode}")


def restore_model(
    model_state: dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    model = H3LightWAMStateFusionPolicy(enabled=True).to(device=device, dtype=dtype)
    model.load_state_dict(model_state, strict=True)
    return model.eval()


@torch.no_grad()
def predict_direct(
    model: nn.Module,
    batch: dict[str, Any],
    cache: dict[int, dict[str, torch.Tensor]],
) -> torch.Tensor:
    return model(
        torch.zeros_like(batch["actions"]),
        torch.zeros((batch["actions"].shape[0],), device=batch["actions"].device, dtype=batch["actions"].dtype),
        text_context=batch["text_context"],
        text_mask=batch["text_mask"],
        proprio=batch["proprio"],
        video_kv_cache=cache,
    )


def _load_checkpoint_and_restore_gate(
    config: EvalConfig,
    *,
    source_sha: str,
    train_sha: str,
    stats_sha: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    restore = json.loads(config.restore_report.read_text(encoding="utf-8"))
    if (
        restore.get("status") != "PASS_C71_STRICT_RESTORE"
        or restore.get("completed_steps") != config.expected_steps
        or restore.get("restore_probe_max_abs") != 0.0
        or restore.get("steps_this_invocation") != 0
        or restore.get("training_samples") != 0
    ):
        raise ValueError("C71 step-1000 independent restore gate failed")
    checkpoint = config.checkpoint.resolve()
    checkpoint_sha = PROTOCOL.sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if set(payload) != set(TRAINER.CHECKPOINT_KEYS):
        raise ValueError("C71 checkpoint top-level schema mismatch")
    if payload.get("schema_version") != TRAINER.CHECKPOINT_SCHEMA:
        raise ValueError("C71 checkpoint schema mismatch")
    if payload.get("completed_steps") != config.expected_steps:
        raise ValueError("C71 balanced80 checkpoint milestone mismatch")
    contract = payload.get("contract")
    required = {
        "classification": "backbone_port",
        "parent": "C58 frozen online INT8 H3 dense action boundary",
        "unique_variable": "byte-pinned Light-WAM three-state direct-regression expert",
        "lightwam_commit": LIGHTWAM_COMMIT,
        "lightwam_state_fusion_sha256": LIGHTWAM_STATE_FUSION_SHA256,
        "h3_checkpoint_sha256": EXPECTED_H3_SHA256,
        "h3_trainable": False,
        "h3_layers": list(LAYERS),
        "feature": "value_state",
        "capture_tokens": 32,
        "action_horizon": 32,
        "action_dim": 7,
        "proprio_dim": 8,
        "objective": "uniform_full_horizon_masked_normalized_action_mse",
        "manifest_sha256": train_sha,
        "source_manifest_sha256": source_sha,
        "stats_sha256": stats_sha,
    }
    if not isinstance(contract, dict):
        raise ValueError("C71 checkpoint contract missing")
    mismatches = {
        key: {"actual": contract.get(key), "expected": value}
        for key, value in required.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"C71 evaluation contract mismatch: {mismatches}")
    if PROTOCOL.sha256_file(config.h3_checkpoint) != EXPECTED_H3_SHA256:
        raise ValueError("C71 H3 checkpoint SHA mismatch")
    return payload, checkpoint_sha, restore


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    started = time.perf_counter()
    if (
        config.batch_size != 1
        or config.samples_per_task != 2
        or config.seed != 42
        or config.num_workers != 0
        or config.expected_steps not in EVALUATION_MILESTONES
    ):
        raise ValueError("C71 balanced80 protocol is fixed")
    source_rows = PROTOCOL.read_jsonl(config.source_manifest)
    train_rows = PROTOCOL.read_jsonl(config.train_manifest)
    val_rows = PROTOCOL.read_jsonl(config.val_manifest)
    split_audit = PROTOCOL.validate_episode_disjoint_manifests(
        source_rows, train_rows, val_rows
    )
    selected, selection = PROTOCOL.select_validation_rows(
        val_rows, samples_per_task=2
    )
    if (
        len(selected) != EXPECTED_ITEMS
        or selection["selected_ids_sha256"] != EXPECTED_SELECTED_IDS_SHA256
    ):
        raise ValueError("C71 balanced80 selected-ID gate mismatch")
    visual_mapping, visual_contract = PROTOCOL.build_visual_feature_shuffle(selected)
    hashes = {
        "source_manifest_sha256": PROTOCOL.sha256_file(config.source_manifest),
        "train_manifest_sha256": PROTOCOL.sha256_file(config.train_manifest),
        "validation_manifest_sha256": PROTOCOL.sha256_file(config.val_manifest),
        "stats_sha256": PROTOCOL.sha256_file(config.cache_root / "stats.pt"),
    }
    payload, checkpoint_sha, restore = _load_checkpoint_and_restore_gate(
        config,
        source_sha=hashes["source_manifest_sha256"],
        train_sha=hashes["train_manifest_sha256"],
        stats_sha=hashes["stats_sha256"],
    )
    dataset = C71ValidationDataset(
        selected,
        cache_root=config.cache_root,
        action_horizon=32,
        visual_mapping=visual_mapping,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_c71,
    )
    device, dtype = PROTOCOL._resolve_device_dtype(config.device)
    if device.type != "cuda":
        raise ValueError("C71 online balanced80 requires CUDA")
    provider = C58OnlineFrozenH3Provider(config.h3_checkpoint, layers=LAYERS).to(
        device=device
    ).eval()
    first_batch = move_c71_batch(next(iter(loader)), device, dtype)
    first_cache = provider(provider_input(first_batch, "baseline"))
    predictions: list[torch.Tensor] = []
    restored_model: nn.Module | None = None
    for _ in range(2):
        restored_model = restore_model(payload["model"], device=device, dtype=dtype)
        predictions.append(
            predict_direct(restored_model, first_batch, first_cache).float().cpu()
        )
        gc.collect()
        torch.cuda.empty_cache()
    fresh_restore_max_abs = float((predictions[0] - predictions[1]).abs().max())
    if fresh_restore_max_abs != 0.0 or restored_model is None:
        raise RuntimeError(f"C71 fresh evaluator restore mismatch: {fresh_restore_max_abs}")
    model = restored_model

    normalized = PROTOCOL.DomainMetricAccumulator(7)
    physical = PROTOCOL.DomainMetricAccumulator(7)
    gripper = PROTOCOL.GripperSignAccumulator(6)
    language = PROTOCOL.LanguageSensitivityAccumulator()
    shuffled_normalized = PROTOCOL.DomainMetricAccumulator(7)
    shuffled_physical = PROTOCOL.DomainMetricAccumulator(7)
    shuffled_gripper = PROTOCOL.GripperSignAccumulator(6)
    normalized_delta = PROTOCOL.DomainMetricAccumulator(7)
    physical_delta = PROTOCOL.DomainMetricAccumulator(7)
    evaluated_ids: list[str] = []
    evaluated_pairs: list[str] = []
    evaluated_tasks: Counter[str] = Counter()
    per_sample_seconds: list[float] = []
    for cpu_batch in loader:
        sample_started = time.perf_counter()
        batch = move_c71_batch(cpu_batch, device, dtype)
        baseline_cache = provider(provider_input(batch, "baseline"))
        shuffled_cache = provider(provider_input(batch, "visual_shuffle"))
        replacement_cache = provider(provider_input(batch, "replacement_language"))
        prediction = predict_direct(model, batch, baseline_cache)
        shuffled = predict_direct(model, batch, shuffled_cache)
        replacement = predict_direct(model, batch, replacement_cache)
        action_min = dataset.action_min.to(device=device, dtype=dtype)
        action_max = dataset.action_max.to(device=device, dtype=dtype)
        prediction_physical = PROTOCOL.denormalize_minmax_official(
            prediction, action_min, action_max
        )
        shuffled_physical_value = PROTOCOL.denormalize_minmax_official(
            shuffled, action_min, action_max
        )
        pad = batch["action_is_pad"]
        normalized.update(prediction, batch["actions"], pad)
        physical.update(prediction_physical, batch["raw_actions"], pad)
        gripper.update(prediction, batch["actions"], pad)
        language.update(prediction, replacement, pad)
        shuffled_normalized.update(shuffled, batch["actions"], pad)
        shuffled_physical.update(shuffled_physical_value, batch["raw_actions"], pad)
        shuffled_gripper.update(shuffled, batch["actions"], pad)
        normalized_delta.update(shuffled, prediction, pad)
        physical_delta.update(shuffled_physical_value, prediction_physical, pad)
        evaluated_ids.extend(cpu_batch["sample_ids"])
        evaluated_tasks.update(cpu_batch["tasks"])
        evaluated_pairs.extend(
            f"{target}\0{source}"
            for target, source in zip(
                cpu_batch["sample_ids"],
                cpu_batch["visual_shuffle_source_ids"],
                strict=True,
            )
        )
        per_sample_seconds.append(time.perf_counter() - sample_started)

    ids_sha = PROTOCOL.sha256_strings(evaluated_ids)
    pairs_sha = PROTOCOL.sha256_strings(evaluated_pairs)
    if ids_sha != EXPECTED_SELECTED_IDS_SHA256:
        raise RuntimeError("C71 evaluated IDs drifted")
    if pairs_sha != visual_contract["ordered_mapping_sha256"]:
        raise RuntimeError("C71 visual shuffle mapping drifted")
    if dict(sorted(evaluated_tasks.items())) != selection["task_counts"]:
        raise RuntimeError("C71 evaluated task counts drifted")
    normalized_report = normalized.finalize()
    physical_report = physical.finalize()
    gripper_report = gripper.finalize()
    shuffled_normalized_report = shuffled_normalized.finalize()
    shuffled_physical_report = shuffled_physical.finalize()
    shuffled_gripper_report = shuffled_gripper.finalize()
    report = {
        "format": FORMAT,
        "event": "h3_c71_lightwam_direct_action_balanced80",
        "status": "completed_offline_not_closed_loop_evidence",
        "candidate": "C71_LIGHTWAM_THREE_STATE_DIRECT_ACTION",
        "effect_status": "NOT_EVIDENCE_READY",
        "checkpoint": {
            "path": str(config.checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "completed_steps": payload["completed_steps"],
            "independent_restore_report": str(config.restore_report.resolve()),
            "independent_restore_report_sha256": PROTOCOL.sha256_file(config.restore_report),
            "independent_restore_max_abs": restore["restore_probe_max_abs"],
            "fresh_evaluator_restore_max_abs": fresh_restore_max_abs,
        },
        "contract": payload["contract"],
        "execution": {
            "h3": "online_frozen_int8",
            "disk_kv_read": False,
            "disk_feature_read": False,
            "carrier_layers": list(LAYERS),
            "action_decode": "single_pass_direct_regression",
            "visual_shuffle": "rerun_online_h3_on_source_latent_and_source_context",
            "language_replacement": "rerun_online_h3_on_target_latent_and_replacement_context",
        },
        "data": {
            **hashes,
            "source_manifest_items": len(source_rows),
            "train_manifest_items": len(train_rows),
            "validation_manifest_items": len(val_rows),
            "selected_sample_ids_sha256": ids_sha,
            "selection": selection,
            "split_audit": split_audit,
        },
        "inference": {
            "steps": 1,
            "sampler": "none_direct_regression",
            "seed": 42,
            "batch_size": 1,
            "device": str(device),
            "dtype": str(dtype),
        },
        "metrics": {
            "normalized_clip5_model_domain": normalized_report,
            "denormalized_official_minmax_clamp": physical_report,
            "gripper_sign": gripper_report,
            "language_replacement_sensitivity": language.finalize(),
            "visual_feature_shuffle": {
                "contract": visual_contract,
                "evaluated_mapping_sha256": pairs_sha,
                "baseline_vs_shuffle_action_delta": {
                    "normalized_model_domain": normalized_delta.finalize(),
                    "denormalized_official_minmax_clamp": physical_delta.finalize(),
                },
                "shuffle_vs_target": {
                    "normalized_clip5_model_domain": shuffled_normalized_report,
                    "denormalized_official_minmax_clamp": shuffled_physical_report,
                    "gripper_sign": shuffled_gripper_report,
                },
                "metric_change_shuffle_minus_baseline": {
                    "normalized_clip5_model_domain": PROTOCOL.metric_changes(
                        normalized_report,
                        shuffled_normalized_report,
                        PROTOCOL.DOMAIN_CHANGE_KEYS,
                    ),
                    "denormalized_official_minmax_clamp": PROTOCOL.metric_changes(
                        physical_report,
                        shuffled_physical_report,
                        PROTOCOL.DOMAIN_CHANGE_KEYS,
                    ),
                    "gripper_sign": PROTOCOL.metric_changes(
                        gripper_report,
                        shuffled_gripper_report,
                        PROTOCOL.GRIPPER_CHANGE_KEYS,
                    ),
                },
            },
        },
        "timing": {
            "elapsed_seconds": time.perf_counter() - started,
            "mean_seconds_per_sample": sum(per_sample_seconds) / len(per_sample_seconds),
        },
        "claim_boundary": "Held-out regression and counterfactual sensitivity only; no LIBERO closed-loop claim.",
    }
    output = config.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return report


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--restore-report", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--expected-steps", type=int, default=1000)
    values = parser.parse_args()
    return EvalConfig(**vars(values))


if __name__ == "__main__":
    print(json.dumps(run_evaluation(parse_args()), indent=2), flush=True)
