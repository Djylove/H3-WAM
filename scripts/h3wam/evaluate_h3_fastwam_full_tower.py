#!/usr/bin/env python3
"""Evaluate C58 with the frozen balanced-80 DreamWAM/D0 protocol.

The metric, selection, cache, flow-sampling and perturbation implementation is
delegated to ``evaluate_h3_dreamwam_kv_carrier.py``.  This adapter replaces
only the checkpoint contract and model constructor needed by the official
FastWAM full-30 ActionDiT backbone port.  It intentionally does not loosen the
balanced-80 sample-ID gate or any inference setting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import types
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_sibling(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_sibling(
    "_c58_frozen_balanced80_evaluator", "evaluate_h3_dreamwam_kv_carrier.py"
)
C58 = _load_sibling(
    "_c58_fastwam_full_tower_trainer", "train_h3_fastwam_full_tower.py"
)

V7_SELECTED_IDS_SHA256 = (
    "26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42"
)
C58_MODEL_SPEC_KEYS = set(asdict(C58.ModelSpec()))


def _c58_facade() -> types.SimpleNamespace:
    """Expose C58 construction while retaining the audited D0 cache contract."""

    values = {
        name: value
        for name, value in vars(C58.PARENT).items()
        if not name.startswith("__")
    }
    values.update(
        {
            "CHECKPOINT_KEYS": C58.CHECKPOINT_KEYS,
            "CHECKPOINT_SCHEMA": C58.CHECKPOINT_SCHEMA,
            "ModelSpec": C58.ModelSpec,
            "build_model": C58.build_model,
        }
    )
    return types.SimpleNamespace(**values)


BASE.CANDIDATE_D = _c58_facade()
BASE.EXPECTED_H3_CHECKPOINT_SHA256 = C58.PARENT.H3_INT8_CHECKPOINT_SHA256
BASE.MODEL_SPEC_KEYS = C58_MODEL_SPEC_KEYS


def require_c58_contract(
    payload: dict[str, Any],
    *,
    config: Any,
    source_manifest_sha256: str,
    source_manifest_items: int,
    train_manifest_sha256: str,
    train_manifest_items: int,
    stats_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Reject any checkpoint whose training or H3 carrier identity drifts."""

    if set(payload) != set(C58.CHECKPOINT_KEYS):
        raise ValueError("C58 checkpoint top-level schema mismatch")
    if payload.get("schema_version") != C58.CHECKPOINT_SCHEMA:
        raise ValueError("C58 checkpoint schema mismatch")
    if not isinstance(payload.get("completed_steps"), int) or payload["completed_steps"] < 0:
        raise ValueError("C58 completed_steps must be a non-negative integer")
    if not isinstance(payload.get("model"), dict) or not payload["model"]:
        raise ValueError("C58 model state is missing or empty")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("C58 checkpoint contract is missing")
    required = {
        "candidate": "C58_FASTWAM_FULL30_H3_LAYER49",
        "classification": "action-only-on-frozen-features_backbone_port",
        "fastwam_commit": C58.FASTWAM_COMMIT,
        "fastwam_action_dit_sha256": C58.FASTWAM_ACTION_DIT_SHA256,
        "fastwam_video_dit_sha256": C58.FASTWAM_VIDEO_DIT_SHA256,
        "fastwam_gradient_sha256": C58.FASTWAM_GRADIENT_SHA256,
        "fastwam_mot_sha256": C58.FASTWAM_MOT_SHA256,
        "d0_parent_optimizer_restored": False,
        "carrier_source_mode": C58.PARENT.REPEAT_LAYER49_CARRIER_SOURCE,
        "h3_checkpoint_sha256": C58.PARENT.H3_INT8_CHECKPOINT_SHA256,
        "verify_h3_checkpoint_sha256": True,
        "kv_schema": C58.PARENT.DREAMWAM_KV_SCHEMA,
        "kv_strategy": C58.PARENT.DREAMWAM_KV_STRATEGY,
        "kv_layers": list(C58.DEFAULT_H3_CARRIER_LAYERS),
        "kv_tokens": 32,
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_items": source_manifest_items,
        "split_manifest_sha256": train_manifest_sha256,
        "split_manifest_items": train_manifest_items,
        "stats_sha256": stats_sha256,
        "action_normalization": "starwam_minmax_clip5",
        "state_normalization": "starwam_minmax_clip5",
        "action_horizon": 32,
    }
    mismatches = {
        key: {"checkpoint": contract.get(key), "expected": expected}
        for key, expected in required.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"C58 evaluation contract mismatch: {mismatches}")
    if not math.isclose(float(contract.get("action_shift", -1.0)), config.action_shift):
        raise ValueError("C58 action_shift differs from balanced-80 protocol")
    if not isinstance(contract.get("d0_parent_sha256"), str) or len(
        contract["d0_parent_sha256"]
    ) != 64:
        raise ValueError("C58 D0 parent identity is missing")
    if int(contract.get("d0_parent_completed_steps", -1)) != 14_000:
        raise ValueError("C58 must descend from the audited D0 step14000 parent")
    initialization = contract.get("initialization")
    initialization_required = {
        "source_layers": 5,
        "target_layers": 30,
        "anchor_target_indices": [0, 7, 14, 22, 29],
        "identity_target_indices": [
            1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19,
            20, 21, 23, 24, 25, 26, 27, 28,
        ],
        "alpha_scaling_applied": False,
        "width_interpolation_applied": False,
        "initialization_contract": "exact_d0_function_preserving_depth_expansion_v1",
    }
    if not isinstance(initialization, dict):
        raise ValueError("C58 initialization contract is missing")
    initialization_mismatches = {
        key: {"checkpoint": initialization.get(key), "expected": expected}
        for key, expected in initialization_required.items()
        if initialization.get(key) != expected
    }
    if initialization_mismatches:
        raise ValueError(
            f"C58 initialization contract mismatch: {initialization_mismatches}"
        )
    model_spec = contract.get("model_spec")
    if not isinstance(model_spec, dict) or set(model_spec) != C58_MODEL_SPEC_KEYS:
        raise ValueError("C58 model_spec schema mismatch")
    expected_spec = asdict(C58.ModelSpec())
    expected_spec["carrier_layers"] = list(expected_spec["carrier_layers"])
    normalized_spec = dict(model_spec)
    if isinstance(normalized_spec.get("carrier_layers"), tuple):
        normalized_spec["carrier_layers"] = list(normalized_spec["carrier_layers"])
    if normalized_spec != expected_spec:
        raise ValueError("C58 model_spec differs from the audited full-30 tower")
    if not str(contract.get("h3_checkpoint_path", "")):
        raise ValueError("C58 H3 checkpoint path is missing")
    kv_subdir = str(contract.get("kv_subdir", ""))
    if not kv_subdir:
        raise ValueError("C58 kv_subdir is missing")
    if config.kv_subdir is not None and config.kv_subdir != kv_subdir:
        raise ValueError("requested kv_subdir differs from C58 checkpoint")
    return contract, model_spec, kv_subdir


BASE._require_contract = require_c58_contract


def run_evaluation(config: Any) -> dict[str, Any]:
    """Run the unchanged balanced-80 protocol and relabel only its report."""

    report = BASE.run_evaluation(replace(config, output=None))
    report.update(
        {
            "event": "h3_c58_fastwam_full30_balanced80_offline_evaluation",
            "candidate": "C58_FASTWAM_FULL30_H3_LAYER49",
            "classification": "fastwam-full30-action-on-frozen-h3-layer49-kv",
            "status": "completed_not_closed_loop_evidence",
        }
    )
    report["protocol_identity"].update(
        {
            "adapter": str(Path(__file__).resolve()),
            "architecture_source": "official FastWAM ActionDiT full30",
            "fastwam_commit": C58.FASTWAM_COMMIT,
            "fastwam_action_dit_sha256": C58.FASTWAM_ACTION_DIT_SHA256,
            "evidence_boundary": (
                "offline held-out action/condition-response diagnostic; "
                "not LIBERO closed-loop success evidence"
            ),
        }
    )
    if config.output is not None:
        output = config.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--kv-subdir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-audit-aggregate-sha256")
    parser.add_argument(
        "--expected-selected-ids-sha256", default=V7_SELECTED_IDS_SHA256
    )
    values = parser.parse_args()
    return BASE.EvalConfig(
        checkpoint=values.checkpoint,
        source_manifest=values.source_manifest,
        train_manifest=values.train_manifest,
        val_manifest=values.val_manifest,
        cache_root=values.cache_root,
        kv_subdir=values.kv_subdir,
        output=values.output,
        device=values.device,
        num_workers=values.num_workers,
        cache_audit_aggregate_sha256=values.cache_audit_aggregate_sha256,
        expected_selected_ids_sha256=values.expected_selected_ids_sha256,
    )


if __name__ == "__main__":
    run_evaluation(parse_args())
