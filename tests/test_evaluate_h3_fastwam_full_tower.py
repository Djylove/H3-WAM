from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/h3wam/evaluate_h3_fastwam_full_tower.py"
    spec = importlib.util.spec_from_file_location("_test_c58_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def valid_payload(module, *, matched_control=False):
    spec = asdict(module.C58.ModelSpec())
    if matched_control:
        spec["action_layers"] = 5
    parent = "a" * 64
    initialization = {
        "source_layers": 5,
        "target_layers": 5 if matched_control else 30,
        "anchor_target_indices": (
            [0, 1, 2, 3, 4]
            if matched_control
            else [0, 7, 14, 22, 29]
        ),
        "nearest_source_indices": [
            0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
            2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
        ],
        "identity_target_indices": (
            []
            if matched_control
            else [
                1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19,
                20, 21, 23, 24, 25, 26, 27, 28,
            ]
        ),
        "source_prefix": "blocks",
        "target_prefix": "action_expert.blocks",
        "alpha_scaling_applied": False,
        "width_interpolation_applied": False,
        "initialization_contract": (
            "exact_d0_weights_fresh_optimizer_v1"
            if matched_control
            else "exact_d0_function_preserving_depth_expansion_v1"
        ),
    }
    contract = {
        "candidate": (
            "C58_MATCHED_D0_FRESH_OPTIMIZER"
            if matched_control
            else "C58_FASTWAM_FULL30_H3_LAYER49"
        ),
        "classification": (
            "matched-five-layer-depth-control_fresh-optimizer"
            if matched_control
            else "action-only-on-frozen-features_backbone_port"
        ),
        "fastwam_commit": module.C58.FASTWAM_COMMIT,
        "fastwam_action_dit_sha256": module.C58.FASTWAM_ACTION_DIT_SHA256,
        "fastwam_video_dit_sha256": module.C58.FASTWAM_VIDEO_DIT_SHA256,
        "fastwam_gradient_sha256": module.C58.FASTWAM_GRADIENT_SHA256,
        "fastwam_mot_sha256": module.C58.FASTWAM_MOT_SHA256,
        "d0_parent_sha256": parent,
        "d0_parent_completed_steps": 14_000,
        "d0_parent_optimizer_restored": False,
        "initialization": initialization,
        "carrier_source_mode": module.C58.PARENT.REPEAT_LAYER49_CARRIER_SOURCE,
        "h3_checkpoint_path": "/checkpoint/H3_int8.safetensors",
        "h3_checkpoint_sha256": module.C58.PARENT.H3_INT8_CHECKPOINT_SHA256,
        "verify_h3_checkpoint_sha256": True,
        "kv_subdir": "kv",
        "kv_schema": module.C58.PARENT.DREAMWAM_KV_SCHEMA,
        "kv_strategy": module.C58.PARENT.DREAMWAM_KV_STRATEGY,
        "kv_layers": list(module.C58.DEFAULT_H3_CARRIER_LAYERS),
        "kv_tokens": 32,
        "source_manifest_sha256": "s",
        "source_manifest_items": 10,
        "split_manifest_sha256": "t",
        "split_manifest_items": 8,
        "stats_sha256": "n",
        "action_normalization": "starwam_minmax_clip5",
        "state_normalization": "starwam_minmax_clip5",
        "action_horizon": 32,
        "action_shift": 5.0,
        "model_spec": spec,
    }
    payload = {key: None for key in module.C58.CHECKPOINT_KEYS}
    payload.update(
        schema_version=module.C58.CHECKPOINT_SCHEMA,
        completed_steps=10,
        model={"parameter": 1},
        contract=contract,
    )
    return payload


def require(module, payload):
    return module.require_c58_contract(
        payload,
        config=SimpleNamespace(action_shift=5.0, kv_subdir=None),
        source_manifest_sha256="s",
        source_manifest_items=10,
        train_manifest_sha256="t",
        train_manifest_items=8,
        stats_sha256="n",
    )


def test_contract_accepts_only_full30_function_preserving_port():
    module = load_module()
    contract, model_spec, kv_subdir = require(module, valid_payload(module))
    assert contract["candidate"] == "C58_FASTWAM_FULL30_H3_LAYER49"
    assert model_spec["action_layers"] == 30
    assert kv_subdir == "kv"


def test_contract_accepts_only_strict_matched_five_layer_control():
    module = load_module()
    contract, model_spec, kv_subdir = require(
        module, valid_payload(module, matched_control=True)
    )
    assert contract["candidate"] == "C58_MATCHED_D0_FRESH_OPTIMIZER"
    assert model_spec["action_layers"] == 5
    assert contract["d0_parent_optimizer_restored"] is False
    assert kv_subdir == "kv"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("contract", "fastwam_commit"), "wrong"),
        (("contract", "model_spec", "action_layers"), 5),
        (("contract", "initialization", "target_layers"), 5),
        (("contract", "verify_h3_checkpoint_sha256"), False),
    ],
)
def test_contract_rejects_identity_or_architecture_drift(path, value):
    module = load_module()
    payload = valid_payload(module)
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError):
        require(module, payload)
