from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h3wam/audit_h3_starwam_feature_cache.py"
SPEC = importlib.util.spec_from_file_location("audit_h3_starwam_feature_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        capture_token_count=32,
        feature_dim=5376,
        action_horizon=32,
        timestep=1.0,
        condition_video_timestep=1.0,
        expected_checkpoint=Path("/tmp/h3.safetensors"),
        producer_num_shards=32,
    )


def _row() -> dict:
    return {
        "id": "libero_goal_ep000001_s000002",
        "episode": 1,
        "start": 2,
        "suite": "libero_goal",
        "context_id": "task_abc",
    }


def _payload() -> dict:
    return {
        "features": torch.zeros(1, 32, 5376, dtype=torch.bfloat16),
        "layers": (49,),
        "capture_token_count": 32,
        "capture_token_strategy": "starwam_adaptive_avg_pool1d_v1",
        "capture_compatibility": "none",
        "episode": 1,
        "start": 2,
        "suite": "libero_goal",
        "context_id": "task_abc",
        "context_width": 5120,
        "context_mode": "raw_qwen",
        "timestep": 1.0,
        "condition_video_timestep": 1.0,
        "action_horizon": 32,
        "backbone": "H3Int8FeatureBackbone",
        "quantization": "int8_tensorwise_convrot",
        "checkpoint": "/tmp/h3.safetensors",
        "manifest_items": 222929,
        "num_shards": 32,
        "shard_index": 2,
    }


def test_accepts_real_contract_shape_and_metadata() -> None:
    assert MODULE.audit_payload(
        _payload(), _row(), row_index=2, manifest_items=222929, args=_args()
    ) == []


def test_rejects_wrong_shard_shape_dtype_and_nonfinite() -> None:
    payload = _payload()
    payload["shard_index"] = 3
    payload["features"] = torch.zeros(1, 31, 5376, dtype=torch.float32)
    payload["features"][0, 0, 0] = float("nan")
    errors = MODULE.audit_payload(
        payload, _row(), row_index=2, manifest_items=222929, args=_args()
    )
    assert any("shard_index" in error for error in errors)
    assert any("features:shape" in error for error in errors)
    assert any("features:dtype" in error for error in errors)
    assert any("features:nonfinite" in error for error in errors)


def test_rejects_unknown_context_contract() -> None:
    payload = _payload()
    payload["context_width"] = 4096
    assert any(
        "context_contract" in error
        for error in MODULE.audit_payload(
            payload, _row(), row_index=2, manifest_items=222929, args=_args()
        )
    )
