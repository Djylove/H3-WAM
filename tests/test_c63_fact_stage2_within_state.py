from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    path = ROOT / "scripts" / "h3wam" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PAIRS = load("_test_c63_pairs", "prepare_c63_fact_stage2_pairs.py")
AGG = load("_test_c63_aggregate", "aggregate_c63_fact_stage2_within_state.py")


def test_executed_chunk_splices_between_replans_and_pads():
    actions = np.zeros((3, 32, 7), dtype=np.float32)
    actions[0, :, 0] = 1
    actions[1, :, 0] = 2
    actions[2, :, 0] = 3
    archive = {
        "step": np.array([0, 3, 8]),
        "terminal_step": np.array(10),
        "policy_actions": actions,
    }
    chunk, pad = PAIRS.executed_chunk(archive, 0)
    assert chunk.shape == (32, 7)
    assert chunk[:3, 0].tolist() == [1, 1, 1]
    assert chunk[3:8, 0].tolist() == [2, 2, 2, 2, 2]
    assert chunk[8:10, 0].tolist() == [3, 3]
    assert pad[:10].tolist() == [False] * 10
    assert pad[10:].tolist() == [True] * 22


def test_stage2_value_is_exactly_one_token_per_candidate():
    state = torch.zeros(2, 1, 8)
    value = torch.tensor([[[0.25]], [[0.75]]])
    representation = torch.zeros(2, 1, 56 * 128)
    selected = PAIRS.assert_stage2_track_shapes(
        state, value, representation, batch=2, future_dim=56 * 128
    )
    assert selected.tolist() == [0.25, 0.75]
    with pytest.raises(ValueError, match="token shape mismatch"):
        PAIRS.assert_stage2_track_shapes(
            state, torch.zeros(2, 2, 1), representation,
            batch=2, future_dim=56 * 128,
        )


def rows(success_spatial: int, success_object: int):
    result = []
    for index in range(32):
        suite = "libero_object" if index < 2 else "libero_spatial"
        within = index if suite == "libero_object" else index - 2
        preferred = within < (success_object if suite == "libero_object" else success_spatial)
        margin = 1.0 if preferred else -1.0
        result.append(
            {
                "pair_index": index,
                "suite": suite,
                "success_preferred": preferred,
                "failure_minus_success": margin,
                "score_finite": True,
                "action_conditioned_value_delta_nonzero": True,
                "order_invariance_pass": True,
            }
        )
    return result


def test_aggregate_exact_frozen_pass_boundary():
    result = AGG.aggregate_rows(rows(success_spatial=20, success_object=2))
    assert result["success_preferred"] == 22
    assert result["one_sided_exact_binomial_p"] == 0.025051229866221547
    assert result["status"] == "PASS_C63_STAGE2_WITHIN_STATE_DIAGNOSTIC"
    assert result["permission"] == "GO_COLLECT_CROSS_SUITE_C63_CONFIRMATORY_PAIRS"


def test_aggregate_rejects_primary_or_suite_shortfall():
    primary = AGG.aggregate_rows(rows(success_spatial=19, success_object=2))
    assert primary["status"] == "FAIL_C63_STAGE2_WITHIN_STATE_DIAGNOSTIC"
    assert not primary["gates"]["primary_at_least_22_of_32"]
    spatial = AGG.aggregate_rows(rows(success_spatial=19, success_object=2))
    assert not spatial["gates"]["spatial_at_least_20_of_30"]


def test_aggregate_rejects_mechanical_failure():
    values = rows(success_spatial=20, success_object=2)
    values[0]["order_invariance_pass"] = False
    result = AGG.aggregate_rows(values)
    assert result["status"] == "FAIL_C63_STAGE2_WITHIN_STATE_DIAGNOSTIC"
    assert not result["gates"]["mechanics"]
    assert "FAIL_C63_STAGE2_SHARD_MECHANICS" in AGG.SHARD_STATUSES
