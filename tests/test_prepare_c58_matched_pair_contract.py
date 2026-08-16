from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/h3wam/prepare_c58_matched_pair_contract.py"
    spec = importlib.util.spec_from_file_location("_test_c58_pair_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pair_plan_freezes_exact_unique_80k_rank_and_step_order():
    module = load_module()
    rows = [{"id": f"sample_{index:06d}"} for index in range(200_779)]
    first = module.build_pair_plan(rows)
    second = module.build_pair_plan(rows)
    assert first == second
    assert first["common_contract"]["training_samples"] == 80_000
    assert first["common_contract"]["optimizer_steps"] == 10_000
    assert len(first["stages"]) == 10
    assert [stage["sample_offset"] for stage in first["stages"]] == [
        112_000 + 8_000 * index for index in range(10)
    ]
    assert all(len(stage["rank_order_sha256"]) == 8 for stage in first["stages"])
    assert len(first["combined_selected_manifest_order_sha256"]) == 64
