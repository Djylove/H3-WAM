from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/h3wam"))
SPEC = importlib.util.spec_from_file_location(
    "c40_ranker_test_module",
    ROOT / "scripts/h3wam/train_c40_fresh_consequence_value_ranker.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_source_fold_is_stable_and_source_locked():
    assert MODULE.source_fold("same-source") == MODULE.source_fold("same-source")
    assert 0 <= MODULE.source_fold("another-source") < 5


def test_mixed_groups_uses_only_declared_consequence_split():
    states = [
        {"group_id": 0, "consequence_split": "train", "mixed_outcomes": True},
        {"group_id": 1, "consequence_split": "reserved_ranking_val", "mixed_outcomes": True},
        {"group_id": 2, "consequence_split": "train", "mixed_outcomes": False},
    ]
    assert MODULE.mixed_groups(states, "train") == [0]
    assert MODULE.mixed_groups(states, "reserved_ranking_val") == [1]
