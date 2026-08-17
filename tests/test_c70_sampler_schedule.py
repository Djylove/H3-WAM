from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_c56b_fact_online_c70_test",
    ROOT / "scripts/h3wam/train_c56b_fact_online.py",
)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


def test_default_schedule_is_byte_contract_compatible() -> None:
    assert TRAIN.rank_schedule_contract(TRAIN.DEFAULT_RANK_SCHEDULE) == {
        "rank_categories": list(TRAIN.RANK_CATEGORIES)
    }
    for step in (1, 2, 999, 20_000):
        assert [
            TRAIN.rank_category(TRAIN.DEFAULT_RANK_SCHEDULE, rank, step)
            for rank in range(8)
        ] == list(TRAIN.RANK_CATEGORIES)


def test_c70_schedule_has_exact_two_step_mixture() -> None:
    by_step = []
    for step in (1, 2):
        categories = [
            TRAIN.rank_category(TRAIN.C70_RANK_SCHEDULE, rank, step)
            for rank in range(8)
        ]
        by_step.append(categories)
        assert Counter(categories)["expert_demo"] == 6
        assert Counter(categories)["success_rollout"] == 1
        assert sum("failure" in category for category in categories) == 1
    assert by_step[0][7] == "observational_failure"
    assert by_step[1][7] == "causal_failure"
    assert Counter(by_step[0] + by_step[1]) == Counter({
        "expert_demo": 12,
        "success_rollout": 2,
        "observational_failure": 1,
        "causal_failure": 1,
    })


def test_c70_vae_ownership_covers_every_possible_stream() -> None:
    assert [
        TRAIN.rank_requires_vae(TRAIN.C70_RANK_SCHEDULE, rank)
        for rank in range(8)
    ] == [False, False, False, False, False, False, True, True]


@pytest.mark.parametrize("rank,step", [(-1, 1), (8, 1), (0, 0)])
def test_rank_schedule_rejects_invalid_coordinates(rank: int, step: int) -> None:
    with pytest.raises(ValueError):
        TRAIN.rank_category(TRAIN.C70_RANK_SCHEDULE, rank, step)
