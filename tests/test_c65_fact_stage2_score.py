from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "h3wam" / "aggregate_c65_fact_stage2_pairs.py"
SPEC = importlib.util.spec_from_file_location("_test_c65_score_aggregate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AGG = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AGG
SPEC.loader.exec_module(AGG)


def rows(*, wins=(13, 13, 13, 13), ties=(0, 0, 0, 0)):
    result = []
    index = 0
    for suite_index, suite in enumerate(AGG.SUITES):
        for within in range(20):
            tied = within < ties[suite_index]
            preferred = not tied and within < ties[suite_index] + wins[suite_index]
            margin = 0.0 if tied else (1.0 if preferred else -1.0)
            result.append(
                {
                    "pair_index": index,
                    "source_id": within,
                    "suite": suite,
                    "success_preferred": preferred,
                    "tie": tied,
                    "failure_minus_success": margin,
                    "score_finite": True,
                    "order_invariance_pass": True,
                    "identity_pass": True,
                    "success_action_sha256": f"success-{index}",
                    "failure_action_sha256": f"failure-{index}",
                }
            )
            index += 1
    return result


def test_exact_frozen_pass_boundary_without_ties():
    result = AGG.aggregate_rows(rows())
    assert result["success_preferred"] == 52
    assert result["conditional_preference"] == 0.65
    assert result["one_sided_exact_binomial_p"] <= 0.05
    assert result["status"] == "PASS_C65_FACT_STAGE2_CROSS_SUITE_CONFIRMATORY"
    assert result["permission"] == "GO_SEPARATE_PREREGISTERED_N1_VS_N4_CLOSED_LOOP_ONLY"


def test_one_tie_per_suite_can_pass_coverage_and_preference():
    result = AGG.aggregate_rows(rows(wins=(13, 13, 12, 12), ties=(1, 1, 1, 1)))
    assert result["non_ties"] == 76
    assert result["ties"] == 4
    assert result["conditional_preference"] == 50 / 76
    assert result["status"] == "PASS_C65_FACT_STAGE2_CROSS_SUITE_CONFIRMATORY"


def test_two_ties_in_one_suite_fail_frozen_coverage():
    result = AGG.aggregate_rows(rows(ties=(2, 0, 0, 0)))
    assert not result["gates"]["every_suite_non_tie_coverage_at_least_19_of_20"]
    assert result["status"] == "FAIL_C65_FACT_STAGE2_CROSS_SUITE_CONFIRMATORY"


def test_suite_shortfall_cannot_be_hidden_by_overall_wins():
    result = AGG.aggregate_rows(rows(wins=(11, 20, 20, 20)))
    assert result["conditional_preference"] > 0.65
    assert not result["gates"]["every_suite_conditional_preference_at_least_0_60"]
    assert result["status"] == "FAIL_C65_FACT_STAGE2_CROSS_SUITE_CONFIRMATORY"


def test_mechanical_failure_is_structured_fail():
    values = rows()
    values[0]["order_invariance_pass"] = False
    result = AGG.aggregate_rows(values)
    assert not result["gates"]["mechanics"]
    assert result["status"] == "FAIL_C65_FACT_STAGE2_CROSS_SUITE_CONFIRMATORY"
