import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AGG = load(
    "c58b_expanded_aggregate_test",
    ROOT / "scripts/h3wam/aggregate_c58b_expanded_paired_eval.py",
)
AUDIT = load(
    "c58b_expanded_audit_test",
    ROOT / "scripts/h3wam/audit_freeze_c58b_expanded_d0.py",
)
PREPARE = load(
    "c58b_expanded_prepare_test",
    ROOT / "scripts/h3wam/prepare_c58b_expanded_paired_eval.py",
)


def episode(task: int, trial: int, replans: int = 7):
    seed = 42 + task * 100_000 + trial * 1_000
    return {
        "trial": trial,
        "episode_seed": seed,
        "environment_seed": None,
        "replans": replans,
        "replan_noise_seeds": list(range(seed, seed + replans)),
        "first_environment_action": np.zeros(7).tolist(),
        "first_environment_action_chunk": np.zeros((32, 7)).tolist(),
        "replan_first_actions": np.zeros((replans, 7)).tolist(),
    }


def test_task_specific_seed_and_early_success_prefix_are_strict():
    AGG.validate_episode(episode(4, 47, replans=3), 4, 47)
    invalid = episode(4, 47, replans=3)
    invalid["episode_seed"] = 47_042
    with pytest.raises(ValueError, match="seed contract"):
        AGG.validate_episode(invalid, 4, 47)


def test_paired_summary_reports_exact_mcnemar_and_intervals():
    rows = (
        [{"candidate": True, "control": False}] * 5
        + [{"candidate": False, "control": True}] * 3
        + [{"candidate": True, "control": True}] * 10
        + [{"candidate": False, "control": False}] * 22
    )
    result = AGG.paired_summary(rows)
    assert result["pairs"] == 40
    assert result["candidate_wins"] == 5
    assert result["control_wins"] == 3
    assert result["one_sided_p_candidate_better"] == pytest.approx(0.36328125)
    assert result["two_sided_p"] == pytest.approx(0.7265625)
    assert result["candidate_rate_wilson95"][0] < result["candidate_rate"]
    assert result["candidate_rate_wilson95"][1] > result["candidate_rate"]


def test_initial_state_digest_is_ordered_and_value_sensitive():
    values = {
        name: np.asarray([index], dtype=np.float32)
        for index, name in enumerate(AGG.INITIAL_KEYS)
    }
    assert AGG.tensor_digest(values) == AUDIT.tensor_digest(values)
    changed = {name: value.copy() for name, value in values.items()}
    changed["sim_state"][0] += 1
    assert AGG.tensor_digest(values) != AGG.tensor_digest(changed)


def test_prepared_jobs_cover_exact_640_without_overlap():
    assert PREPARE.SUITES == AGG.SUITES
    trials = [trial for group in PREPARE.TRIAL_GROUPS for trial in group]
    assert trials == list(range(34, 50))
    assert len(PREPARE.SUITES) * len(PREPARE.TRIAL_GROUPS) == 8
    assert 8 * 10 * len(PREPARE.TRIAL_GROUPS[0]) == 640


def test_launcher_pins_verified_execution_contract():
    source = (
        ROOT / "scripts/h3wam/launch_c58b_expanded_candidate_only.sh"
    ).read_text(encoding="utf-8")
    assert "--wait-steps 30" in source
    assert "--replan-steps 8" in source
    assert "--action-horizon 32" in source
    assert "--model-evaluations 10" in source
    assert "--save-trajectories" in source
    assert "--environment-seed" not in source
    assert "--policy-noise-seed-base" not in source
    assert "for index in 0 1 2 3 4 5 6 7" in source


def test_control_freeze_is_exact_trials34_through49():
    assert AUDIT.TRIALS == tuple(range(34, 50))
    assert AUDIT.D0_SHA256 == AGG.D0_SHA256
    assert len(AUDIT.SUITES) * len(AUDIT.TRIALS) * 10 == 640
