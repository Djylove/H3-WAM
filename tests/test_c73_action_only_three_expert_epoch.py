import importlib.util
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_finalizer():
    path = ROOT / "scripts/h3wam/finalize_c73_action_only_three_expert_epoch.py"
    spec = importlib.util.spec_from_file_location("_test_c73_finalizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_c73_is_a_fresh_fixed_parent_three_expert_epoch_budget_ablation():
    canary = (
        ROOT / "scripts/h3wam/launch_c73_action_only_three_expert_epoch_canary_8gpu.sh"
    ).read_text()
    long = (
        ROOT / "scripts/h3wam/launch_c73_action_only_three_expert_epoch_130585_8gpu.sh"
    ).read_text()
    assert "--c58-parent-checkpoint" in canary and "--c58-parent-checkpoint" in long
    assert "--objective-mode action_only" in canary and "--objective-mode action_only" in long
    assert "--scheduler-horizon 130585" in canary and "--scheduler-horizon 130585" in long
    assert "--nproc-per-node 8" in canary and "--nproc-per-node 8" in long
    assert "milestones=($(seq 1000 1000 30000) 30195 $(seq 31000 1000 130000) 130585)" in long
    assert "restore-check-only" in long
    assert "C73_CANARY_GO_LONG" in long and "C73_RELEASE_FILE" in long
    assert "c69_action_only_s20000.pt" not in long


def test_c73_finalizer_tracks_one_and_three_expert_epoch_endpoints():
    finalizer = load_finalizer()
    assert finalizer.MILESTONES[:2] == (1000, 2000)
    assert finalizer.MILESTONES[29:32] == (30000, 30195, 31000)
    assert finalizer.MILESTONES[-2:] == (130000, 130585)
    assert len(finalizer.MILESTONES) == 132
    assert finalizer.previous_milestone(30195) == 30000
    assert finalizer.previous_milestone(31000) == 30195
    assert finalizer.expected_lr_factor(130585) == 0.0
    expected = 0.5 * (1.0 + math.cos(math.pi * (30195 - 500) / 130085))
    assert math.isclose(finalizer.expected_lr_factor(30195), expected)


def test_c73_dossier_budget_arithmetic_and_control_are_explicit():
    dossier = json.loads((
        ROOT / "experiments/dossiers/h3_c73_action_only_three_expert_epoch_v1.json"
    ).read_text())
    budget = dossier["budget"]
    assert budget["training_samples"] == budget["global_batch"] * budget["optimizer_steps"]
    assert budget["expert_training_samples"] == 4 * budget["optimizer_steps"]
    assert math.isclose(
        budget["cumulative_expert_effective_epochs_with_c58_parent"],
        (80000 + 4 * 130585) / 200779,
        rel_tol=1e-6,
    )
    assert "C73-s130585 minus C73-s30195" in dossier["effect_preregistration"]["primary_comparison"]
    assert dossier["decision"]["status"] == "GO_CANARY"
