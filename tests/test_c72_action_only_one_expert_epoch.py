import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_finalizer():
    path = ROOT / "scripts/h3wam/finalize_c72_action_only_one_expert_epoch.py"
    spec = importlib.util.spec_from_file_location("_test_c72_finalizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_c72_is_a_fresh_fixed_parent_one_expert_epoch_budget_ablation():
    canary = (
        ROOT / "scripts/h3wam/launch_c72_action_only_one_expert_epoch_canary_8gpu.sh"
    ).read_text()
    long = (
        ROOT / "scripts/h3wam/launch_c72_action_only_one_expert_epoch_30195_8gpu.sh"
    ).read_text()
    assert "--c58-parent-checkpoint" in canary and "--c58-parent-checkpoint" in long
    assert "--objective-mode action_only" in canary and "--objective-mode action_only" in long
    assert "--scheduler-horizon 30195" in canary and "--scheduler-horizon 30195" in long
    assert "--nproc-per-node 8" in canary and "--nproc-per-node 8" in long
    assert "milestones=($(seq 1000 1000 30000) 30195)" in long
    assert "restore-check-only" in long
    assert "C72_CANARY_GO_LONG" in long
    assert "C72_RELEASE_FILE" in long
    assert "c69_action_only_s20000.pt" not in long


def test_c72_finalizer_tracks_exact_endpoint_and_cosine():
    finalizer = load_finalizer()
    assert finalizer.MILESTONES[:2] == (1000, 2000)
    assert finalizer.MILESTONES[-2:] == (30000, 30195)
    assert len(finalizer.MILESTONES) == 31
    assert finalizer.previous_milestone(30195) == 30000
    assert finalizer.expected_lr_factor(30195) == 0.0
    expected = 0.5 * (1.0 + math.cos(math.pi * (20000 - 500) / 29695))
    assert math.isclose(finalizer.expected_lr_factor(20000), expected)


def test_c72_dossier_budget_arithmetic_and_control_are_explicit():
    import json

    dossier = json.loads((
        ROOT / "experiments/dossiers/h3_c72_action_only_one_expert_epoch_v1.json"
    ).read_text())
    budget = dossier["budget"]
    assert budget["training_samples"] == budget["global_batch"] * budget["optimizer_steps"]
    assert budget["expert_training_samples"] == 4 * budget["optimizer_steps"]
    assert math.isclose(
        budget["cumulative_expert_effective_epochs_with_c58_parent"],
        (80000 + 4 * 30195) / 200779,
        rel_tol=1e-6,
    )
    assert "C72-s30195 minus C72-s20000" in dossier["effect_preregistration"]["primary_comparison"]
    assert dossier["decision"]["status"] == "GO_LONG"


def test_c72_preview_queue_is_read_only_and_covers_the_exact_trajectory():
    queue = (
        ROOT / "scripts/h3wam/launch_c72_action_only_milestone_preview_queue.sh"
    ).read_text()
    evaluator = (
        ROOT / "scripts/h3wam/evaluate_c67_fact_milestone_balanced80.py"
    ).read_text()
    auditor = (
        ROOT / "scripts/h3wam/prepare_c72_milestone_preview_audit.py"
    ).read_text()
    assert "milestones=($(seq 1000 1000 30000) 30195)" in queue
    assert "--variant c72" in queue
    assert "c72_action_only_s${milestone}.pt" in queue
    assert "for gpu in 0 1 2 3 4 5 6 7" in queue
    assert "torch.distributed.run" not in queue
    assert "train_c56b_fact_online.py" not in queue
    assert "rollout_libero" not in queue
    assert 'C72_MILESTONES = tuple(range(1_000, 30_001, 1_000)) + (30_195,)' in evaluator
    assert '"c72": "h3wam-c72-action-only-milestone-balanced80-v1"' in evaluator
    assert '"PASS_C72_MILESTONE_STRICT_RESTORE"' in evaluator
    assert "FINAL.validate_milestone" in auditor
    assert "PREVIEW_EVALUATION_ONLY" in auditor
