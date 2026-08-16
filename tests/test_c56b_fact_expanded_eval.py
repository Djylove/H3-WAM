import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PREPARE = load(
    "c56_fact_expanded_prepare_test",
    ROOT / "scripts/h3wam/prepare_c56b_fact_expanded_eval.py",
)


def test_exact_grid_and_round_robin_gpu_contract():
    identities = [
        (suite, task, trial)
        for trial in PREPARE.TRIALS
        for suite in PREPARE.SUITES
        for task in range(10)
    ]
    assert len(identities) == len(set(identities)) == 640
    assert PREPARE.TRIALS == tuple(range(34, 50))


def test_launcher_is_fresh_process_and_never_batches():
    source = (ROOT / "scripts/h3wam/launch_c56b_fact_expanded_isolated.sh").read_text()
    assert '--task-ids "${task}"' in source
    assert '--trial-indices "${trial}"' in source
    assert "for gpu in 0 1 2 3 4 5 6 7" in source
    assert '--wait-steps 30' in source
    assert '--replan-steps 8' in source
    assert '--action-horizon 32' in source
    assert '--model-evaluations 10' in source
    assert "--environment-seed" not in source
    assert "--policy-noise-seed-base" not in source


def test_canary_is_real_restore_and_outcome_redacted():
    launch = (ROOT / "scripts/h3wam/run_c56b_fact_expanded_canary.sh").read_text()
    audit = (ROOT / "scripts/h3wam/audit_c56b_fact_expanded_canary.py").read_text()
    assert "rollout_libero.py" in launch and "--save-trajectories" in launch
    assert '"stage": "ready"' in audit
    assert '"success_redacted": True' in audit
    assert 'episode["success"]' not in audit


def test_pinned_upstream_dependencies_are_all_required():
    source = (ROOT / "scripts/h3wam/run_c56b_fact_expanded_canary.sh").read_text()
    for path in (
        "action_dit.py", "wan_video_dit.py", "helpers/gradient.py",
        "StarWAM/starwam/modules/action_dit.py",
        "StarWAM/starwam/modules/wan_block.py",
        "DreamWAM/dreamwam/layers.py", "DreamWAM/dreamwam/experts.py",
        "DreamWAM/dreamwam/mot.py",
    ):
        assert path in source


def test_prepare_rejects_nonpositive_trial33_screen(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"; checkpoint.write_bytes(b"x")
    ready = tmp_path / "READY.json"
    ready.write_text(json.dumps({
        "status": "PASS_C56B_ONLINE_LONG10000_STRICT_RESTORE",
        "permission": "READY_FOR_PAIRED_HELDOUT", "arm": "C60_MAIN",
        "checkpoint_sha256": PREPARE.C60_SHA256, "checkpoint": str(checkpoint),
    }))
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "status": "PASS_PAIRED_BALANCED80", "permission": "GO_PAIRED_LIBERO",
        "checkpoint_identity": {"c60_main_checkpoint_sha256": PREPARE.C60_SHA256},
    }))
    trial = tmp_path / "trial.json"
    trial.write_text(json.dumps({
        "format": "h3wam-c56b-fact-paired-libero-trial33-v1", "status": "COMPLETE",
        "paired_episodes_per_arm": 40, "main_successes": 18,
        "c58_parent_successes": 18,
        "paired_effects": {"main_vs_c58": {"first_wins": 2, "second_wins": 2}},
        "suites": [],
    }))
    monkeypatch.setattr(PREPARE, "sha256_file", lambda path: {
        checkpoint: PREPARE.C60_SHA256, ready: PREPARE.C60_READY_SHA256,
        gate: PREPARE.PAIRED_GATE_SHA256, trial: PREPARE.TRIAL33_SHA256,
    }[Path(path)])
    try:
        PREPARE.prepare(checkpoint, ready, gate, trial, tmp_path / "out")
    except ValueError as error:
        assert "expansion screen" in str(error)
    else:
        raise AssertionError("nonpositive trial33 screen was accepted")


def test_finalizer_waits_for_both_completed_before_aggregate():
    source = (
        ROOT / "scripts/h3wam/watch_c56b_fact_expanded_finalizer.sh"
    ).read_text()
    wait = source.index('while [[ ! -s "${c60_root}/COMPLETED.json"')
    aggregate = source.index('aggregate_c56b_fact_expanded_paired_eval.py \\')
    assert wait < aggregate
    prefix = source[:aggregate]
    assert "results.json" not in prefix
    assert "success" not in prefix


def test_expanded_aggregator_pins_full_promotion_gate():
    source = (
        ROOT / "scripts/h3wam/aggregate_c56b_fact_expanded_paired_eval.py"
    ).read_text()
    for gate in (
        "absolute_gain_at_least_0_03", "net_wins_at_least_20",
        "one_sided_exact_mcnemar_p_at_most_0_05",
        "no_suite_regression_below_minus_0_03",
    ):
        assert gate in source
    assert 'len(pairs) != 680' in source
    assert "initial_state_sha256" in source
