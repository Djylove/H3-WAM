from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FREEZE = load("_c67_freeze_test", "scripts/h3wam/freeze_c67_rollout_source.py")
PREPARE = load("_c67_prepare_test", "scripts/h3wam/prepare_c67_budget_rollout.py")
AGGREGATE = load("_c67_aggregate_test", "scripts/h3wam/aggregate_c67_budget_paired680.py")
SERVER = load("_c67_server_test", "scripts/h3wam/serve_rollout_policy.py")


EXPECTED_SEVEN = {
    "demo_manifest_sha256": "b0d611c21059fa7da6fb08162b03efadd59aff68354bb101be41d3ae20d98eb1",
    "source_manifest_sha256": "cab8876f067114dce41d16ca52cb0bafddf17da33c92d0adde5f11d7ac9555b9",
    "demo_stats_sha256": "6f7e9f4a2232a798e4e30ad26f5748e71aeeda7fa54cb6ea2d0a3ec7d290e814",
    "c48_dataset_sha256": "d416d86c09ba334fae449a131510b84fa1d111e665a77eabfb248f1c79a5bc61",
    "c48_observations_sha256": "399d93f31a8f26297145942387a233b9667049efc60ac1f46514a3f7ce77a638",
    "c59_completed_sha256": "4e67bb95b69ada2a854d3b2bf4ba434c6b3072c2bba11a91df2c30c6de5eeb99",
    "c59_sample_labels_sha256": "f2be6801cac2f1c5b680b30c5e089f47e2bf428f179ee13c1ae283e2d47a9d53",
}


def test_historical_c60_seven_data_hashes_are_exact_and_shared_with_server():
    assert PREPARE.HISTORICAL_C60_DATA_SHA256 == EXPECTED_SEVEN
    assert SERVER._C67_HISTORICAL_DATA_SHA256 == EXPECTED_SEVEN
    assert len(EXPECTED_SEVEN) == 7


def test_exact_680_pair_grid_has_two_fresh_process_jobs_per_pair(tmp_path: Path):
    endpoints = {
        "matched_control": (Path("/checkpoints/s10.pt"), "a" * 64, 10_000),
        "treatment": (Path("/checkpoints/s20.pt"), "b" * 64, 20_000),
    }
    jobs = PREPARE.build_jobs(tmp_path / "rollout", endpoints)
    assert len(jobs) == 1_360
    assert len({row["pair_id"] for row in jobs}) == 680
    for pair_id in range(680):
        rows = [row for row in jobs if row["pair_id"] == pair_id]
        assert {row["arm"] for row in rows} == {"matched_control", "treatment"}
        assert len({(row["suite"], tuple(row["tasks"]), tuple(row["trials"])) for row in rows}) == 1
        assert all(row["episodes"] == 1 for row in rows)


def test_offline_authorization_requires_exact_pass_and_endpoint_restore_hashes(
    tmp_path: Path,
):
    training = tmp_path / "TRAINING_COMPLETE.json"
    training.write_text(json.dumps({"contract_sha256": "c" * 64}))
    endpoints = {}
    identity = {}
    for name, milestone in (("matched_control", 10_000), ("treatment", 20_000)):
        checkpoint = tmp_path / f"s{milestone}.pt"
        checkpoint.write_bytes(name.encode())
        restore = tmp_path / f"restore{milestone}.json"
        restore.write_text(json.dumps({"step": milestone}))
        digest = PREPARE.sha256_file(checkpoint)
        endpoints[name] = (checkpoint.resolve(), digest, milestone)
        identity[name] = {
            "milestone": milestone, "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": digest, "restore_audit": str(restore.resolve()),
            "restore_audit_sha256": PREPARE.sha256_file(restore),
        }
    offline = tmp_path / "RESULTS.json"
    payload = {
        "format": "h3wam-c67-budget-balanced80-result-v1",
        "status": "PASS_C67_BUDGET_BALANCED80_GATE",
        "permission": "GO_C67_PAIRED_680_ROLLOUT",
        "training_complete": {
            "path": str(training.resolve()),
            "sha256": PREPARE.sha256_file(training),
            "contract_sha256": "c" * 64,
        },
        "endpoint_identity": identity,
        "gates": {"offline_sustained": True, "conditioning": True},
    }
    offline.write_text(json.dumps(payload))
    assert PREPARE.validate_offline(offline, training, endpoints)["permission"] == (
        "GO_C67_PAIRED_680_ROLLOUT"
    )
    payload["gates"]["conditioning"] = False
    offline.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not authorize"):
        PREPARE.validate_offline(offline, training, endpoints)


def test_server_authorization_accepts_only_two_frozen_endpoints():
    payload = {
        "format": PREPARE.FORMAT,
        "status": "AUTHORIZED_C67_S10_S20_PAIRED_680",
        "permission": "GO_C67_1360_FRESH_PROCESSES_NO_INTERMEDIATE_STOP",
        "effect_status": "NOT_EVIDENCE_READY",
        "release_signed": False,
        "endpoints": {
            "matched_control": {"milestone": 10_000, "checkpoint_sha256": "a" * 64},
            "treatment": {"milestone": 20_000, "checkpoint_sha256": "b" * 64},
        },
        "jobs": 1_360, "pairs": 680, "episodes_per_arm": 680,
        "one_episode_per_process": True,
        "historical_c60_data_sha256": EXPECTED_SEVEN,
        "offline_results": {
            "status": "PASS_C67_BUDGET_BALANCED80_GATE",
            "permission": "GO_C67_PAIRED_680_ROLLOUT",
        },
        "source_freeze": {
            "git_commit": "c" * 40, "git_tree": "d" * 40,
            "snapshot": "/readonly/snapshot", "sha256": "f" * 64,
            "dynamic_execution_sha256": {"serve": "e" * 64},
        },
    }
    assert SERVER._validate_c67_budget_rollout_authorization(
        payload, "a" * 64
    )[:2] == ("matched_control", 10_000)
    payload["historical_c60_data_sha256"]["demo_stats_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="authorization contract"):
        SERVER._validate_c67_budget_rollout_authorization(payload, "b" * 64)


def test_complete_commit_tree_snapshot_and_dynamic_source_are_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "runtime.py").write_text("VALUE = 1\n")
    subprocess.run(("git", "init", "-q", str(project)), check=True)
    subprocess.run(("git", "-C", str(project), "config", "user.email", "test@example.com"), check=True)
    subprocess.run(("git", "-C", str(project), "config", "user.name", "test"), check=True)
    subprocess.run(("git", "-C", str(project), "add", "runtime.py"), check=True)
    subprocess.run(("git", "-C", str(project), "commit", "-qm", "freeze"), check=True)
    commit = FREEZE.run_git(project, "rev-parse", "HEAD")
    tree = FREEZE.run_git(project, "rev-parse", "HEAD^{tree}")
    monkeypatch.setattr(FREEZE, "DYNAMIC_EXECUTION_FILES", ("runtime.py",))
    snapshot = tmp_path / "snapshot"
    report = FREEZE.freeze(project, commit, tree, snapshot)
    assert report["git_commit"] == commit and report["git_tree"] == tree
    assert report["dynamic_execution_sha256"] == {
        "runtime.py": FREEZE.sha256_file(snapshot / "runtime.py")
    }
    assert FREEZE.verify(snapshot)["tracked_file_count"] == 1
    assert stat.S_IMODE((snapshot / "runtime.py").stat().st_mode) & 0o222 == 0
    (snapshot / "runtime.py").chmod(0o644)
    with pytest.raises(ValueError, match="writable"):
        FREEZE.verify(snapshot)
    (snapshot / "runtime.py").chmod(0o444)
    snapshot.chmod(0o755)


def _passing_pairs() -> list[dict]:
    rows = []
    for suite in PREPARE.SUITES:
        for index in range(170):
            rows.append({
                "suite": suite,
                "candidate": index < 81,
                "control": index < 75,
            })
    return rows


def test_aggregator_preregisters_all_budget_effect_gates():
    threshold = {
        "absolute_gain_at_least_0_03": 0.03,
        "net_wins_at_least": 20,
        "one_sided_exact_mcnemar_p_at_most": 0.05,
        "no_suite_regression_below": -0.03,
        "treatment_successes_at_least_historical_c60": 313,
    }
    overall, _, gates = AGGREGATE.decision_gates(_passing_pairs(), threshold)
    assert overall["candidate_successes"] == 324
    assert all(gates.values())
    weak = _passing_pairs()
    for row in weak[69:81]:
        row["candidate"] = False
    _, _, failed = AGGREGATE.decision_gates(weak, threshold)
    assert not all(failed.values())


def test_launcher_and_rollout_are_fail_closed_and_do_not_sign_or_launch_here():
    launcher = (
        ROOT / "scripts/h3wam/launch_c67_budget_paired680_8gpu.sh"
    ).read_text()
    rollout = (ROOT / "scripts/h3wam/rollout_libero.py").read_text()
    for token in (
        "SOURCE_FREEZE.json", "--expected-manifest-sha256",
        "historical seven-data SHA fail-close", "len(jobs)!=1360",
        "for gpu in 0 1 2 3 4 5 6 7", "--wait-steps 30",
        "--replan-steps 8", "--model-evaluations 10",
        "--c67-budget-rollout-authorization",
        "export PROJECT_ROOT=\"${project}\"",
    ):
        assert token in launcher
    assert "--environment-seed" not in launcher
    assert "--policy-noise-seed-base" not in launcher
    assert "c67_budget_rollout_authorization_sha256" in rollout
    assert "sys.argv[4:11]" in launcher and "sys.argv[11]" in launcher
    assert "release_signed" in launcher
    assert "git commit" not in launcher.lower()


def test_dynamic_execution_manifest_covers_runtime_and_import_targets():
    for name in (
        "scripts/h3wam/rollout_libero.py",
        "scripts/h3wam/serve_rollout_policy.py",
        "scripts/h3wam/aggregate_c58b_expanded_paired_eval.py",
        "src/fastwam/models/h3wam/fastwam_full_tower.py",
        "src/fastwam/models/h3wam/starwam_feature_action.py",
        "third_party/FastWAM/src/fastwam/models/wan22/action_dit.py",
        "third_party/StarWAM/starwam/modules/action_dit.py",
    ):
        assert name in FREEZE.DYNAMIC_EXECUTION_FILES
