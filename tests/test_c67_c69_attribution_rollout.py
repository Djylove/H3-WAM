import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREPARE = load(
    "_test_c67_c69_prepare", "scripts/h3wam/prepare_c67_c69_attribution_rollout.py"
)
AGGREGATE = load(
    "_test_c67_c69_aggregate",
    "scripts/h3wam/aggregate_c67_c69_attribution_paired680.py",
)
SERVER = load("_test_c67_c69_server", "scripts/h3wam/serve_rollout_policy.py")


def authorization_fixture():
    return {
        "format": PREPARE.FORMAT,
        "status": "AUTHORIZED_C67_C69_FIXED_S20_PAIRED_680",
        "permission": "GO_C67_C69_1360_FRESH_PROCESSES_NO_INTERMEDIATE_STOP",
        "effect_status": "NOT_EVIDENCE_READY",
        "release_signed": False,
        "endpoints": {
            "c67_fact_joint": {"milestone": 20_000, "checkpoint_sha256": "a" * 64},
            "c69_action_only": {"milestone": 20_000, "checkpoint_sha256": "b" * 64},
        },
        "jobs": 1_360,
        "pairs": 680,
        "episodes_per_arm": 680,
        "one_episode_per_process": True,
        "historical_c60_data_sha256": dict(PREPARE.HISTORICAL_C60_DATA_SHA256),
        "offline_attribution": {
            "status": "PASS_C67_C69_FIXED_S20_ATTRIBUTION_CHAIN",
            "permission": "GO_C67_VS_C69_FIXED_S20_PAIRED_LIBERO_ATTRIBUTION",
        },
        "source_freeze": {
            "git_commit": "c" * 40,
            "git_tree": "d" * 40,
            "snapshot": "/readonly/snapshot",
            "sha256": "e" * 64,
            "dynamic_execution_sha256": {"serve": "f" * 64},
        },
    }


def test_server_accepts_exact_fixed_s20_attribution_endpoints():
    authorization = authorization_fixture()
    assert SERVER._validate_c67_c69_attribution_authorization(
        authorization, "a" * 64
    )[:2] == ("c67_fact_joint", 20_000)
    assert SERVER._validate_c67_c69_attribution_authorization(
        authorization, "b" * 64
    )[:2] == ("c69_action_only", 20_000)
    authorization["endpoints"]["c69_action_only"]["milestone"] = 19_000
    try:
        SERVER._validate_c67_c69_attribution_authorization(
            authorization, "b" * 64
        )
    except ValueError as error:
        assert "contract failed" in str(error)
    else:
        raise AssertionError("drifted C69 endpoint was accepted")


def test_grid_is_full_fresh_680_pair_benchmark(tmp_path: Path):
    endpoints = {
        "c67_fact_joint": (tmp_path / "c67.pt", "a" * 64, 20_000),
        "c69_action_only": (tmp_path / "c69.pt", "b" * 64, 20_000),
    }
    jobs = PREPARE.build_jobs(tmp_path / "rollout", endpoints)
    assert len(jobs) == 1_360
    assert len({row["pair_id"] for row in jobs}) == 680
    assert {row["arm"] for row in jobs} == {"c67_fact_joint", "c69_action_only"}
    assert {row["trials"][0] for row in jobs} == set(range(50, 67))
    assert {row["suite"] for row in jobs} == set(PREPARE.SUITES)
    assert all(row["milestone"] == 20_000 for row in jobs)


def summary(*, joint: int, action: int, joint_wins: int, action_wins: int):
    return {
        "success_rate_delta": (joint - action) / 680,
        "candidate_wins": joint_wins,
        "control_wins": action_wins,
        "one_sided_p_candidate_better": 0.001 if joint_wins > action_wins else 0.9,
    }


def test_directional_decision_can_support_either_arm_or_null():
    threshold = {
        "absolute_delta": 0.03,
        "net_wins": 20,
        "one_sided_exact_mcnemar_p": 0.05,
        "suite_regression_floor": -0.03,
    }
    per_suite = {
        suite: {"success_rate_delta": 0.04} for suite in PREPARE.SUITES
    }
    decision, c67, c69 = AGGREGATE.directional_decision(
        summary(joint=330, action=300, joint_wins=50, action_wins=20),
        per_suite,
        threshold,
    )
    assert decision == "SUPPORT_C67_CONSEQUENCE_OBJECTIVE" and all(c67.values())
    reverse_suite = {
        suite: {"success_rate_delta": -0.04} for suite in PREPARE.SUITES
    }
    decision, _, c69 = AGGREGATE.directional_decision(
        summary(joint=300, action=330, joint_wins=20, action_wins=50),
        reverse_suite,
        threshold,
    )
    assert decision.startswith("SUPPORT_C69_ACTION_ONLY") and all(c69.values())
    decision, _, _ = AGGREGATE.directional_decision(
        summary(joint=310, action=305, joint_wins=25, action_wins=20),
        {suite: {"success_rate_delta": 0.0} for suite in PREPARE.SUITES},
        threshold,
    )
    assert decision == "NO_DETECTABLE_INCREMENTAL_CONSEQUENCE_EFFECT"


def test_multinode_launcher_is_pair_sharded_and_never_reads_success():
    launcher = (
        ROOT / "scripts/h3wam/launch_c67_c69_attribution_paired680_shard.sh"
    ).read_text()
    for token in (
        "pair_id\"]%num_shards",
        "(row[\"pair_id\"]//total)%8",
        "--c67-c69-attribution-authorization",
        "--wait-steps 30",
        "--replan-steps 8",
        "--model-evaluations 10",
        "one or more C67/C69 shard workers failed",
    ):
        assert token in launcher
    assert "episode[\"success\"]" not in launcher
    assert "stop early" not in launcher.lower()


def test_rollout_metadata_and_freeze_cover_new_protocol():
    rollout = (ROOT / "scripts/h3wam/rollout_libero.py").read_text()
    freeze = PREPARE.SOURCE.DYNAMIC_EXECUTION_FILES
    assert "c67_c69_attribution_authorization_sha256" in rollout
    for name in (
        "scripts/h3wam/prepare_c67_c69_attribution_rollout.py",
        "scripts/h3wam/launch_c67_c69_attribution_paired680_shard.sh",
        "scripts/h3wam/finalize_c67_c69_attribution_rollout.py",
        "scripts/h3wam/aggregate_c67_c69_attribution_paired680.py",
    ):
        assert name in freeze
