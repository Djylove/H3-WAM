import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FINAL = load(
    "c58b_online_balanced80_final_test",
    ROOT / "scripts/h3wam/finalize_c58b_online_balanced80.py",
)
AGG = load(
    "c58b_fresh_libero_aggregate_test",
    ROOT / "scripts/h3wam/aggregate_c58b_fresh_libero.py",
)
ROLLOUT = load(
    "c58b_online_rollout_test", ROOT / "scripts/h3wam/rollout_libero.py"
)


def balanced_report(tmp_path: Path) -> Path:
    path = tmp_path / "report.json"
    payload = {
        "format": "h3wam-c58b-online-h3-balanced80-v1",
        "candidate": "C58B_FASTWAM_FULL30_H3_LAYERWISE",
        "checkpoint": {
            "path": str((tmp_path / "s10000.pt").resolve()),
            "sha256": "a" * 64,
            "completed_steps": 10_000,
            "fresh_restore": {"max_abs": 0.0},
        },
        "execution": {
            "h3": "online_frozen_int8",
            "h3_checkpoint_sha256": FINAL.H3_SHA256,
            "disk_kv_read": False,
            "disk_feature_read": False,
            "carrier_layers": list(FINAL.LAYERS),
            "carrier_mapping": "one_to_one_uniform_h3_50_to_action30",
        },
        "data": {
            "selected_sample_ids_sha256": FINAL.SELECTED_IDS_SHA256,
            "selection": {
                "selected_ids_sha256": FINAL.SELECTED_IDS_SHA256,
                "selected_items": 80,
                "selected_task_count": 40,
                "task_counts": {f"task{i}": 2 for i in range(40)},
            },
            "split_audit": {"window_overlap": 0, "episode_overlap": 0},
        },
        "inference": {
            "shift": 5.0,
            "steps": 10,
            "seed": 42,
            "batch_size": 1,
            "same_noise_for_baseline_language_visual": True,
        },
        "metrics": {
            "normalized_clip5_model_domain": {
                "action_mse": 0.4, "prediction_std": 0.2
            },
            "denormalized_official_minmax_clamp": {"action_mse": 0.01},
            "gripper_sign": {"macro_f1": 0.6},
            "language_replacement_sensitivity": {
                "mean_abs_prediction_delta": 0.02
            },
            "visual_feature_shuffle": {
                "baseline_vs_shuffle_action_delta": {
                    "normalized_model_domain": {"action_mse": 0.001}
                }
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_balanced80_gate_releases_only_online_noncollapsed_report(tmp_path):
    path = balanced_report(tmp_path)
    result = FINAL.finalize(path)
    assert result["permission"] == "GO_FRESH_LIBERO"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["execution"]["disk_kv_read"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="execution mismatch"):
        FINAL.finalize(path)


def test_rollout_routes_c58b_with_offline_gate(tmp_path):
    argv = [
        "rollout", "--policy", "h3_fastwam_online_int8",
        "--checkpoint", "c58b.pt", "--cache-root", "cache",
        "--policy-python", "python", "--output-dir", "out",
        "--h3-checkpoint", "h3.safetensors", "--h3-model", "h3-model",
        "--dreamwam-source-manifest", "source.jsonl",
        "--c58b-balanced80-ready", "BALANCED80_READY.json",
        "--model-evaluations", "10",
    ]
    with patch.object(sys, "argv", argv):
        args = ROLLOUT.parse_args()
    command = ROLLOUT.policy_command(args, 1234, tmp_path / "ready.json")
    assert "h3_fastwam_online_int8" in command
    assert "--c58b-balanced80-ready" in command


def test_fresh_aggregate_requires_all_four_exact_suites(tmp_path, monkeypatch):
    checkpoint = tmp_path / "s10000.pt"
    checkpoint.touch()
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "permission": "GO_FRESH_LIBERO",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": "a" * 64,
        "closed_loop_protocol": {"trial_indices": [33]},
    }), encoding="utf-8")
    d0_checkpoint = tmp_path / "d0.pt"
    d0_checkpoint.write_bytes(b"d0")
    monkeypatch.setattr(AGG, "EXPECTED_D0_SHA256", AGG.sha256_file(d0_checkpoint))
    for arm, policy in AGG.ARMS.items():
        for suite in AGG.SUITES:
            directory = tmp_path / arm / suite
            directory.mkdir(parents=True)
            arm_checkpoint = checkpoint if arm == "candidate_c58b" else d0_checkpoint
            payload = {
                "policy": policy, "suite": suite,
                "task_ids": list(range(10)), "trial_indices": [33],
                "trials_per_task": 1, "replan_steps": 8,
                "action_horizon": 32, "model_evaluations": 10,
                "wait_steps": 30,
                "environment_seed": None, "policy_noise_seed_base": None,
                "normalized_action_pre_clamp": True,
                "sample_ensemble_size": 1, "use_action_ensembler": False,
                "save_trajectories": False,
                "checkpoint": str(arm_checkpoint.resolve()),
                "tasks": [
                    {
                        "task_id": task,
                        "episodes": [{
                            "trial": 33,
                            "episode_seed": 33_042 + task * 100_000,
                            "environment_seed": None,
                            "replans": 50,
                            "replan_noise_seeds": list(range(
                                33_042 + task * 100_000,
                                33_092 + task * 100_000,
                            )),
                            "success": (
                                task % 2 == 0
                                if arm == "candidate_c58b" else task % 3 == 0
                            ),
                        }],
                    }
                    for task in range(10)
                ],
            }
            (directory / "results.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
    result = AGG.aggregate(tmp_path, gate, d0_checkpoint)
    assert result["paired_episodes_per_arm"] == 40
    assert result["candidate_successes"] == 20
    assert result["control_successes"] == 16
    assert result["paired_effect"]["candidate_wins"] == 12
    assert result["paired_effect"]["control_wins"] == 8
    assert 0.0 <= result["paired_effect"]["one_sided_p_candidate_better"] <= 1.0


def test_exact_mcnemar_direction_and_no_discordance():
    assert AGG._exact_mcnemar(0, 0)["one_sided_p_candidate_better"] == 1.0
    strong = AGG._exact_mcnemar(10, 0)
    assert strong["one_sided_p_candidate_better"] == pytest.approx(1 / 1024)
    assert strong["two_sided_p"] == pytest.approx(2 / 1024)


def test_fresh_aggregate_accepts_early_success_seed_prefix(tmp_path, monkeypatch):
    checkpoint = tmp_path / "s10000.pt"
    checkpoint.touch()
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "permission": "GO_FRESH_LIBERO",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": "a" * 64,
        "closed_loop_protocol": {"trial_indices": [33]},
    }), encoding="utf-8")
    d0_checkpoint = tmp_path / "d0.pt"
    d0_checkpoint.write_bytes(b"d0")
    monkeypatch.setattr(AGG, "EXPECTED_D0_SHA256", AGG.sha256_file(d0_checkpoint))
    for arm, policy in AGG.ARMS.items():
        for suite in AGG.SUITES:
            directory = tmp_path / arm / suite
            directory.mkdir(parents=True)
            arm_checkpoint = checkpoint if arm == "candidate_c58b" else d0_checkpoint
            tasks = []
            for task in range(10):
                replans = 7 if task == 0 else 50
                episode_seed = 33_042 + task * 100_000
                tasks.append({
                    "task_id": task,
                    "episodes": [{
                        "trial": 33,
                        "episode_seed": episode_seed,
                        "environment_seed": None,
                        "replans": replans,
                        "replan_noise_seeds": list(range(
                            episode_seed, episode_seed + replans
                        )),
                        "success": task == 0,
                    }],
                })
            payload = {
                "policy": policy, "suite": suite,
                "task_ids": list(range(10)), "trial_indices": [33],
                "trials_per_task": 1, "replan_steps": 8,
                "action_horizon": 32, "model_evaluations": 10,
                "wait_steps": 30, "environment_seed": None,
                "policy_noise_seed_base": None,
                "normalized_action_pre_clamp": True,
                "sample_ensemble_size": 1, "use_action_ensembler": False,
                "save_trajectories": False,
                "checkpoint": str(arm_checkpoint.resolve()),
                "tasks": tasks,
            }
            (directory / "results.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
    result = AGG.aggregate(tmp_path, gate, d0_checkpoint)
    assert result["candidate_successes"] == 4
    assert result["control_successes"] == 4


def test_final_watcher_pins_official_fastwam_source_for_eval_and_rollout():
    source = (
        ROOT / "scripts/h3wam/watch_c58b_online_final_eval.sh"
    ).read_text(encoding="utf-8")
    assert "upstream-readonly/FastWAM-45d8e145/wan22" in source
    assert 'export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"' in source
    assert "--wait-steps 30" in source
    assert "--environment-seed" not in source
    assert "--policy-noise-seed-base" not in source
