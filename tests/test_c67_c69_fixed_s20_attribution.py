from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

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


SEAL = load("_c69_preview_sealer_test", "scripts/h3wam/seal_c69_milestone_previews.py")
AGG = load(
    "_c67_c69_attribution_test",
    "scripts/h3wam/aggregate_c67_c69_fixed_s20_attribution.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract(arm: str) -> dict:
    joint = arm == "c67"
    value = {
        "format": "h3wam-c56b-fact-online-training-v1",
        "classification": "FACT_full_backbone_port_online_frozen_int8_h3",
        "objective_mode": "fact_joint" if joint else "action_only",
        "rank_categories": [
            "expert_demo", "expert_demo", "expert_demo", "expert_demo",
            "success_rollout", "success_rollout", "observational_failure",
            "causal_failure",
        ],
        "loss_weights": [10.0, 1.0, 0.4, 0.4] if joint else [10.0, 0.0, 0.0, 0.0],
        "frozen_auxiliary_parameters": [] if joint else [
            prefix + "0.weight" for prefix in AGG.AUXILIARY_PREFIXES
        ],
        "target_norm_sha256": "1" * 64,
        "h3_sha256": "2" * 64,
        "d0_sha256": "3" * 64,
        "initialization": {
            "initialization_contract": "strict_online_c58b_parent_v1",
            "c58_completed_steps": 10_000,
        },
        "c58_parent_sha256": "4" * 64,
        "demo_manifest_sha256": "5" * 64,
        "source_manifest_sha256": "6" * 64,
        "demo_stats_sha256": "7" * 64,
        "c48_dataset_sha256": "8" * 64,
        "c48_observations_sha256": "9" * 64,
        "c59_completed_sha256": "a" * 64,
        "c59_sample_labels_sha256": "b" * 64,
        "causal_failure_dataset_sha256": "c" * 64,
        "causal_failure_observations_sha256": "d" * 64,
        "base_lr": 2e-5,
        "action_lr": 2e-4,
        "warmup_steps": 500,
        "scheduler_horizon": 20_000,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "seed": 20260816,
        "gradient_checkpointing": True,
        "action_horizon": 32,
        "action_shift": 5.0,
        "h3_carrier_layers": list(range(30)),
        "h3_execution": "online_frozen_int8_per_rank_v1",
        "no_kv_cache": True,
    }
    if joint:
        # The real C67 trajectory predates the explicit objective-mode fields.
        # Its evaluator/finalizer canonically interpret their absence as the
        # original FACT-joint objective with no frozen auxiliary heads.
        value.pop("objective_mode")
        value.pop("frozen_auxiliary_parameters")
    return value


def metrics(action_mse: float, physical_mse: float) -> dict:
    return {
        "normalized_clip5_model_domain": {
            "action_mse": action_mse,
            "prediction_std": 0.47,
        },
        "denormalized_official_minmax_clamp": {"action_mse": physical_mse},
        "gripper_sign": {"macro_f1": 0.93},
        "language_replacement_end_to_end_h3_and_action": {
            "mean_abs_prediction_delta": 0.22,
        },
        "visual_feature_shuffle_baseline_delta": {"action_mse": 0.034},
    }


def fixture(
    tmp_path: Path,
    *,
    c69_unsafe_step: int | None = None,
    c69_execution_seed: int = 42,
    c69_contract_seed: int = 20260816,
) -> dict[str, Path]:
    c67_train = tmp_path / "c67-train"
    c69_train = tmp_path / "c69-train"
    c67_sealed = tmp_path / "c67-sealed"
    c69_preview = tmp_path / "c69-preview"
    c69_sealed = tmp_path / "c69-sealed"
    for root in (c67_train, c69_train):
        (root / "checkpoints").mkdir(parents=True)
        (root / "reports").mkdir()
    (c67_train / "milestone-audit").mkdir()
    (c67_sealed / "reports").mkdir(parents=True)
    (c69_preview / "preview-audit").mkdir(parents=True)
    (c69_preview / "reports").mkdir()

    c67_contract = contract("c67")
    c69_contract = contract("c69")
    c69_contract["seed"] = c69_contract_seed
    c67_contract_sha = AGG.contract_sha256(c67_contract)
    c69_contract_sha = AGG.contract_sha256(c69_contract)
    c67_audits, c69_audits = [], []
    checkpoint_identity: dict[str, dict[int, tuple[Path, str]]] = {"c67": {}, "c69": {}}
    for step in AGG.MILESTONES:
        for arm, root, prefix, active_contract in (
            ("c67", c67_train, "c67_online", c67_contract),
            ("c69", c69_train, "c69_action_only", c69_contract),
        ):
            checkpoint = root / f"checkpoints/{prefix}_s{step}.pt"
            checkpoint.write_bytes(f"{arm}-fixed-checkpoint-{step}".encode())
            checkpoint_identity[arm][step] = (checkpoint.resolve(), sha(checkpoint))
            (root / f"reports/train_s{step}.json").write_text(
                json.dumps({
                    "status": "PASS_C56B_ONLINE_TRAINING_INVOCATION",
                    "completed_steps": step,
                    "checkpoint": str(checkpoint.resolve()),
                    "contract": active_contract,
                })
            )
        c67_checkpoint, _ = checkpoint_identity["c67"][step]
        c69_checkpoint, c69_checkpoint_sha = checkpoint_identity["c69"][step]
        c67_audit = {
            "milestone": step,
            "status": "PASS_C67_BUDGET_MILESTONE_STRICT_RESTORE",
            "checkpoint": str(c67_checkpoint),
            "restore_max_abs": 0.0,
            "gate": {"strict_restore": True, "checkpoint_schema": True},
        }
        c69_audit = {
            "milestone": step,
            "status": "PASS_C69_MILESTONE_STRICT_RESTORE",
            "checkpoint": str(c69_checkpoint),
            "checkpoint_sha256": c69_checkpoint_sha,
            "gate": {"restore": True, "checkpoint_schema": True},
        }
        c67_audits.append(c67_audit)
        c69_audits.append(c69_audit)
        (c67_train / f"milestone-audit/s{step}.json").write_text(json.dumps(c67_audit))

    c67_s20, c67_s20_sha = checkpoint_identity["c67"][20_000]
    c69_s20, c69_s20_sha = checkpoint_identity["c69"][20_000]
    c67_complete_path = c67_train / "TRAINING_COMPLETE.json"
    c69_complete_path = c69_train / "TRAINING_COMPLETE.json"
    c67_complete = {
        "format": AGG.C67_COMPLETE_FORMAT,
        "status": "PASS_C67_BUDGET_TRAINING_COMPLETE",
        "permission": "READY_FOR_PREREGISTERED_OFFLINE_ONLY",
        "effect_status": "NOT_EVIDENCE_READY",
        "completed_steps": 20_000,
        "global_batch": 8,
        "training_samples": 160_000,
        "contract_sha256": c67_contract_sha,
        "milestone_audits": c67_audits,
        "treatment": {
            "milestone": 20_000,
            "training_samples": 160_000,
            "checkpoint": str(c67_s20),
            "checkpoint_sha256": c67_s20_sha,
        },
    }
    c69_complete = {
        "format": AGG.C69_COMPLETE_FORMAT,
        "status": "PASS_C69_MATCHED_ACTION_ONLY_TRAINING_COMPLETE",
        "permission": "READY_FOR_PREREGISTERED_OFFLINE_ONLY",
        "effect_status": "NOT_EVIDENCE_READY",
        "completed_steps": 20_000,
        "training_samples": 160_000,
        "matched_joint_arm": "C67-s20000",
        "contract_sha256": c69_contract_sha,
        "milestone_audits": c69_audits,
        "final_checkpoint": str(c69_s20),
        "final_checkpoint_sha256": c69_s20_sha,
    }
    c67_complete_path.write_text(json.dumps(c67_complete))
    c69_complete_path.write_text(json.dumps(c69_complete))
    c67_complete_sha, c69_complete_sha = sha(c67_complete_path), sha(c69_complete_path)

    sample_ids = [f"sample-{index:02d}" for index in range(80)]
    selection = {
        "selected_ids_sha256": AGG.SELECTED_IDS_SHA256,
        "selected_task_count": 40,
        "task_counts": {f"task-{index:02d}": 2 for index in range(40)},
    }
    data = {
        "source_manifest_sha256": "6" * 64,
        "demo_manifest_sha256": "5" * 64,
        "demo_stats_sha256": "7" * 64,
        "validation_manifest_sha256": "e" * 64,
        "selection": selection,
        "split_audit": {"episode_disjoint": True},
        "visual_shuffle": {"self_maps": 0},
    }
    execution = {
        "h3": "online_frozen_int8",
        "h3_checkpoint_sha256": "2" * 64,
        "disk_kv_read": False,
        "disk_kv_write": False,
        "disk_feature_read": False,
        "carrier_layers": list(range(30)),
        "same_selected_samples_noise_solver_normalization": True,
        "seed": 42,
        "inference_steps": 10,
        "shift": 5.0,
    }

    c67_report_sha: dict[str, str] = {}
    c69_preview_sha: dict[str, str] = {}
    for step in AGG.MILESTONES:
        c67_checkpoint, c67_checkpoint_sha = checkpoint_identity["c67"][step]
        c69_checkpoint, c69_checkpoint_sha = checkpoint_identity["c69"][step]
        c67_restore = c67_train / f"milestone-audit/s{step}.json"
        c67_report = {
            "format": AGG.C67_REPORT_FORMAT,
            "variant": "c67",
            "status": "PASS_FIXED_BALANCED80",
            "permission": "DIAGNOSTIC_ONLY_PENDING_FIXED_AGGREGATION",
            "effect_status": "DIAGNOSTIC_NOT_CHECKPOINT_SELECTION",
            "milestone": step,
            "checkpoint": str(c67_checkpoint),
            "checkpoint_sha256": c67_checkpoint_sha,
            "restore_audit": str(c67_restore.resolve()),
            "restore_audit_sha256": sha(c67_restore),
            "training_complete": str(c67_complete_path.resolve()),
            "training_complete_sha256": c67_complete_sha,
            "training_contract_sha256": c67_contract_sha,
            "data": data,
            "execution": execution,
            "arm": {
                "checkpoint_completed_steps": step,
                "strict_fresh_restore": {"max_abs": 0.0},
                "evaluated_ids_sha256": AGG.SELECTED_IDS_SHA256,
                "metrics": metrics(0.08, 0.04),
                "per_sample": {
                    sample_id: {
                        "normalized_action_mse": 0.08,
                        "physical_action_mse": 0.04,
                    }
                    for sample_id in sample_ids
                },
            },
            "conditioning_gates": {
                "prediction_not_constant": True,
                "gripper_metric_finite": True,
                "language_sensitive": True,
                "visual_sensitive": True,
            },
        }
        c67_path = c67_sealed / f"reports/s{step}.json"
        c67_path.write_text(json.dumps(c67_report))
        c67_report_sha[str(step)] = sha(c67_path)

        c69_audit = c69_audits[step // 1_000 - 1]
        preview_audit_path = c69_preview / f"preview-audit/s{step}.json"
        preview_audit = {
            "format": SEAL.PREVIEW_AUDIT_FORMAT,
            "status": "PASS_C69_MILESTONE_PREVIEW_AUDIT",
            "permission": "PREVIEW_EVALUATION_ONLY",
            "effect_status": "NOT_EVIDENCE_READY",
            "milestone": step,
            "checkpoint": str(c69_checkpoint),
            "checkpoint_sha256": c69_checkpoint_sha,
            "training_contract_sha256": c69_contract_sha,
            "milestone_audit": c69_audit,
        }
        preview_audit_path.write_text(json.dumps(preview_audit))
        gates = {
            "prediction_not_constant": step != c69_unsafe_step,
            "gripper_metric_finite": True,
            "language_sensitive": True,
            "visual_sensitive": True,
        }
        c69_execution = dict(execution, seed=c69_execution_seed)
        c69_report = {
            "format": AGG.C69_REPORT_FORMAT,
            "variant": "c69",
            "status": (
                "PASS_FIXED_BALANCED80" if all(gates.values())
                else "FAIL_CONDITIONING_COLLAPSE"
            ),
            "permission": "PREVIEW_ONLY_PENDING_TRAINING_COMPLETE_REBIND",
            "effect_status": "PREVIEW_NOT_EVIDENCE_NOT_FOR_EARLY_STOPPING",
            "milestone": step,
            "checkpoint": str(c69_checkpoint),
            "checkpoint_sha256": c69_checkpoint_sha,
            "restore_audit": str(preview_audit_path.resolve()),
            "restore_audit_sha256": sha(preview_audit_path),
            "training_complete": None,
            "training_complete_sha256": None,
            "training_contract_sha256": c69_contract_sha,
            "data": data,
            "execution": c69_execution,
            "arm": {
                "checkpoint_completed_steps": step,
                "strict_fresh_restore": {"max_abs": 0.0},
                "evaluated_ids_sha256": AGG.SELECTED_IDS_SHA256,
                "metrics": metrics(0.09, 0.045),
                "per_sample": {
                    sample_id: {
                        "normalized_action_mse": 0.09,
                        "physical_action_mse": 0.045,
                    }
                    for sample_id in sample_ids
                },
            },
            "conditioning_gates": gates,
        }
        c69_path = c69_preview / f"reports/s{step}.json"
        c69_path.write_text(json.dumps(c69_report))
        c69_preview_sha[str(step)] = sha(c69_path)

    (c67_sealed / "SEALED.json").write_text(json.dumps({
        "format": "h3wam-c67-sealed-preview-manifest-v1",
        "status": "PASS_C67_PREVIEWS_REBOUND_TO_TRAINING_COMPLETE",
        "permission": "READY_FOR_PREREGISTERED_20_POINT_AGGREGATION_ONLY",
        "effect_status": "NOT_EVIDENCE_READY",
        "training_complete": str(c67_complete_path.resolve()),
        "training_complete_sha256": c67_complete_sha,
        "milestones": list(AGG.MILESTONES),
        "reports_sha256": c67_report_sha,
        "model_reevaluations_during_seal": 0,
    }))
    (c69_preview / "PREVIEWS_COMPLETE.json").write_text(json.dumps({
        "format": SEAL.PREVIEWS_COMPLETE_FORMAT,
        "status": "PASS_C69_ALL_20_PREVIEWS_COMPLETE",
        "permission": "WAIT_FOR_FIXED_C67_VS_C69_AGGREGATION",
        "effect_status": "NOT_EVIDENCE_READY",
        "reports_sha256": c69_preview_sha,
    }))
    SEAL.seal(c69_preview, c69_train, c69_complete_path, c69_sealed)
    return {
        "c67_root": c67_sealed,
        "c67_train": c67_train,
        "c67_complete": c67_complete_path,
        "c69_root": c69_sealed,
        "c69_train": c69_train,
        "c69_complete": c69_complete_path,
    }


def run(paths: dict[str, Path]) -> dict:
    return AGG.aggregate(
        paths["c67_root"], paths["c67_train"], paths["c67_complete"],
        paths["c69_root"], paths["c69_train"], paths["c69_complete"],
    )


def test_c69_previews_seal_without_model_reevaluation(tmp_path: Path):
    paths = fixture(tmp_path)
    manifest = json.loads((paths["c69_root"] / "SEALED.json").read_text())
    assert manifest["model_reevaluations_during_seal"] == 0
    assert len(manifest["reports_sha256"]) == 20
    report = json.loads((paths["c69_root"] / "reports/s20000.json").read_text())
    assert report["training_complete_sha256"] == sha(paths["c69_complete"])
    assert report["preview_provenance"]["rebound_without_model_reevaluation"] is True
    assert Path(report["restore_audit"]).is_file()


def test_c69_sealer_rejects_preview_complete_hash_drift(tmp_path: Path):
    paths = fixture(tmp_path)
    shutil.rmtree(paths["c69_root"])
    preview = tmp_path / "c69-preview"
    marker_path = preview / "PREVIEWS_COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["reports_sha256"]["7000"] = "0" * 64
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="preview report binding"):
        SEAL.seal(
            preview,
            paths["c69_train"],
            paths["c69_complete"],
            paths["c69_root"],
        )


def test_fixed_s20_cross_arm_chain_authorizes_only_paired_libero(tmp_path: Path):
    result = run(fixture(tmp_path))
    assert result["status"] == "PASS_C67_C69_FIXED_S20_ATTRIBUTION_CHAIN"
    assert result["permission"] == "GO_C67_VS_C69_FIXED_S20_PAIRED_LIBERO_ATTRIBUTION"
    assert result["fixed_comparison"] == {
        "c67": "fact_joint_s20000",
        "c69": "action_only_s20000",
        "checkpoint_selection": False,
        "intermediate_milestones_used_for_selection": False,
    }
    paired = result["paired_s20000_c69_action_only_vs_c67_fact_joint"]
    assert paired["physical_action_mse"]["c67_fact_joint_wins"] == 80
    assert "winner" not in result
    assert '"winner":' not in json.dumps(result).lower()
    assert result["contract_attribution"]["historical_c67_defaults_applied"] == [
        "frozen_auxiliary_parameters", "objective_mode",
    ]
    assert all(result["evidence_gates"].values())


def test_conditioning_failure_seals_but_denies_rollout(tmp_path: Path):
    result = run(fixture(tmp_path, c69_unsafe_step=20_000))
    assert result["status"] == "FAIL_C67_C69_CONDITIONING_SAFETY"
    assert result["permission"] == "NO_C67_VS_C69_PAIRED_LIBERO_ATTRIBUTION"
    assert result["evidence_gates"]["c69_all_20_conditioning_safe"] is False


def test_cross_arm_gate_rejects_data_execution_noise_solver_drift(tmp_path: Path):
    paths = fixture(tmp_path, c69_execution_seed=43)
    with pytest.raises(ValueError, match="data/execution/noise/solver identity drift"):
        run(paths)


def test_cross_arm_gate_rejects_non_preregistered_contract_difference(tmp_path: Path):
    paths = fixture(tmp_path, c69_contract_seed=20260817)
    with pytest.raises(ValueError, match="outside the preregistered attribution"):
        run(paths)


def test_only_two_historical_c67_contract_defaults_are_allowed():
    c67, c69 = contract("c67"), contract("c69")
    attribution = AGG._contract_attribution(c67, c69)
    assert attribution["historical_c67_defaults_applied"] == [
        "frozen_auxiliary_parameters", "objective_mode",
    ]
    c67.pop("seed")
    with pytest.raises(ValueError, match="key sets differ"):
        AGG._contract_attribution(c67, c69)


def test_attribution_watcher_waits_for_both_fixed_complete_chains_without_training():
    source = (
        ROOT / "scripts/h3wam/watch_c67_c69_fixed_s20_attribution_gate.sh"
    ).read_text()
    assert '"${c67_sealed_root}/SEALED.json"' in source
    assert '"${c69_preview_root}/PREVIEWS_COMPLETE.json"' in source
    assert '"${c69_complete}"' in source
    assert "launch_c67_c69_fixed_s20_attribution_gate.sh" in source
    assert "train_c56b_fact_online.py" not in source
    assert "rollout_libero" not in source


def test_cross_arm_gate_rehashes_fixed_endpoint_bytes(tmp_path: Path):
    paths = fixture(tmp_path)
    endpoint = paths["c69_train"] / "checkpoints/c69_action_only_s20000.pt"
    endpoint.write_bytes(endpoint.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="fixed s20 endpoint identity failed"):
        run(paths)


def test_dynamic_source_freeze_includes_cross_arm_release_chain():
    source = (ROOT / "scripts/h3wam/freeze_c67_rollout_source.py").read_text()
    assert "scripts/h3wam/seal_c69_milestone_previews.py" in source
    assert "scripts/h3wam/aggregate_c67_c69_fixed_s20_attribution.py" in source
    assert "scripts/h3wam/launch_c67_c69_fixed_s20_attribution_gate.sh" in source
    launcher = (
        ROOT / "scripts/h3wam/launch_c67_c69_fixed_s20_attribution_gate.sh"
    ).read_text()
    assert "torch.distributed.run" not in launcher
    assert "train_c56b_fact_online.py" not in launcher
    assert "rollout_libero" not in launcher
