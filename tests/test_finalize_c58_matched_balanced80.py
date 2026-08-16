from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/h3wam/finalize_c58_matched_balanced80.py"
    spec = importlib.util.spec_from_file_location("_test_c58_balanced_finalizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def payload(module, *, candidate, step, norm, physical, gripper, language, visual):
    return {
        "candidate": candidate,
        "checkpoint": {
            "completed_steps": step,
            "fresh_restore": {"max_abs": 0.0},
            "contract": {
                "candidate": candidate,
                "d0_parent_optimizer_restored": False,
            },
        },
        "protocol_identity": {
            "actual_selected_ids_sha256": module.SELECTED_IDS_SHA256,
        },
        "metrics": {
            "normalized_clip5_model_domain": {"action_mse": norm},
            "denormalized_official_minmax_clamp": {"action_mse": physical},
            "gripper_sign": {"macro_f1": gripper},
            "language_replacement_sensitivity": {
                "mean_abs_prediction_delta": language,
            },
            "visual_feature_shuffle": {
                "baseline_vs_shuffle_action_delta": {
                    "normalized_model_domain": {"action_mse": visual},
                }
            },
        },
    }


def test_pre_registered_gate_accepts_only_joint_action_and_sensitivity_win():
    module = load_module()
    control = payload(
        module,
        candidate="C58_MATCHED_D0_FRESH_OPTIMIZER",
        step=1000,
        norm=0.1,
        physical=0.04,
        gripper=0.8,
        language=0.2,
        visual=0.08,
    )
    candidate = payload(
        module,
        candidate="C58_FASTWAM_FULL30_H3_LAYER49",
        step=1000,
        norm=0.098,
        physical=0.039,
        gripper=0.797,
        language=0.195,
        visual=0.078,
    )
    result = module.summarize_pair(candidate, control, 1000)
    assert result["eligible_for_fresh_closed_loop"] is True
    candidate["metrics"]["language_replacement_sensitivity"][
        "mean_abs_prediction_delta"
    ] = 0.18
    result = module.summarize_pair(candidate, control, 1000)
    assert result["eligible_for_fresh_closed_loop"] is False


def test_identity_gate_rejects_nonfresh_or_wrong_selection():
    module = load_module()
    item = payload(
        module,
        candidate="C58_FASTWAM_FULL30_H3_LAYER49",
        step=1000,
        norm=0.1,
        physical=0.04,
        gripper=0.8,
        language=0.2,
        visual=0.08,
    )
    item["checkpoint"]["fresh_restore"]["max_abs"] = 0.1
    try:
        module.require_identity(
            item, candidate="C58_FASTWAM_FULL30_H3_LAYER49", step=1000
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-exact restore passed")
