from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


def load_script(name: str):
    path = Path("scripts/h3wam") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FINALIZER = load_script("finalize_c57_lingbot_long5000")
AGGREGATOR = load_script("aggregate_c57_final_libero_canary")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_finalizer_promotes_only_strict_step5000_fixed_heldout(tmp_path: Path, monkeypatch) -> None:
    selected = tmp_path / "selected.jsonl"
    rows = []
    for suite in ("goal", "spatial", "object", "10"):
        for index in range(20):
            rows.append({"suite": suite, "current_id": f"{suite}-{index}", "eval_flow_seed": index})
    selected.write_text("".join(json.dumps(row) + "\n" for row in rows))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "schema": "c57_heldout_eval_plan_v1", "candidate": "C57",
        "control": "D0_frozen_source_checkpoint", "promotion_checkpoint": 5000,
        "checkpoint_milestones": list(range(200, 5001, 200)), "samples": 80,
        "selected_manifest": str(selected.resolve()), "selected_manifest_sha256": sha(selected),
    }, sort_keys=True))
    monkeypatch.setattr(FINALIZER, "EXPECTED_PLAN_SHA256", sha(plan))
    checkpoint = tmp_path / "c57_step05000.pt"
    torch.save({
        "schema_version": 1, "completed_steps": 5000, "model": {"weight": torch.ones(1)},
        "optimizer": {}, "lr_scheduler": {}, "contract": {
            "candidate": "C57", "classification": "action-only-on-frozen-h3-kv",
            "method": "lingbot_persistent_observation_action_kv",
            "sequence_schema": "c57_lingbot_replan8_v1", "replan": 8,
            "observe_every": 4, "world_size": 8,
        },
    }, checkpoint)
    train = tmp_path / "train.json"
    train.write_text(json.dumps({
        "event": "c57_lingbot_persistent_kv_training", "status": "PASS", "gate": "PASS",
        "completed_steps": 5000, "steps": 5000, "world_size": 8,
        "history": [
            {"step": step, "loss": 1.0, "gradient_norm": 1.0, "head_update_max_abs": 1e-6}
            for step in range(1, 5001)
        ],
    }))
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({
        "schema": "c57_paired_heldout_eval_v1", "checkpoint_step": 5000,
        "checkpoint": str(checkpoint.resolve()), "plan": str(plan.resolve()),
        "plan_sha256": sha(plan), "strict_restore": True,
        "strict_restore_details": {"c57_policy_load_state_dict": "strict=True", "all_heldout_forwards_completed": True},
        "sample_count": 80, "gate": "GO_CLOSED_LOOP_CANARY",
        "c57_mean_loss": 0.8, "d0_mean_loss": 1.0,
        "relative_improvement": 0.2, "sample_win_fraction": 0.6,
        "samples": [
            {"current_id": row["current_id"], "flow_seed": row["eval_flow_seed"],
             "c57_loss": 0.8, "d0_loss": 1.0, "c57_minus_d0": -0.2}
            for row in rows
        ],
    }))
    result = FINALIZER.finalize(checkpoint.resolve(), train.resolve(), heldout.resolve(), plan.resolve())
    assert result["permission"] == "GO_FRESH_LIBERO_CANARY"
    assert result["strict_restore"] is True


def test_trace_audit_accepts_commits_and_terminal_tail(tmp_path: Path) -> None:
    log = tmp_path / "policy.log"
    rows = [
        {"event": "c57_persistent_trace", "command": "c57_reset"},
        {"event": "c57_persistent_trace", "command": "predict", "lifecycle": "reset_predict_obs4_commit8"},
        {"event": "c57_persistent_trace", "command": "c57_feedback", "action_count": 4, "committed": False},
        {"event": "c57_persistent_trace", "command": "c57_feedback", "action_count": 8, "committed": True},
        {"event": "c57_persistent_trace", "command": "predict", "lifecycle": "reset_predict_obs4_commit8"},
    ]
    log.write_text("non-json startup\n" + "".join(json.dumps(row) + "\n" for row in rows))
    audit = AGGREGATOR.audit_trace(log)
    assert audit["commit8"] == 1
    assert audit["terminal_tail"] == "predict_without_obs4"


def test_final_launcher_never_creates_a_new_h3_cache() -> None:
    source = Path("scripts/h3wam/launch_c57_final_fresh_libero_canary.sh").read_text()
    assert "extract_" not in source
    assert "v7_dense_h3_cache" in source
    assert "GO_FRESH_LIBERO_CANARY" in source
    assert "--save-video" not in source
    assert "--save-trajectories" not in source
    assert "[t]rain_c56b_fact_online.py" in source
    assert "([c]56|[C]56)" not in source


def test_eval_queue_does_not_treat_c56_watcher_as_training() -> None:
    source = Path("scripts/h3wam/run_c57_heldout_eval_queue.sh").read_text()
    assert "[t]rain_c56b_fact_online.py" in source
    assert "([c]56|[C]56)" not in source
