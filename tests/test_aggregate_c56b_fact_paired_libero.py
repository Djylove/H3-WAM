from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h3wam/aggregate_c56b_fact_paired_libero.py"
SPEC = importlib.util.spec_from_file_location("_c56b_libero_aggregate_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha(path: Path) -> str:
    return MODULE.sha256_file(path)


def endpoint(tmp_path: Path, arm: str, byte: bytes) -> Path:
    checkpoint = tmp_path / f"{arm}.pt"
    checkpoint.write_bytes(byte)
    ready = tmp_path / f"{arm}.READY.json"
    ready.write_text(json.dumps({
        "status": "PASS_C56B_ONLINE_LONG10000_STRICT_RESTORE",
        "permission": "READY_FOR_PAIRED_HELDOUT",
        "arm": arm,
        "completed_steps": 10000,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha(checkpoint),
        "c58_parent_sha256": "a" * 64,
    }))
    return ready


def result(path: Path, suite: str, checkpoint: Path, arm: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    successes = (
        (lambda task: task % 2 == 0)
        if arm == "c60_main" else (lambda task: task % 3 == 0)
    )
    path.write_text(json.dumps({
        "policy": "h3_fact_online_int8",
        "checkpoint": str(checkpoint.resolve()),
        "suite": suite,
        "task_ids": list(range(10)),
        "trial_indices": [33],
        "trials_per_task": 1,
        "max_steps": 400,
        "wait_steps": 0,
        "replan_steps": 8,
        "action_horizon": 32,
        "model_evaluations": 10,
        "environment_seed": 42,
        "policy_noise_seed_base": 330042,
        "normalized_action_pre_clamp": True,
        "sample_ensemble_size": 1,
        "use_action_ensembler": False,
        "save_trajectories": False,
        "tasks": [
            {
                "task_id": task,
                "episodes": [{"trial": 33, "success": successes(task)}],
            }
            for task in range(10)
        ],
    }))


def fixture(tmp_path: Path):
    main_ready = endpoint(tmp_path, "C60_MAIN", b"main")
    c61_ready = endpoint(tmp_path, "C61_MATCHED", b"c61")
    main = json.loads(main_ready.read_text())
    c61 = json.loads(c61_ready.read_text())
    gate = tmp_path / "PAIRED.json"
    gate.write_text(json.dumps({
        "format": "h3wam-c56b-fact-online-paired-balanced80-v1",
        "status": "PASS_PAIRED_BALANCED80",
        "permission": "GO_PAIRED_LIBERO",
        "checkpoint_identity": {
            "c60_main_ready_sha256": sha(main_ready),
            "c61_matched_ready_sha256": sha(c61_ready),
            "c60_main_checkpoint_sha256": main["checkpoint_sha256"],
            "c61_matched_checkpoint_sha256": c61["checkpoint_sha256"],
        },
    }))
    root = tmp_path / "rollouts"
    for arm, ready in (("c60_main", main), ("c61_matched", c61)):
        for suite in MODULE.SUITES:
            result(
                root / arm / suite / "results.json", suite,
                Path(ready["checkpoint"]), arm,
            )
    return root, gate, main_ready, c61_ready


def test_aggregate_requires_complete_matched_grid(tmp_path):
    root, gate, main_ready, c61_ready = fixture(tmp_path)
    report = MODULE.aggregate(root, gate, main_ready, c61_ready)
    assert report["paired_episodes_per_arm"] == 40
    assert report["main_successes"] == 20
    assert report["c61_successes"] == 16
    assert report["protocol"]["globally_unused_init_state"] is False
    assert report["paired_effect"]["c61_wins"] == 8
    assert report["paired_effect"]["main_wins"] == 12


def test_aggregate_rejects_execution_drift(tmp_path):
    root, gate, main_ready, c61_ready = fixture(tmp_path)
    path = root / "c61_matched/libero_goal/results.json"
    payload = json.loads(path.read_text())
    payload["replan_steps"] = 32
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="rollout contract mismatch"):
        MODULE.aggregate(root, gate, main_ready, c61_ready)


def test_exact_mcnemar_direction():
    assert MODULE.exact_mcnemar(10, 0)["one_sided_p_c61_better"] == pytest.approx(
        1 / 1024
    )
