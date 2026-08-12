from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "h3wam"
    / "train_h3_wam_joint_fsdp.py"
)
SPEC = importlib.util.spec_from_file_location("train_h3_wam_joint_fsdp", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cosine_with_warmup_factor() -> None:
    factor = MODULE.cosine_with_warmup_factor

    assert factor(step=0, warmup_steps=2, total_steps=10, minimum_ratio=0.1) == 0.5
    assert factor(step=1, warmup_steps=2, total_steps=10, minimum_ratio=0.1) == 1.0
    assert factor(step=2, warmup_steps=2, total_steps=10, minimum_ratio=0.1) == 1.0
    assert factor(step=10, warmup_steps=2, total_steps=10, minimum_ratio=0.1) == pytest.approx(0.1)


def test_checkpoint_steps_only_accepts_complete_checkpoint_names(tmp_path: Path) -> None:
    (tmp_path / "step000100").mkdir()
    (tmp_path / "step000020").mkdir()
    (tmp_path / "step000030.partial").mkdir()
    (tmp_path / "not-a-checkpoint").mkdir()

    assert MODULE.checkpoint_steps(tmp_path) == [
        (20, tmp_path / "step000020"),
        (100, tmp_path / "step000100"),
    ]


def test_training_manifest_supports_shared_context_id(tmp_path: Path) -> None:
    (tmp_path / "windows").mkdir()
    (tmp_path / "contexts").mkdir()
    torch.save({}, tmp_path / "windows" / "window_a.pt")
    torch.save({}, tmp_path / "windows" / "window_b.pt")
    torch.save({}, tmp_path / "contexts" / "task_a.pt")
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"id": "window_a", "context_id": "task_a"},
        {"id": "window_b", "context_id": "task_a"},
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    assert MODULE.load_training_rows(tmp_path, manifest, None) == rows


def test_state_projection_migration_preserves_proprio_and_moves_phase() -> None:
    saved = torch.arange(18, dtype=torch.float32).reshape(2, 9)
    target = torch.full((2, 16), 99.0)

    migrated = MODULE.migrate_state_projection(
        saved,
        target,
        saved_previous_action=False,
        saved_phase=True,
        target_previous_action=True,
        target_phase=True,
    )

    assert torch.equal(migrated[:, :8], saved[:, :8])
    assert torch.equal(migrated[:, 8:15], torch.zeros(2, 7))
    assert torch.equal(migrated[:, 15], saved[:, 8])


def test_action_initialization_preserves_existing_history_gate() -> None:
    saved = {
        "state_projection.weight": torch.randn(2, 9),
        "history_gate": torch.full((2, 4), 0.25),
    }
    current = {
        "state_projection.weight": torch.zeros(2, 9),
        "history_gate": torch.zeros(2, 4),
    }

    migrated = MODULE.migrate_action_initialization_state(
        saved,
        current,
        saved_previous_action=False,
        saved_phase=True,
        target_previous_action=False,
        target_phase=True,
        target_history=True,
    )

    assert torch.equal(migrated["history_gate"], saved["history_gate"])


def test_action_initialization_zero_initializes_new_history_gate() -> None:
    saved = {"state_projection.weight": torch.randn(2, 9)}
    current = {
        "state_projection.weight": torch.zeros(2, 9),
        "history_gate": torch.zeros(2, 4),
    }

    migrated = MODULE.migrate_action_initialization_state(
        saved,
        current,
        saved_previous_action=False,
        saved_phase=True,
        target_previous_action=False,
        target_phase=True,
        target_history=True,
    )

    assert torch.equal(migrated["history_gate"], current["history_gate"])


def test_action_initialization_adds_adapter_without_resetting_gate() -> None:
    saved = {
        "state_projection.weight": torch.randn(2, 9),
        "history_gate": torch.full((2, 4), 0.25),
    }
    current = {
        "state_projection.weight": torch.zeros(2, 9),
        "history_gate": torch.zeros(2, 4),
        "history_down.0.weight": torch.randn(2, 4),
        "history_up.0.weight": torch.zeros(4, 2),
    }

    migrated = MODULE.migrate_action_initialization_state(
        saved,
        current,
        saved_previous_action=False,
        saved_phase=True,
        target_previous_action=False,
        target_phase=True,
        target_history=True,
        target_history_adapter=True,
    )

    assert torch.equal(migrated["history_gate"], saved["history_gate"])
    assert torch.equal(
        migrated["history_down.0.weight"], current["history_down.0.weight"]
    )
    assert torch.equal(
        migrated["history_up.0.weight"], current["history_up.0.weight"]
    )
