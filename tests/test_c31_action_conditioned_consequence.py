from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "c31_action_conditioned_consequence_test_module",
    ROOT / "scripts/h3wam/train_c31_action_conditioned_consequence.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_two_step_temporal_training_writes_audited_checkpoints(tmp_path, monkeypatch):
    states = []
    branches = []
    for group in range(3):
        split = "train" if group < 2 else "val"
        states.append({
            "group_id": group,
            "split": split,
            "consequence_split": "train" if group == 0 else (
                "validation" if group == 1 else "reserved_ranking_val"
            ),
            "suite": "libero_goal",
            "source_episode": f"source-{group}",
            "proprio": torch.linspace(0, 1, 8) + group,
        })
        for candidate in range(4):
            ordinal = group * 4 + candidate
            actions = torch.full((32, 7), candidate / 4)
            pad = torch.zeros(32, dtype=torch.bool)
            if candidate == 0:
                actions[5:] = 0
                pad[5:] = True
            branches.append({
                "ordinal": ordinal,
                "group_id": group,
                "split": split,
                "consequence_split": states[-1]["consequence_split"],
                "environment_actions": actions,
                "action_is_pad": pad,
            })
    dataset = tmp_path / "dataset.pt"
    torch.save({
        "format": MODULE.DATA_FORMAT,
        "states": states,
        "branches": branches,
        "audit": {
            "partial_action_branches": 3,
            "fresh_ranking_validation_sources": 1,
        },
    }, dataset)
    features = tmp_path / "features.pt"
    torch.save({
        "format": MODULE.FEATURE_FORMAT,
        "dataset_sha256": digest(dataset),
        "sample_kinds": ["current"] * 3 + ["future"] * 12,
        "sample_indices": torch.tensor(list(range(3)) + list(range(12))),
        "fact_layer49_hidden": torch.randn(15, 1, 2, 5376, dtype=torch.bfloat16),
    }, features)
    report = tmp_path / "report.json"
    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setattr(sys, "argv", [
        "train_c31_action_conditioned_consequence.py",
        "--dataset", str(dataset),
        "--features", str(features),
        "--output", str(report),
        "--checkpoint-dir", str(checkpoint_dir),
        "--model-variant", "temporal",
        "--steps", "2",
        "--save-every", "1",
        "--batch-size", "4",
        "--target-dim", "4",
        "--hidden-dim", "8",
        "--num-heads", "2",
        "--target-error-scaling", "train_delta_std",
        "--device", "cpu",
    ])
    MODULE.main()
    result = json.loads(report.read_text())
    assert result["data"]["source_overlap"] == 0
    assert result["data"]["partial_action_branches"] == 3
    assert result["data"]["reserved_ranking_validation_sources"] == 1
    assert result["optimization"]["effective_train_examples"] == 8
    assert result["optimization"]["target_error_scaling"] == "train_delta_std"
    checkpoint = torch.load(
        checkpoint_dir / "temporal_seed42_step00002.pt", weights_only=False
    )
    assert checkpoint["completed_steps"] == 2
    assert checkpoint["contract"]["unexecuted_action_tail_zero_masked"] is True
    assert checkpoint["normalization"]["target_error_scaling"] == "train_delta_std"
