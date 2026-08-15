from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "c31_consequence_sweep_test_module",
    ROOT / "scripts/h3wam/evaluate_c31_consequence_sweep.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_repeated_temporal_winner_is_selected(tmp_path, monkeypatch):
    for variant, seed, mse in (
        ("flattened", 42, 0.8), ("flattened", 314159, 0.7),
        ("temporal", 42, 0.5), ("temporal", 314159, 0.4),
    ):
        run = tmp_path / f"{variant}_seed{seed}"
        run.mkdir()
        (run / "report.json").write_text(json.dumps({
            "model_variant": variant,
            "status": "PASS_C31_ACTION_CONDITIONED_CONSEQUENCE",
            "source": {"dataset_sha256": "d", "features_sha256": "f"},
            "optimization": {
                "seed": seed, "steps": 10000, "batch_size": 64,
                "learning_rate": 3e-4, "weight_decay": 1e-2,
                "target_error_scaling": "train_delta_std",
                "condition_dropout_prob": 0.1,
                "mechanism_gate": "paired_null",
            },
            "final_metrics": {"mse": {"conditioned_true": mse}},
            "mechanism": {
                "conditioned_gain_over_independent": 0.1,
                "conditioned_gain_over_paired_null": 0.2,
                "conditioned_gain_over_shuffled_train": 0.1,
                "conditioned_within_state_shuffle_degradation": 0.1,
            },
            "checkpoint_sha256": f"checkpoint-{variant}-{seed}",
        }))
    output = tmp_path / "COMPLETED"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_c31_consequence_sweep.py", "--root", str(tmp_path),
        "--output", str(output),
    ])
    MODULE.main()
    result = json.loads(output.read_text())
    assert result["selected_variant"] == "temporal"
    assert result["temporal_beats_flattened_both_seeds"] is True
    assert result["permission"] == "GO_FROZEN_CONSEQUENCE_VALUE_RANKING"
    assert result["optimization_contract"][-3:] == [
        "train_delta_std", 0.1, "paired_null"
    ]
    assert result["variants"]["temporal"]["minimum_gain_over_paired_null"] == 0.2
