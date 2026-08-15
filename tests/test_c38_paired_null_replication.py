from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "c38_replication_test_module",
    ROOT / "scripts/h3wam/evaluate_c38_paired_null_replication.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_four_new_seed_passes_release_fresh_ranking(tmp_path, monkeypatch):
    for seed in sorted(MODULE.EXPECTED_SEEDS):
        run = tmp_path / f"temporal_seed{seed}"
        run.mkdir()
        (run / "report.json").write_text(json.dumps({
            "model_variant": "temporal",
            "status": "PASS_C31_ACTION_CONDITIONED_CONSEQUENCE",
            "source": {"dataset_sha256": "d", "features_sha256": "f"},
            "optimization": {
                "seed": seed, "steps": 10000, "batch_size": 64,
                "learning_rate": 3e-4, "weight_decay": 1e-2,
                "target_error_scaling": "raw", "condition_dropout_prob": 0.0,
                "mechanism_gate": "paired_null",
            },
            "mechanism": {
                "conditioned_gain_over_paired_null": 0.1,
                "conditioned_within_state_shuffle_degradation": 0.05,
                "conditioned_gain_over_shuffled_train": 0.2,
            },
            "checkpoint_sha256": f"checkpoint-{seed}",
        }))
    output = tmp_path / "COMPLETED"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_c38_paired_null_replication.py", "--root", str(tmp_path),
        "--output", str(output),
    ])
    MODULE.main()
    result = json.loads(output.read_text())
    assert result["status"] == "PASS_C38_FOUR_SEED_PAIRED_NULL"
    assert result["permission"] == "GO_FRESH_RANKING_VALIDATION"
    assert len(result["runs"]) == 4
