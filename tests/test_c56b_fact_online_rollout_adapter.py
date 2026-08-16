from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SERVE = load("c56b_fact_online_serve_test", ROOT / "scripts/h3wam/serve_rollout_policy.py")
ROLLOUT = load("c56b_fact_online_rollout_test", ROOT / "scripts/h3wam/rollout_libero.py")


def paired_ready() -> dict:
    return {
        "format": "h3wam-c56b-fact-online-paired-balanced80-v1",
        "status": "PASS_PAIRED_BALANCED80",
        "permission": "GO_PAIRED_LIBERO",
        "effect_status": "NOT_EVIDENCE_READY",
        "checkpoint_identity": {
            "c60_main_checkpoint_sha256": "a" * 64,
            "c61_matched_checkpoint_sha256": "b" * 64,
        },
        "execution": {"disk_kv_read": False, "disk_kv_write": False},
        "conditioning_gates": {
            "c60_main": {"visual_sensitive": True, "language_sensitive": True},
            "c61_matched": {"visual_sensitive": True, "language_sensitive": True},
        },
        "paired_contract": {
            "main": {
                "causal_failure_dataset_sha256": "c" * 64,
                "causal_failure_observations_sha256": "d" * 64,
            },
            "c61": {
                "causal_failure_dataset_sha256": "e" * 64,
                "causal_failure_observations_sha256": "f" * 64,
            },
        },
    }


@pytest.mark.parametrize("sha,arm", [("a" * 64, "C60_MAIN"), ("b" * 64, "C61_MATCHED")])
def test_paired_ready_routes_exact_checkpoint(sha, arm):
    actual, causal = SERVE._validate_c56b_paired_ready(paired_ready(), sha)
    assert actual == arm
    assert set(causal) == {
        "causal_failure_dataset_sha256", "causal_failure_observations_sha256"
    }


def test_paired_ready_rejects_cache_or_unknown_checkpoint():
    ready = paired_ready()
    ready["execution"]["disk_kv_read"] = True
    with pytest.raises(ValueError, match="paired heldout"):
        SERVE._validate_c56b_paired_ready(ready, "a" * 64)
    with pytest.raises(ValueError, match="paired heldout"):
        SERVE._validate_c56b_paired_ready(paired_ready(), "0" * 64)


def test_rollout_routes_fact_with_paired_gate(tmp_path):
    argv = [
        "rollout", "--policy", "h3_fact_online_int8",
        "--checkpoint", "c56b.pt", "--cache-root", "cache",
        "--policy-python", "python", "--output-dir", "out",
        "--h3-checkpoint", "h3.safetensors", "--h3-model", "h3-model",
        "--dreamwam-source-manifest", "source.jsonl",
        "--c56b-paired-ready", "PAIRED_READY.json",
        "--model-evaluations", "10",
    ]
    with patch.object(sys, "argv", argv):
        args = ROLLOUT.parse_args()
    command = ROLLOUT.policy_command(args, 1234, tmp_path / "ready.json")
    assert "h3_fact_online_int8" in command
    assert command[command.index("--c56b-paired-ready") + 1].endswith(
        "PAIRED_READY.json"
    )


def test_rollout_rejects_missing_paired_gate(tmp_path):
    argv = [
        "rollout", "--policy", "h3_fact_online_int8",
        "--checkpoint", "c56b.pt", "--cache-root", "cache",
        "--policy-python", "python", "--output-dir", "out",
        "--h3-checkpoint", "h3.safetensors", "--h3-model", "h3-model",
        "--dreamwam-source-manifest", "source.jsonl",
        "--model-evaluations", "10",
    ]
    with patch.object(sys, "argv", argv):
        args = ROLLOUT.parse_args()
    with pytest.raises(ValueError, match="c56b-paired-ready"):
        ROLLOUT.policy_command(args, 1234, tmp_path / "ready.json")


def test_adapter_action_path_never_requests_teacher_tracks():
    source = (ROOT / "scripts/h3wam/serve_rollout_policy.py").read_text()
    block = source[source.index("class _H3FACTActionOnlyAdapter"):]
    assert "forward_action" in block
    assert "clean_actions=" not in block
