from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load(
    "_c67_final_evidence_audit_test",
    ROOT / "scripts/h3wam/audit_c67_final_evidence.py",
)
FIXTURE = load(
    "_c67_final_evidence_fixture",
    ROOT / "tests/test_c67_fact_milestone_offline.py",
)


def completed_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    preview, train, complete = FIXTURE.make_preview_fixture(tmp_path)
    training_complete = train / "TRAINING_COMPLETE.json"
    training_complete.write_bytes(complete.read_bytes())
    complete = training_complete
    sealed = tmp_path / "sealed"
    FIXTURE.SEAL.seal(preview, train, complete, sealed)
    result = FIXTURE.MODULE.aggregate(sealed, complete)
    (sealed / "RESULTS.json").write_text(json.dumps(result))
    (train / "reports").mkdir()
    (train / "restore").mkdir()
    for milestone in AUDIT.MILESTONES:
        (train / f"reports/train_s{milestone}.json").write_text(
            json.dumps({"milestone": milestone, "kind": "train"})
        )
        (train / f"restore/restore_s{milestone}.json").write_text(
            json.dumps({"milestone": milestone, "restore_max_abs": 0.0})
        )
    c58_ready = tmp_path / "C58_READY.json"
    c58_ready.write_text("{}")
    monkeypatch.setattr(
        AUDIT.FINAL,
        "finalize",
        lambda _root, _ready: json.loads(complete.read_text()),
    )
    return preview, train, complete, sealed, c58_ready


def test_independent_audit_rehashes_full_chain_without_model_eval_or_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    preview, train, _, sealed, ready = completed_fixture(tmp_path, monkeypatch)
    report = AUDIT.audit(train, preview, sealed, ready)
    assert report["status"] == "PASS_C67_FINAL_EVIDENCE_INDEPENDENTLY_REPRODUCED"
    assert report["permission"] == "READ_ONLY_AUDIT_COMPLETE_NO_ROLLOUT_AUTHORIZATION"
    assert len(report["milestone_identities"]) == 20
    assert report["fixed_endpoints"]["matched_control"]["milestone"] == 10_000
    assert report["fixed_endpoints"]["treatment"]["milestone"] == 20_000
    assert report["checks"]["model_reevaluations"] == 0
    assert report["checks"]["rollouts_started"] == 0


def test_independent_audit_fails_closed_on_checkpoint_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    preview, train, _, sealed, ready = completed_fixture(tmp_path, monkeypatch)
    checkpoint = train / "checkpoints/c67_online_s18000.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="preview audit identity mismatch"):
        AUDIT.audit(train, preview, sealed, ready)


def test_watcher_only_waits_and_audits_from_frozen_source():
    source = (ROOT / "scripts/h3wam/watch_c67_final_evidence_audit.sh").read_text()
    for token in (
        "C67_FINAL_AUDIT_SOURCE_SNAPSHOT:?",
        "C67_FINAL_AUDIT_SOURCE_FREEZE_SHA256:?",
        '--expected-manifest-sha256 "${freeze_sha}"',
        '"${sealed_root}/SEALED.json"',
        '"${sealed_root}/RESULTS.json"',
        "audit_c67_final_evidence.py",
    ):
        assert token in source
    for forbidden in (
        "torch.distributed.run", "train_c56b_fact_online.py", "rollout_libero",
        "evaluate_c67_fact_milestone_balanced80.py",
    ):
        assert forbidden not in source


def test_complete_source_freeze_includes_independent_final_audit_chain():
    source = (ROOT / "scripts/h3wam/freeze_c67_rollout_source.py").read_text()
    assert "scripts/h3wam/audit_c67_final_evidence.py" in source
    assert "scripts/h3wam/watch_c67_final_evidence_audit.sh" in source
