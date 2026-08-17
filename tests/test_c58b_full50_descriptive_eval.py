import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(ROOT / "scripts/h3wam"))
PREPARE = load(
    "c58_full50_prepare_test",
    ROOT / "scripts/h3wam/prepare_c58b_full50_descriptive_eval.py",
)
AGGREGATE = load(
    "c58_full50_aggregate_test",
    ROOT / "scripts/h3wam/aggregate_c58b_full50_descriptive_eval.py",
)


def test_prepare_freezes_exact_two_arm_trials0_32_grid(tmp_path, monkeypatch):
    c58 = tmp_path / "c58.pt"; c58.write_bytes(b"c58")
    d0 = tmp_path / "d0.pt"; d0.write_bytes(b"d0")
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "permission": "GO_FRESH_LIBERO", "checkpoint": str(c58),
        "checkpoint_sha256": PREPARE.C58_SHA256,
    }))
    evidence = tmp_path / "PAIR_EVIDENCE.jsonl"; evidence.write_text("evidence\n")
    final = tmp_path / "FINAL.json"
    final.write_text(json.dumps({
        "status": "PASS_C58B_EXPANDED_PAIRED", "effect_status": "EVIDENCE_READY",
        "pairs": 680, "pair_evidence": str(evidence),
        "pair_evidence_sha256": PREPARE.CONFIRMATORY_EVIDENCE_SHA256,
        "gates": {"gate": True},
    }))
    snapshot_root = tmp_path / "snapshot"
    snapshot_hashes = {}
    for name, relative in {
        "preparer": "scripts/h3wam/prepare_c58b_full50_descriptive_eval.py",
        "launcher": "scripts/h3wam/launch_c58b_full50_descriptive_arm.sh",
        "aggregator": "scripts/h3wam/aggregate_c58b_full50_descriptive_eval.py",
        "finalizer": "scripts/h3wam/watch_c58b_full50_descriptive_finalizer.sh",
        "starwam_wan_block": "third_party/StarWAM/starwam/modules/wan_block.py",
    }.items():
        path = snapshot_root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name); path.chmod(0o444)
        snapshot_hashes[name] = PREPARE.sha256_file(path)
    snapshot = snapshot_root / "FULL50_SNAPSHOT.json"
    snapshot.write_text(json.dumps({
        "format": "h3wam-c58b-full50-runtime-snapshot-v1",
        "status": "VERIFIED_FOR_READ_ONLY_FREEZE", "source_commit": "test",
        "hashes": snapshot_hashes,
    }))
    real_sha = PREPARE.sha256_file
    expected = {
        c58: PREPARE.C58_SHA256, d0: PREPARE.D0_SHA256,
        final: PREPARE.CONFIRMATORY_FINAL_SHA256,
    }
    monkeypatch.setattr(PREPARE, "sha256_file", lambda path: expected.get(Path(path), real_sha(Path(path))))
    root = tmp_path / "full50"
    report = PREPARE.prepare(c58, d0, gate, final, snapshot, root)
    jobs = [json.loads(line) for line in (root / "jobs.jsonl").read_text().splitlines()]
    identities = {(j["arm"], j["suite"], j["task"], j["trial"]) for j in jobs}
    assert report["permission"] == "GO_DESCRIPTIVE_FULL50_SUPPLEMENT_NO_PROMOTION"
    assert len(jobs) == len(identities) == 2640
    assert sum(j["arm"] == "candidate_c58b" for j in jobs) == 1320
    assert sum(j["arm"] == "control_d0" for j in jobs) == 1320
    assert {j["trial"] for j in jobs} == set(range(33))
    assert all(j["episodes"] == 1 for j in jobs)
    assert all(j["node"] == ("n0" if j["arm"] == "candidate_c58b" else "n3") for j in jobs)
    assert {j["gpu"] for j in jobs} == set(range(8))


def test_launcher_pins_fresh_process_and_action_contract():
    source = (ROOT / "scripts/h3wam/launch_c58b_full50_descriptive_arm.sh").read_text()
    assert '--task-ids "${task}"' in source
    assert '--trial-indices "${trial}"' in source
    assert "--wait-steps 30" in source
    assert "--replan-steps 8" in source
    assert "--action-horizon 32" in source
    assert "--model-evaluations 10" in source
    assert "--environment-seed" not in source
    assert "--policy-noise-seed-base" not in source
    assert "for gpu in 0 1 2 3 4 5 6 7" in source


def test_aggregator_hard_codes_descriptive_boundary_and_2000_pairs():
    source = (ROOT / "scripts/h3wam/aggregate_c58b_full50_descriptive_eval.py").read_text()
    assert '"effect_status": "DESCRIPTIVE_ONLY"' in source
    assert '"permission": "NO_NEW_PROMOTION_CLAIM"' in source
    assert "len(pairs) != 2000" in source
    assert "Trials0..32 were previously consumed" in source
    assert AGGREGATE.CONFIRMATORY_FINAL_SHA256 == PREPARE.CONFIRMATORY_FINAL_SHA256


def test_finalizer_waits_for_both_complete_markers():
    source = (ROOT / "scripts/h3wam/watch_c58b_full50_descriptive_finalizer.sh").read_text()
    wait = source.index('while [[ ! -s "${candidate}" || ! -s "${control}" ]]')
    aggregate = source.index('"${python_bin}" "${script_root}/aggregate_c58b_full50_descriptive_eval.py"')
    assert wait < aggregate
