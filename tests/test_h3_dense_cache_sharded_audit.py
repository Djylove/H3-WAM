from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h3wam/reduce_h3_dense_cache_audits.py"
SPEC = importlib.util.spec_from_file_location("reduce_h3_dense_cache_audits", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_directory_identity_requires_exact_ids_and_no_temporary_files(tmp_path: Path) -> None:
    (tmp_path / "a.pt").write_bytes(b"a")
    (tmp_path / "b.pt").write_bytes(b"b")
    result = MODULE.audit_directory(tmp_path, {"a", "b"})
    assert result["valid"] is True
    assert result["completed_files"] == 2

    (tmp_path / ".a.pt.123.tmp").write_bytes(b"")
    result = MODULE.audit_directory(tmp_path, {"a", "b"})
    assert result["valid"] is False
    assert result["temporary_count"] == 1


def test_shard_aggregate_is_order_stable_and_counts_full_coverage() -> None:
    reports = [
        {
            "audit_shard_index": 1,
            "aggregate_shard_sha256": "b" * 64,
            "audited_rows": 4,
            "expected_rows": 4,
            "total_bytes": 40,
            "tensor_metadata_error_count": 0,
        },
        {
            "audit_shard_index": 0,
            "aggregate_shard_sha256": "a" * 64,
            "audited_rows": 5,
            "expected_rows": 5,
            "total_bytes": 50,
            "tensor_metadata_error_count": 0,
        },
    ]
    first = MODULE.aggregate_reports("kv", reports)
    second = MODULE.aggregate_reports("kv", list(reversed(reports)))
    assert first == second
    assert first["audited_rows"] == 9
    assert first["expected_rows"] == 9
    assert first["total_bytes"] == 90
