#!/usr/bin/env python3
"""Reduce complete cache-shard reports and emit the training READY marker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--kv-subdir", required=True)
    parser.add_argument("--feature-subdir", required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--audit-num-shards", type=int, default=32)
    parser.add_argument("--expected-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--max-error-examples", type=int, default=50)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_reports(args: argparse.Namespace, mode: str) -> tuple[list[dict[str, Any]], list[str]]:
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for index in range(args.audit_num_shards):
        path = args.report_root / f"{mode}_shard{index:02d}.json"
        if not path.is_file():
            errors.append(f"missing report: {path}")
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as error:
            errors.append(f"invalid report {path}: {type(error).__name__}:{error}")
            continue
        reports.append(payload)
        if payload.get("valid") is not True:
            errors.append(f"invalid shard report: {path}")
        expected = {
            "mode": mode,
            "audit_num_shards": args.audit_num_shards,
            "audit_shard_index": index,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                errors.append(f"{path}:{key}={payload.get(key)!r}!={value!r}")
    return reports, errors


def audit_directory(path: Path, expected_ids: set[str]) -> dict[str, Any]:
    completed: set[str] = set()
    temporary: list[str] = []
    if path.is_dir():
        # os.scandir can consume the file type returned by readdir/readdirplus.
        # Path.iterdir() followed by Path.is_file() issues one stat per cache
        # entry, which is prohibitively slow for 200k+ files on shared storage.
        # Do not follow symlinks: any non-regular entry is an audit failure.
        with os.scandir(path) as entries:
            for entry in entries:
                name = entry.name
                regular_pt = (
                    entry.is_file(follow_symlinks=False)
                    and name.endswith(".pt")
                    and not name.startswith(".")
                )
                if regular_pt:
                    completed.add(name[:-3])
                else:
                    temporary.append(name)
    temporary.sort()
    missing = sorted(expected_ids - completed)
    extra = sorted(completed - expected_ids)
    return {
        "completed_files": len(completed),
        "missing_count": len(missing),
        "missing_examples": missing[:50],
        "extra_count": len(extra),
        "extra_examples": extra[:50],
        "temporary_count": len(temporary),
        "temporary_examples": temporary[:50],
        "valid": not missing and not extra and not temporary,
    }


def aggregate_reports(mode: str, reports: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    for report in sorted(reports, key=lambda item: item["audit_shard_index"]):
        digest.update(mode.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(report["audit_shard_index"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(report["aggregate_shard_sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "reports": len(reports),
        "audited_rows": sum(int(item.get("audited_rows", 0)) for item in reports),
        "expected_rows": sum(int(item.get("expected_rows", 0)) for item in reports),
        "total_bytes": sum(int(item.get("total_bytes", 0)) for item in reports),
        "tensor_metadata_error_count": sum(
            int(item.get("tensor_metadata_error_count", 0)) for item in reports
        ),
        "aggregate_cache_sha256": digest.hexdigest(),
    }


def main() -> None:
    args = parse_args()
    if args.audit_num_shards <= 0 or args.max_error_examples <= 0:
        raise ValueError("positive shard and error limits are required")
    manifest = args.manifest.resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    duplicate_ids = len(ids) - len(set(ids))
    expected_ids = set(ids)
    manifest_sha256 = sha256_file(manifest)
    checkpoint = args.expected_checkpoint.resolve()
    checkpoint_sha256 = sha256_file(checkpoint)

    kv_reports, errors = load_reports(args, "kv")
    star_reports, star_errors = load_reports(args, "star")
    errors.extend(star_errors)
    for report in [*kv_reports, *star_reports]:
        if report.get("manifest_sha256") != manifest_sha256:
            errors.append("shard report manifest hash mismatch")
        if report.get("manifest_rows") != len(rows):
            errors.append("shard report manifest row count mismatch")
        if report.get("expected_checkpoint_sha256") != args.expected_checkpoint_sha256:
            errors.append("shard report checkpoint expectation mismatch")

    kv = aggregate_reports("kv", kv_reports)
    star = aggregate_reports("star", star_reports)
    kv_dir = audit_directory(args.cache_root.resolve() / args.kv_subdir, expected_ids)
    star_dir = audit_directory(args.cache_root.resolve() / args.feature_subdir, expected_ids)
    if duplicate_ids:
        errors.append(f"manifest has {duplicate_ids} duplicate IDs")
    if kv["audited_rows"] != len(rows) or kv["expected_rows"] != len(rows):
        errors.append("K/V shard coverage does not equal the manifest")
    if star["audited_rows"] != len(rows) or star["expected_rows"] != len(rows):
        errors.append("StarWAM shard coverage does not equal the manifest")
    if not kv_dir["valid"]:
        errors.append("K/V directory identity audit failed")
    if not star_dir["valid"]:
        errors.append("StarWAM directory identity audit failed")
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        errors.append("H3 checkpoint SHA256 mismatch")

    valid = not errors
    result = {
        "valid": valid,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "manifest_rows": len(rows),
        "manifest_sha256": manifest_sha256,
        "duplicate_manifest_ids": duplicate_ids,
        "audit_num_shards": args.audit_num_shards,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
        "dreamwam_kv": {**kv, "directory": kv_dir},
        "starwam_feature": {**star, "directory": star_dir},
        "errors": errors[: args.max_error_examples],
    }
    atomic_json(args.output, result)
    if not valid:
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(1)
    ready = {
        "ready": True,
        "created_at": result["created_at"],
        "combined_audit": str(args.output.resolve()),
        "dreamwam_kv_audit": str(args.report_root.resolve()),
        "dreamwam_kv_aggregate_sha256": kv["aggregate_cache_sha256"],
        "starwam_feature_audit": str(args.report_root.resolve()),
        "starwam_feature_aggregate_sha256": star["aggregate_cache_sha256"],
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    atomic_json(args.ready_marker, ready)
    print(json.dumps(ready, sort_keys=True))


if __name__ == "__main__":
    main()
