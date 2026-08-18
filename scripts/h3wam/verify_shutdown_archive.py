#!/usr/bin/env python3
"""Verify the small local shutdown archive and any optional checkpoint directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/archive/h3_wam_shutdown_manifest_2026-08-18.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Optional directory containing the archive checkpoint basenames.",
    )
    parser.add_argument("--require-checkpoints", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    errors: list[str] = []
    checked = 0

    if manifest.get("format") != "h3-wam-shutdown-archive-v1":
        errors.append("unexpected manifest format")

    for bundle in manifest["archives"].values():
        if not isinstance(bundle, dict) or "local_path" not in bundle:
            continue
        path = args.repo_root / bundle["local_path"]
        if not path.exists():
            print(f"MISSING_OPTIONAL bundle {path}")
            continue
        if path.stat().st_size != bundle["size_bytes"]:
            errors.append(f"size mismatch: {path}")
            continue
        actual = sha256(path)
        if actual != bundle["sha256"]:
            errors.append(f"sha256 mismatch: {path}")
            continue
        checked += 1
        print(f"PASS bundle {path} {actual}")

    checkpoint_root = args.checkpoint_root
    for checkpoint in manifest["checkpoints"]:
        basename = Path(checkpoint["archive_path"]).name
        path = checkpoint_root / basename if checkpoint_root else None
        if path is None or not path.exists():
            label = "MISSING_REQUIRED" if args.require_checkpoints else "MISSING_OPTIONAL"
            print(f"{label} checkpoint {basename}")
            if args.require_checkpoints:
                errors.append(f"missing checkpoint: {basename}")
            continue
        if path.stat().st_size != checkpoint["size_bytes"]:
            errors.append(f"size mismatch: {path}")
            continue
        actual = sha256(path)
        if actual != checkpoint["sha256"]:
            errors.append(f"sha256 mismatch: {path}")
            continue
        checked += 1
        print(f"PASS checkpoint {path} {actual}")

    if errors:
        for error in errors:
            print(f"FAIL {error}", file=sys.stderr)
        return 1
    print(f"PASS manifest checked_artifacts={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
