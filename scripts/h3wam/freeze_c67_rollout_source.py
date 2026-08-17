#!/usr/bin/env python3
"""Create or verify a complete read-only git-tree snapshot for C67 rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import tempfile
from typing import Any


FORMAT = "h3wam-c67-complete-readonly-source-freeze-v1"
MANIFEST_NAME = "SOURCE_FREEZE.json"
DYNAMIC_EXECUTION_FILES = (
    "scripts/h3wam/rollout_libero.py",
    "scripts/h3wam/serve_rollout_policy.py",
    "scripts/h3wam/run_cloud_libero.sh",
    "scripts/h3wam/prepare_c67_budget_rollout.py",
    "scripts/h3wam/launch_c67_budget_paired680_8gpu.sh",
    "scripts/h3wam/aggregate_c67_budget_paired680.py",
    "scripts/h3wam/aggregate_c58b_expanded_paired_eval.py",
    "scripts/h3wam/freeze_c67_rollout_source.py",
    "src/fastwam/models/h3wam/fastwam_full_tower.py",
    "src/fastwam/models/h3wam/starwam_feature_action.py",
    "third_party/FastWAM/src/fastwam/models/wan22/helpers/gradient.py",
    "third_party/FastWAM/src/fastwam/models/wan22/wan_video_dit.py",
    "third_party/FastWAM/src/fastwam/models/wan22/action_dit.py",
    "third_party/StarWAM/starwam/modules/action_dit.py",
    "third_party/StarWAM/starwam/modules/wan_block.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(project: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(project), *args), check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8", errors="strict").strip()


def git_entries(project: Path, commit: str) -> list[dict[str, str]]:
    raw = subprocess.run(
        ("git", "-C", str(project), "ls-tree", "-r", "-z", "--full-tree", commit),
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    entries = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        mode, kind, object_id = metadata.decode().split()
        path = encoded_path.decode("utf-8")
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or kind != "blob":
            raise ValueError(f"unsafe/non-blob tracked entry: {record!r}")
        entries.append({
            "path": path, "mode": mode, "git_type": kind,
            "git_object": object_id,
        })
    if not entries or len({row["path"] for row in entries}) != len(entries):
        raise ValueError("git tree is empty or has duplicate paths")
    return entries


def file_identity(path: Path) -> dict[str, Any]:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        target = os.readlink(path)
        pure = PurePosixPath(target)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"snapshot contains unsafe symlink: {path} -> {target}")
        return {
            "kind": "symlink", "target": target,
            "sha256": hashlib.sha256(target.encode()).hexdigest(),
        }
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"snapshot tracked path is not regular/symlink: {path}")
    return {"kind": "file", "size": info.st_size, "sha256": sha256_file(path)}


def _safe_extract(archive: Path, output: Path) -> None:
    with tarfile.open(archive, "r:") as stream:
        for member in stream.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe git archive member: {member.name}")
        stream.extractall(output, filter="data")


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            continue
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def freeze(project: Path, expected_commit: str, expected_tree: str, output: Path) -> dict:
    project, output = project.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if run_git(project, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source freeze requires a completely clean git worktree")
    commit = run_git(project, "rev-parse", "HEAD")
    tree = run_git(project, "rev-parse", "HEAD^{tree}")
    if commit != expected_commit or tree != expected_tree:
        raise ValueError(f"git identity mismatch: {commit}/{tree}")
    entries = git_entries(project, commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        archive = Path(temporary) / "tree.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                ("git", "-C", str(project), "archive", "--format=tar", commit),
                check=True, stdout=stream,
            )
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        _safe_extract(archive, staging)
        files = {}
        for row in entries:
            identity = file_identity(staging / row["path"])
            files[row["path"]] = {**row, **identity}
        missing_dynamic = sorted(set(DYNAMIC_EXECUTION_FILES) - set(files))
        if missing_dynamic:
            raise ValueError(f"dynamic execution source missing: {missing_dynamic}")
        manifest = {
            "format": FORMAT,
            "status": "PASS_COMPLETE_COMMIT_TREE_DYNAMIC_SOURCE_FREEZE",
            "permission": "READ_ONLY_EXECUTION_SNAPSHOT_ONLY",
            "git_commit": commit,
            "git_tree": tree,
            "tracked_file_count": len(files),
            "tracked_files": files,
            "dynamic_execution_sha256": {
                name: files[name]["sha256"] for name in DYNAMIC_EXECUTION_FILES
            },
            "python_contract": {
                "python_no_user_site": True,
                "python_dont_write_bytecode": True,
                "fastwam_source_root": (
                    "third_party/FastWAM/src/fastwam/models/wan22"
                ),
            },
            "claim_boundary": (
                "Freezes every git-tracked source byte plus all known runtime/dynamic "
                "import targets. Python wheels, model bytes, and datasets are separately "
                "content-gated by the rollout authorization/launcher."
            ),
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
    _make_read_only(output)
    return manifest


def verify(snapshot: Path, expected_manifest_sha256: str | None = None) -> dict:
    snapshot = snapshot.resolve()
    manifest_path = snapshot / MANIFEST_NAME
    if expected_manifest_sha256 is not None and sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("source freeze manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("tracked_files", {})
    if (
        manifest.get("format") != FORMAT
        or manifest.get("status") != "PASS_COMPLETE_COMMIT_TREE_DYNAMIC_SOURCE_FREEZE"
        or manifest.get("permission") != "READ_ONLY_EXECUTION_SNAPSHOT_ONLY"
        or not isinstance(files, dict)
        or manifest.get("tracked_file_count") != len(files)
        or re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("git_commit"))) is None
        or re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("git_tree"))) is None
    ):
        raise ValueError("source freeze manifest contract mismatch")
    actual_paths = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if not path.is_dir() and path.relative_to(snapshot).as_posix() != MANIFEST_NAME
    }
    if actual_paths != set(files):
        raise ValueError("snapshot file set differs from complete git tree")
    for name, expected in files.items():
        path = snapshot / name
        if file_identity(path) != {
            key: expected[key] for key in ("kind", "target", "sha256")
            if key in expected
        } | ({"size": expected["size"]} if "size" in expected else {}):
            raise ValueError(f"snapshot file identity mismatch: {name}")
        if not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ValueError(f"snapshot file is writable: {name}")
    if stat.S_IMODE(snapshot.stat().st_mode) & 0o222:
        raise ValueError("snapshot root is writable")
    if stat.S_IMODE(manifest_path.stat().st_mode) & 0o222:
        raise ValueError("source freeze manifest is writable")
    writable_directories = [
        path for path in snapshot.rglob("*")
        if path.is_dir() and stat.S_IMODE(path.stat().st_mode) & 0o222
    ]
    if writable_directories:
        raise ValueError(f"snapshot directory is writable: {writable_directories[0]}")
    dynamic = manifest.get("dynamic_execution_sha256", {})
    if set(dynamic) != set(DYNAMIC_EXECUTION_FILES):
        raise ValueError("dynamic execution source set mismatch")
    if any(files[name]["sha256"] != digest for name, digest in dynamic.items()):
        raise ValueError("dynamic execution source hash mismatch")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-tree")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    if args.verify:
        if args.snapshot is None or any(
            value is not None for value in (args.project, args.expected_commit, args.expected_tree, args.output)
        ):
            parser.error("--verify requires only --snapshot and optional expected SHA")
        report = verify(args.snapshot, args.expected_manifest_sha256)
    else:
        if any(value is None for value in (
            args.project, args.expected_commit, args.expected_tree, args.output
        )) or args.snapshot is not None or args.expected_manifest_sha256 is not None:
            parser.error("freeze requires project, expected commit/tree and output")
        report = freeze(
            args.project, args.expected_commit, args.expected_tree, args.output
        )
    print(json.dumps({
        "status": report["status"], "git_commit": report["git_commit"],
        "git_tree": report["git_tree"],
        "tracked_file_count": report["tracked_file_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
