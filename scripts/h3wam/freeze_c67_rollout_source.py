#!/usr/bin/env python3
"""Create or verify a complete read-only git-tree snapshot for C67 rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from typing import Any


FORMAT = "h3wam-c67-complete-readonly-source-freeze-v2"
MANIFEST_NAME = "SOURCE_FREEZE.json"
PINNED_GIT_REPOSITORIES = {
    "third_party/StarWAM": {
        "commit": "cd76d96f273f81e228a05f40f9697fe2514e2356",
        "source": "superproject_gitlink",
    },
    "third_party/FastWAM": {
        "commit": "45d8e1458921d83f8ad6cf9ce993d371208dabd0",
        "source": "ignored_external_git_repository",
    },
    "third_party/FACT": {
        "commit": "618a6c16868699b6d4138941de6a863589ac00dd",
        "source": "ignored_external_git_repository",
    },
    "third_party/Light-WAM": {
        "commit": "b2785f66e13fd9987e94ae1ecc1c441d5059c9ae",
        "source": "ignored_external_git_repository",
    },
}
PINNED_DIRECTORY_SOURCES = {
    "third_party/diffusers_h3": "src/diffusers/__init__.py",
}
DYNAMIC_EXECUTION_FILES = (
    "scripts/h3wam/rollout_libero.py",
    "scripts/h3wam/serve_rollout_policy.py",
    "scripts/h3wam/run_cloud_libero.sh",
    "scripts/h3wam/prepare_c67_budget_rollout.py",
    "scripts/h3wam/launch_c67_budget_paired680_8gpu.sh",
    "scripts/h3wam/aggregate_c67_budget_paired680.py",
    "scripts/h3wam/prepare_c67_c69_attribution_rollout.py",
    "scripts/h3wam/launch_c67_c69_attribution_paired680_shard.sh",
    "scripts/h3wam/finalize_c67_c69_attribution_rollout.py",
    "scripts/h3wam/aggregate_c67_c69_attribution_paired680.py",
    "scripts/h3wam/aggregate_c58b_expanded_paired_eval.py",
    "scripts/h3wam/freeze_c67_rollout_source.py",
    "scripts/h3wam/launch_c67_c60_budget_ablation_20k_8gpu.sh",
    "scripts/h3wam/launch_c69_matched_action_only_canary_8gpu.sh",
    "scripts/h3wam/launch_c69_matched_action_only_20k_8gpu.sh",
    "scripts/h3wam/launch_c72_action_only_one_expert_epoch_canary_8gpu.sh",
    "scripts/h3wam/launch_c72_action_only_one_expert_epoch_30195_8gpu.sh",
    "scripts/h3wam/finalize_c72_action_only_one_expert_epoch.py",
    "scripts/h3wam/prepare_c72_milestone_preview_audit.py",
    "scripts/h3wam/launch_c72_action_only_milestone_preview_queue.sh",
    "scripts/h3wam/launch_c70_sampler_coverage_canary_8gpu.sh",
    "scripts/h3wam/launch_c70_sampler_coverage_20k_8gpu.sh",
    "scripts/h3wam/finalize_c70_sampler_coverage_20k.py",
    "scripts/h3wam/prepare_c70_milestone_preview_audit.py",
    "scripts/h3wam/launch_c70_sampler_milestone_preview_queue.sh",
    "scripts/h3wam/seal_c70_milestone_previews.py",
    "scripts/h3wam/aggregate_c70_c67_fixed_s20.py",
    "scripts/h3wam/watch_c70_final_offline_gate.sh",
    "scripts/h3wam/finalize_c69_matched_action_only_20k.py",
    "scripts/h3wam/prepare_c69_milestone_preview_audit.py",
    "scripts/h3wam/launch_c69_action_only_milestone_preview_queue.sh",
    "scripts/h3wam/seal_c69_milestone_previews.py",
    "scripts/h3wam/aggregate_c67_c69_fixed_s20_attribution.py",
    "scripts/h3wam/launch_c67_c69_fixed_s20_attribution_gate.sh",
    "scripts/h3wam/watch_c67_c69_fixed_s20_attribution_gate.sh",
    "scripts/h3wam/train_c56b_fact_online.py",
    "scripts/h3wam/finalize_c67_c60_budget_ablation_20k.py",
    "scripts/h3wam/prepare_c67_milestone_preview_audit.py",
    "scripts/h3wam/evaluate_c67_fact_milestone_balanced80.py",
    "scripts/h3wam/seal_c67_milestone_previews.py",
    "scripts/h3wam/aggregate_c67_fact_milestone_balanced80.py",
    "scripts/h3wam/launch_c67_fact_milestone_preview_queue.sh",
    "scripts/h3wam/audit_c67_final_evidence.py",
    "scripts/h3wam/watch_c67_final_evidence_audit.sh",
    "scripts/h3wam/probe_c56b_fact_online.py",
    "scripts/h3wam/probe_c71_lightwam_online.py",
    "scripts/h3wam/launch_c71_lightwam_online_probe.sh",
    "scripts/h3wam/train_c71_lightwam_online.py",
    "scripts/h3wam/launch_c71_lightwam_online_canary.sh",
    "scripts/h3wam/evaluate_c71_lightwam_balanced80.py",
    "scripts/h3wam/launch_c71_lightwam_online_long10000.sh",
    "scripts/h3wam/watch_c71_lightwam_milestone_evals.sh",
    "scripts/h3wam/fit_c56b_fact_online_target_norm.py",
    "src/fastwam/models/h3wam/fact_layerwise_tower.py",
    "src/fastwam/models/h3wam/fact_online_data.py",
    "src/fastwam/models/h3wam/c58_online_training.py",
    "src/fastwam/models/h3wam/fastwam_full_tower.py",
    "src/fastwam/models/h3wam/starwam_feature_action.py",
    "src/fastwam/models/h3wam/lightwam_state_fusion.py",
    "third_party/FastWAM/src/fastwam/models/wan22/helpers/gradient.py",
    "third_party/FastWAM/src/fastwam/models/wan22/wan_video_dit.py",
    "third_party/FastWAM/src/fastwam/models/wan22/action_dit.py",
    "third_party/StarWAM/starwam/modules/action_dit.py",
    "third_party/StarWAM/starwam/modules/wan_block.py",
    "third_party/FACT/world_action_model/trainer/wa_casual_trainer.py",
    "third_party/Light-WAM/src/lightwam/models/wan22/state_fusion_action_expert.py",
    "third_party/diffusers_h3/src/diffusers/__init__.py",
    "third_party/diffusers_h3/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py",
    "third_party/diffusers_h3/src/diffusers/modular_pipelines/minimax_h3/encoders.py",
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
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or kind not in {"blob", "commit"}
            or (kind == "commit" and mode != "160000")
        ):
            raise ValueError(f"unsafe tracked entry: {record!r}")
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


def _archive_git_tree(repository: Path, commit: str, output: Path) -> None:
    """Extract exactly one immutable git object tree, never worktree bytes."""

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar", dir=output.parent) as stream:
        subprocess.run(
            ("git", "-C", str(repository), "archive", "--format=tar", commit),
            check=True, stdout=stream,
        )
        stream.flush()
        _safe_extract(Path(stream.name), output)


def _copy_directory_source(source: Path, output: Path) -> dict[str, dict[str, Any]]:
    """Copy a non-git source tree while proving the source did not move."""

    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"directory source is missing: {source}")
    before: dict[str, dict[str, Any]] = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if (
            (path.is_dir() and not path.is_symlink())
            or path.name.endswith((".pyc", ".pyo"))
        ):
            continue
        before[relative.as_posix()] = file_identity(path)
    if not before:
        raise ValueError(f"directory source is empty: {source}")
    for name, identity in before.items():
        source_path, target = source / name, output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if identity["kind"] == "symlink":
            target.symlink_to(identity["target"])
        else:
            shutil.copyfile(source_path, target, follow_symlinks=False)
    after_source = {name: file_identity(source / name) for name in before}
    copied = {name: file_identity(output / name) for name in before}
    if before != after_source or before != copied:
        raise ValueError(f"directory source changed during freeze: {source}")
    return copied


def _content_tree_sha256(files: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            continue
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def freeze(
    project: Path,
    expected_commit: str,
    expected_tree: str,
    output: Path,
    *,
    diffusers_h3_source: Path | None = None,
) -> dict:
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
    blobs = [row for row in entries if row["git_type"] == "blob"]
    gitlinks = {
        row["path"]: row for row in entries if row["git_type"] == "commit"
    }
    expected_gitlinks = {
        name for name, fixed in PINNED_GIT_REPOSITORIES.items()
        if fixed["source"] == "superproject_gitlink"
    }
    if set(gitlinks) != expected_gitlinks:
        raise ValueError(
            f"superproject gitlink set mismatch: {sorted(gitlinks)}"
        )
    for name, row in gitlinks.items():
        if row["git_object"] != PINNED_GIT_REPOSITORIES[name]["commit"]:
            raise ValueError(f"pinned gitlink commit mismatch: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        _archive_git_tree(project, commit, staging)
        files: dict[str, dict[str, Any]] = {}
        for row in blobs:
            identity = file_identity(staging / row["path"])
            files[row["path"]] = {
                **row, **identity, "repository": "superproject",
                "repository_relative_path": row["path"],
            }
        repositories = {}
        for prefix, fixed in PINNED_GIT_REPOSITORIES.items():
            repository = project / prefix
            if not (repository / ".git").exists():
                raise ValueError(f"pinned repository is unavailable: {repository}")
            repository_head = run_git(repository, "rev-parse", "HEAD")
            if repository_head != fixed["commit"]:
                raise ValueError(f"pinned repository HEAD mismatch: {prefix}")
            if fixed["source"] == "superproject_gitlink" and run_git(
                repository, "status", "--porcelain", "--untracked-files=all"
            ):
                raise ValueError(f"pinned submodule worktree is dirty: {prefix}")
            repository_tree = run_git(repository, "rev-parse", f"{fixed['commit']}^{{tree}}")
            repository_entries = git_entries(repository, fixed["commit"])
            if any(row["git_type"] != "blob" for row in repository_entries):
                raise ValueError(f"nested gitlink is unsupported: {prefix}")
            destination = staging / prefix
            _archive_git_tree(repository, fixed["commit"], destination)
            for row in repository_entries:
                snapshot_name = f"{prefix}/{row['path']}"
                if snapshot_name in files:
                    raise ValueError(f"snapshot repository collision: {snapshot_name}")
                files[snapshot_name] = {
                    **row, **file_identity(staging / snapshot_name),
                    "path": snapshot_name, "repository": prefix,
                    "repository_relative_path": row["path"],
                }
            repositories[prefix] = {
                "source": fixed["source"], "git_commit": fixed["commit"],
                "git_tree": repository_tree,
                "tracked_file_count": len(repository_entries),
            }
        directory_sources = {}
        for prefix, required_relative in PINNED_DIRECTORY_SOURCES.items():
            if prefix != "third_party/diffusers_h3":
                raise ValueError(f"unsupported directory source: {prefix}")
            source = (
                diffusers_h3_source.resolve() if diffusers_h3_source is not None
                else (project / prefix).resolve()
            )
            copied = _copy_directory_source(source, staging / prefix)
            if required_relative not in copied:
                raise ValueError(f"directory source contract file missing: {prefix}")
            for relative, identity in copied.items():
                snapshot_name = f"{prefix}/{relative}"
                if snapshot_name in files:
                    raise ValueError(f"snapshot directory collision: {snapshot_name}")
                files[snapshot_name] = {
                    **identity, "path": snapshot_name, "mode": "external",
                    "git_type": "directory_source", "git_object": None,
                    "repository": prefix, "repository_relative_path": relative,
                }
            directory_sources[prefix] = {
                "source_path_at_freeze": str(source),
                "required_contract_file": required_relative,
                "file_count": len(copied),
                "content_tree_sha256": _content_tree_sha256(copied),
            }
        missing_dynamic = sorted(set(DYNAMIC_EXECUTION_FILES) - set(files))
        if missing_dynamic:
            raise ValueError(f"dynamic execution source missing: {missing_dynamic}")
        manifest = {
            "format": FORMAT,
            "status": "PASS_COMPLETE_COMMIT_TREE_DYNAMIC_SOURCE_FREEZE",
            "permission": "READ_ONLY_EXECUTION_SNAPSHOT_ONLY",
            "git_commit": commit,
            "git_tree": tree,
            "repositories": repositories,
            "directory_sources": directory_sources,
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
                "lightwam_source_root": (
                    "third_party/Light-WAM/src/lightwam/models/wan22"
                ),
            },
            "claim_boundary": (
                "Freezes every superproject byte, every tracked byte from pinned "
                "StarWAM/FastWAM/FACT/Light-WAM commits, the complete supplied "
                "diffusers_h3 source tree, and all known runtime/dynamic import "
                "targets. Python wheels, model bytes, and datasets are separately "
                "content-gated."
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
    repositories = manifest.get("repositories", {})
    directory_sources = manifest.get("directory_sources", {})
    if (
        manifest.get("format") != FORMAT
        or manifest.get("status") != "PASS_COMPLETE_COMMIT_TREE_DYNAMIC_SOURCE_FREEZE"
        or manifest.get("permission") != "READ_ONLY_EXECUTION_SNAPSHOT_ONLY"
        or not isinstance(files, dict)
        or manifest.get("tracked_file_count") != len(files)
        or re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("git_commit"))) is None
        or re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("git_tree"))) is None
        or set(repositories) != set(PINNED_GIT_REPOSITORIES)
        or set(directory_sources) != set(PINNED_DIRECTORY_SOURCES)
    ):
        raise ValueError("source freeze manifest contract mismatch")
    for name, fixed in PINNED_GIT_REPOSITORIES.items():
        repository = repositories[name]
        if (
            repository.get("source") != fixed["source"]
            or repository.get("git_commit") != fixed["commit"]
            or re.fullmatch(r"[0-9a-f]{40}", str(repository.get("git_tree"))) is None
            or repository.get("tracked_file_count") != sum(
                row.get("repository") == name for row in files.values()
            )
        ):
            raise ValueError(f"pinned repository manifest mismatch: {name}")
    for name, required_relative in PINNED_DIRECTORY_SOURCES.items():
        source = directory_sources[name]
        selected = {
            row["repository_relative_path"]: {
                key: row[key] for key in ("kind", "target", "sha256", "size")
                if key in row
            }
            for row in files.values() if row.get("repository") == name
        }
        if (
            source.get("required_contract_file") != required_relative
            or source.get("file_count") != len(selected)
            or required_relative not in selected
            or source.get("content_tree_sha256") != _content_tree_sha256(selected)
        ):
            raise ValueError(f"directory source manifest mismatch: {name}")
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
    parser.add_argument("--diffusers-h3-source", type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.snapshot is None or any(
            value is not None for value in (
                args.project, args.expected_commit, args.expected_tree, args.output,
                args.diffusers_h3_source,
            )
        ):
            parser.error("--verify requires only --snapshot and optional expected SHA")
        report = verify(args.snapshot, args.expected_manifest_sha256)
    else:
        if any(value is None for value in (
            args.project, args.expected_commit, args.expected_tree, args.output
        )) or args.snapshot is not None or args.expected_manifest_sha256 is not None:
            parser.error("freeze requires project, expected commit/tree and output")
        report = freeze(
            args.project, args.expected_commit, args.expected_tree, args.output,
            diffusers_h3_source=args.diffusers_h3_source,
        )
    print(json.dumps({
        "status": report["status"], "git_commit": report["git_commit"],
        "git_tree": report["git_tree"],
        "tracked_file_count": report["tracked_file_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
