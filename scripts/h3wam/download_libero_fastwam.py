#!/usr/bin/env python3
"""Download, verify, and safely extract the four FastWAM LIBERO archives."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO = "yuanty/LIBERO-fastwam"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true&expand=false"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
ARCHIVES = {
    "libero_10_no_noops_lerobot.tar.gz",
    "libero_goal_no_noops_lerobot.tar.gz",
    "libero_object_no_noops_lerobot.tar.gz",
    "libero_spatial_no_noops_lerobot.tar.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract-to", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--backend",
        choices=("urllib", "hf_hub", "segmented"),
        default="urllib",
        help="hf_hub enables the Xet high-performance transfer path when available",
    )
    parser.add_argument("--segments-per-file", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def archive_manifest() -> list[dict]:
    with urllib.request.urlopen(API, timeout=60) as response:
        payload = json.load(response)
    files = []
    for item in payload:
        if item.get("path") not in ARCHIVES:
            continue
        lfs = item.get("lfs") or {}
        files.append(
            {
                "path": item["path"],
                "size": int(item["size"]),
                "sha256": lfs.get("oid"),
            }
        )
    if {item["path"] for item in files} != ARCHIVES:
        raise RuntimeError("Hugging Face archive manifest is incomplete")
    if any(not item["sha256"] for item in files):
        raise RuntimeError("Hugging Face archive manifest lacks LFS SHA256 values")
    return sorted(files, key=lambda item: item["path"])


def download_one(item: dict, output: Path) -> dict:
    destination = output / item["path"]
    partial = destination.with_name(destination.name + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = item["size"]
    if destination.is_file() and destination.stat().st_size == expected_size:
        if sha256(destination) == item["sha256"]:
            return {"path": item["path"], "bytes": expected_size, "status": "cached"}
    started = time.perf_counter()
    failures = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected_size:
            break
        if offset > expected_size:
            raise RuntimeError(f"oversized partial archive: {partial}")
        request = urllib.request.Request(f"{RESOLVE}/{item['path']}")
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and getattr(response, "status", None) != 206:
                    raise RuntimeError("archive server ignored resume range")
                with partial.open("ab" if offset else "wb") as handle:
                    while chunk := response.read(16 * 1024 * 1024):
                        handle.write(chunk)
            failures = 0
        except (
            TimeoutError,
            ConnectionError,
            OSError,
            http.client.IncompleteRead,
            urllib.error.URLError,
        ) as error:
            failures += 1
            if failures > 20:
                raise RuntimeError(f"download repeatedly failed: {item['path']}") from error
            time.sleep(min(2**failures, 30))
    if partial.stat().st_size != expected_size or sha256(partial) != item["sha256"]:
        raise RuntimeError(f"archive verification failed: {item['path']}")
    os.replace(partial, destination)
    return {
        "path": item["path"],
        "bytes": expected_size,
        "status": "downloaded",
        "seconds": time.perf_counter() - started,
    }


def download_one_hf(item: dict, output: Path) -> dict:
    from huggingface_hub import hf_hub_download

    started = time.perf_counter()
    destination = output / item["path"]
    if destination.is_file() and destination.stat().st_size == item["size"]:
        if sha256(destination) == item["sha256"]:
            return {"path": item["path"], "bytes": item["size"], "status": "cached"}
    path = Path(
        hf_hub_download(
            repo_id=REPO,
            filename=item["path"],
            repo_type="dataset",
            local_dir=output,
        )
    )
    if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
        raise RuntimeError(f"archive verification failed: {item['path']}")
    return {
        "path": item["path"],
        "bytes": item["size"],
        "status": "downloaded_hf_hub",
        "seconds": time.perf_counter() - started,
    }


def download_range(url: str, path: Path, start: int, end: int) -> None:
    expected_size = end - start + 1
    failures = 0
    while True:
        offset = path.stat().st_size if path.exists() else 0
        if offset == expected_size:
            return
        if offset > expected_size:
            raise RuntimeError(f"oversized segment: {path}")
        request = urllib.request.Request(url)
        request.add_header("Range", f"bytes={start + offset}-{end}")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if getattr(response, "status", None) != 206:
                    raise RuntimeError("archive server ignored segment range")
                with path.open("ab" if offset else "wb") as handle:
                    remaining = expected_size - offset
                    while remaining:
                        chunk = response.read(min(16 * 1024 * 1024, remaining))
                        if not chunk:
                            break
                        handle.write(chunk)
                        remaining -= len(chunk)
            failures = 0
        except (
            TimeoutError,
            ConnectionError,
            OSError,
            http.client.IncompleteRead,
            urllib.error.URLError,
        ) as error:
            failures += 1
            if failures > 20:
                raise RuntimeError(f"segment repeatedly failed: {path.name}") from error
            time.sleep(min(2**failures, 30))


def download_one_segmented(item: dict, output: Path, segments: int) -> dict:
    destination = output / item["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == item["size"]:
        if sha256(destination) == item["sha256"]:
            return {"path": item["path"], "bytes": item["size"], "status": "cached"}

    started = time.perf_counter()
    segment_size = (item["size"] + segments - 1) // segments
    ranges = []
    for index in range(segments):
        start = index * segment_size
        if start >= item["size"]:
            break
        end = min(item["size"] - 1, start + segment_size - 1)
        path = output / f"{item['path']}.segment_{index:03d}"
        ranges.append((path, start, end))
    url = f"{RESOLVE}/{item['path']}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        futures = [pool.submit(download_range, url, path, start, end) for path, start, end in ranges]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    assembling = output / f"{item['path']}.assembling"
    with assembling.open("wb") as destination_handle:
        for path, _, _ in ranges:
            with path.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, destination_handle, 16 * 1024 * 1024)
    if assembling.stat().st_size != item["size"] or sha256(assembling) != item["sha256"]:
        raise RuntimeError(f"archive verification failed: {item['path']}")
    os.replace(assembling, destination)
    for path, _, _ in ranges:
        path.unlink()
    return {
        "path": item["path"],
        "bytes": item["size"],
        "status": "downloaded_segmented",
        "seconds": time.perf_counter() - started,
    }


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (root / member.name).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"unsafe archive path: {member.name}")
        handle.extractall(root)


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.segments_per_file <= 0:
        raise ValueError("worker and segment counts must be positive")
    output = args.output.resolve()
    extract_to = args.extract_to.resolve()
    output.mkdir(parents=True, exist_ok=True)
    extract_to.mkdir(parents=True, exist_ok=True)
    manifest = archive_manifest()
    print(json.dumps({"event": "manifest", "files": len(manifest), "bytes": sum(x["size"] for x in manifest)}), flush=True)
    records = []
    if args.backend == "hf_hub":
        download = download_one_hf
    elif args.backend == "segmented":
        download = lambda item, output: download_one_segmented(item, output, args.segments_per_file)
    else:
        download = download_one
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, item, output) for item in manifest]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({"event": "file", **record}), flush=True)
    for item in manifest:
        safe_extract(output / item["path"], extract_to)
        print(json.dumps({"event": "extracted", "path": item["path"]}), flush=True)
    report = {"event": "complete", "files": records, "extract_to": str(extract_to)}
    (output / "download_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
