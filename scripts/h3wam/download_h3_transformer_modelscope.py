#!/usr/bin/env python3
"""Download and verify official Diffusers H3 components from ModelScope."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO = "MiniMax/MiniMax-H3"
REVISION = "master"
API = (
    f"https://modelscope.cn/api/v1/models/{REPO}/repo/files"
    f"?Revision={REVISION}&Recursive=true"
)
RESOLVE = f"https://modelscope.cn/models/{REPO}/resolve/{REVISION}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--component",
        choices=("transformer", "vae", "text_encoder"),
        default="transformer",
        help="Top-level Diffusers component to download.",
    )
    parser.add_argument("--workers", type=int, default=14)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def component_manifest(component: str) -> list[dict]:
    with urllib.request.urlopen(API, timeout=60) as response:
        payload = json.load(response)
    files = []
    for item in payload["Data"]["Files"]:
        path = item["Path"]
        prefix = f"{component}/"
        relative = path.removeprefix(prefix)
        if (
            path.startswith(prefix)
            and "/" not in relative
            and int(item["Size"]) > 0
            and relative.endswith((".json", ".safetensors", ".txt"))
        ):
            files.append(
                {
                    "path": path,
                    "size": int(item["Size"]),
                    "sha256": item["Sha256"],
                }
            )
    expected_files = {"transformer": 16, "vae": 5, "text_encoder": 23}[component]
    if len(files) != expected_files:
        raise RuntimeError(
            f"expected {expected_files} files for {component}, found {len(files)}"
        )
    return sorted(files, key=lambda item: item["path"])


def download_one(item: dict, output: Path) -> dict:
    relative = Path(item["path"])
    destination = output / relative
    partial = destination.with_name(destination.name + ".modelscope.partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = item["size"]
    expected_sha = item["sha256"]

    if destination.is_file() and destination.stat().st_size == expected_size:
        if sha256(destination) == expected_sha:
            return {"path": str(relative), "bytes": expected_size, "status": "cached"}

    started = time.perf_counter()
    failures = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected_size:
            break
        if offset > expected_size:
            raise RuntimeError(
                f"oversized partial file: {partial} ({offset} > {expected_size})"
            )
        request = urllib.request.Request(f"{RESOLVE}/{item['path']}")
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", None)
                if offset and status != 206:
                    raise RuntimeError(
                        f"server ignored resume range for {relative}: HTTP {status}"
                    )
                mode = "ab" if offset else "wb"
                with partial.open(mode) as handle:
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
                raise RuntimeError(
                    f"download repeatedly failed for {relative} at byte {offset}"
                ) from error
            time.sleep(min(2**failures, 30))

    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"size mismatch for {relative}: {actual_size} != {expected_size}"
        )
    actual_sha = sha256(partial)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"sha256 mismatch for {relative}: {actual_sha} != {expected_sha}"
        )
    os.replace(partial, destination)
    return {
        "path": str(relative),
        "bytes": expected_size,
        "status": "downloaded",
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = component_manifest(args.component)
    print(
        json.dumps(
            {
                "event": "manifest",
                "component": args.component,
                "files": len(manifest),
                "bytes": sum(item["size"] for item in manifest),
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, item, output) for item in manifest]
        for future in concurrent.futures.as_completed(futures):
            print(json.dumps({"event": "file", **future.result()}, sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                "event": "complete",
                "component": args.component,
                "files": len(manifest),
                "bytes": sum(item["size"] for item in manifest),
                "seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
