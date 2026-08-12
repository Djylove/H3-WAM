#!/usr/bin/env python3
"""Resume a large HTTP checkpoint with independent byte-range workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--retries", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.workers <= 0 or args.chunk_mib <= 0:
        raise ValueError("size, workers and chunk-mib must be positive")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path = output.with_suffix(output.suffix + ".ranges.json")
    chunk_bytes = args.chunk_mib * 2**20
    ranges = [
        (start, min(start + chunk_bytes, args.size) - 1)
        for start in range(0, args.size, chunk_bytes)
    ]
    completed: set[int] = set()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("size") != args.size or state.get("chunk_bytes") != chunk_bytes:
            raise ValueError("resume metadata does not match size/chunk-mib")
        completed = {int(index) for index in state.get("completed", [])}
    initial_completed = set(completed)
    initial_bytes = sum(
        ranges[item][1] - ranges[item][0] + 1 for item in initial_completed
    )
    file_descriptor = os.open(output, os.O_RDWR | os.O_CREAT)
    os.ftruncate(file_descriptor, args.size)
    lock = threading.Lock()
    started = time.perf_counter()

    def save_state() -> None:
        temporary = state_path.with_suffix(state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "url": args.url,
                    "size": args.size,
                    "chunk_bytes": chunk_bytes,
                    "completed": sorted(completed),
                }
            )
        )
        temporary.replace(state_path)

    def download(index: int) -> int:
        start, stop = ranges[index]
        expected = stop - start + 1
        for attempt in range(args.retries):
            try:
                request = urllib.request.Request(
                    args.url,
                    headers={
                        "Range": f"bytes={start}-{stop}",
                        "User-Agent": "fastwam-parallel-checkpoint/1.0",
                    },
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status != 206 or not content_range.startswith(
                        f"bytes {start}-{stop}/"
                    ):
                        raise RuntimeError(
                            f"range {index} returned status={response.status} "
                            f"content-range={content_range!r}"
                        )
                    offset = start
                    received = 0
                    while True:
                        payload = response.read(2**20)
                        if not payload:
                            break
                        os.pwrite(file_descriptor, payload, offset)
                        offset += len(payload)
                        received += len(payload)
                    if received != expected:
                        raise RuntimeError(
                            f"range {index} received {received}, expected {expected}"
                        )
                with lock:
                    completed.add(index)
                    save_state()
                    elapsed = time.perf_counter() - started
                    downloaded = sum(
                        ranges[item][1] - ranges[item][0] + 1 for item in completed
                    )
                    session_bytes = downloaded - initial_bytes
                    if len(completed) % 64 == 0 or len(completed) == len(ranges):
                        print(
                            json.dumps(
                                {
                                    "completed_ranges": len(completed),
                                    "total_ranges": len(ranges),
                                    "downloaded_gib": round(downloaded / 2**30, 3),
                                    "session_rate_mib_s": round(
                                        session_bytes / 2**20 / elapsed, 3
                                    ),
                                }
                            ),
                            flush=True,
                        )
                return index
            except Exception as error:
                if attempt + 1 == args.retries:
                    raise RuntimeError(
                        f"range {index} failed after {args.retries} attempts"
                    ) from error
                time.sleep(min(2**attempt, 15))
        raise AssertionError("unreachable")

    pending = [index for index in range(len(ranges)) if index not in completed]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download, index) for index in pending]
            for future in as_completed(futures):
                future.result()
    finally:
        os.close(file_descriptor)
    if args.sha256:
        digest = hashlib.sha256()
        with output.open("rb") as handle:
            for payload in iter(lambda: handle.read(8 * 2**20), b""):
                digest.update(payload)
        actual = digest.hexdigest()
        if actual != args.sha256.lower():
            raise RuntimeError(f"sha256 mismatch: expected {args.sha256}, got {actual}")
        print(json.dumps({"output": str(output), "sha256": actual}), flush=True)
    state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
