#!/usr/bin/env python3
"""Download the exact quantized MiniMax-H3 bundle used by the 5090 baseline.

The downloader intentionally uses only the Python standard library so an empty
cloud image can resume the large ModelScope files before the runtime environment
is installed.  File size and SHA256 are part of the model identity contract.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import time
import urllib.request


BASE_URL = "https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3/resolve/master"
FILES = (
    (
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        20_970_379_616,
        "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
    ),
    (
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        15_687_142_551,
        "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
    ),
    (
        "vae/minimax_h3_video_vae_fp16.safetensors",
        5_207_808_496,
        "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
    ),
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: pathlib.Path, expected_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > expected_size:
        raise RuntimeError(f"oversized partial file: {target}")

    last_report = 0.0
    while (offset := target.stat().st_size if target.exists() else 0) < expected_size:
        request = urllib.request.Request(
            url,
            headers={
                "Range": f"bytes={offset}-",
                "User-Agent": "H3-WAM-evidence-downloader/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    mode = "wb"
                    offset = 0
                else:
                    mode = "ab" if offset else "wb"
                with target.open(mode) as handle:
                    while chunk := response.read(8 << 20):
                        handle.write(chunk)
                        offset += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 30:
                            print(
                                f"[download] {target.name}: "
                                f"{offset}/{expected_size} ({offset / expected_size:.1%})",
                                flush=True,
                            )
                            last_report = now
        except Exception as error:  # Network failures are expected on long cloud transfers.
            print(f"[retry] {target.name}: {error}", flush=True)
            time.sleep(5)

    actual_size = target.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"size mismatch for {target}: {actual_size}/{expected_size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()

    for relative_path, expected_size, expected_sha256 in FILES:
        target = args.output_root / relative_path
        download(f"{BASE_URL}/{relative_path}", target, expected_size)
        actual_sha256 = sha256(target)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA256 mismatch for {target}: {actual_sha256}/{expected_sha256}"
            )
        print(f"[complete] {relative_path} sha256={actual_sha256}", flush=True)


if __name__ == "__main__":
    main()
