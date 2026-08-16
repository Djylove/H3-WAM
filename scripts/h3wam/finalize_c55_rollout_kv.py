#!/usr/bin/env python3
"""Finalize exact coverage and identity for a C48/C60 FACT rollout K/V cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from fastwam.models.h3wam.dreamwam_kv_carrier import DEFAULT_H3_CARRIER_LAYERS
from fastwam.models.h3wam.fastwam_full_tower import LAYERWISE_H3_50_TO_ACTION_30


FORMAT = "h3wam-c55-rollout-kv-ready-v1"
ITEM_FORMAT = "h3wam-c55-rollout-kv-shard-v1"
EXPECTED_H3_SHA256 = (
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
)
SUPPORTED_DATASET_FORMATS = {
    "h3wam-c48-fact-dense-value-dataset-v1",
    "h3wam-c60-counterfactual-failure-dataset-v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-observations-sha256", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layers", nargs="+", type=int, default=DEFAULT_H3_CARRIER_LAYERS
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite C55 READY: {output}")
    dataset_path = args.dataset.resolve()
    observations_path = args.observations.resolve()
    root = args.root.resolve()
    layers = tuple(int(value) for value in args.layers)
    if layers not in {
        tuple(DEFAULT_H3_CARRIER_LAYERS),
        tuple(LAYERWISE_H3_50_TO_ACTION_30),
    }:
        raise ValueError("unsupported FACT H3 K/V layer schema")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if dataset.get("format") not in SUPPORTED_DATASET_FORMATS:
        raise ValueError(f"unsupported FACT rollout dataset: {dataset.get('format')!r}")
    dataset_sha256 = sha256_file(dataset_path)
    observations_sha256 = sha256_file(observations_path)
    if dataset_sha256 != args.expected_dataset_sha256:
        raise ValueError("FACT rollout dataset identity mismatch")
    if observations_sha256 != args.expected_observations_sha256:
        raise ValueError("FACT rollout observations identity mismatch")
    required = {
        int(row["current_observation_id"])
        for row in dataset["samples"]
        if row["split"] in {"train", "validation"}
    }
    expected_names = {f"obs_{value:06d}.pt" for value in required}
    item_paths = sorted((root / "items").glob("obs_*.pt"))
    actual_names = {path.name for path in item_paths}
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    partials = sorted(root.rglob("*.partial"))
    if missing or extra or partials:
        raise ValueError(
            f"C55 K/V coverage failed: missing={len(missing)} "
            f"extra={len(extra)} partial={len(partials)}"
        )
    markers = []
    for shard in range(32):
        path = root / "markers" / f"shard{shard}.json"
        marker = json.loads(path.read_text(encoding="utf-8"))
        if (
            marker.get("format") != ITEM_FORMAT
            or marker.get("shard") != shard
            or marker.get("num_shards") != 32
            or marker.get("h3_checkpoint_sha256") != EXPECTED_H3_SHA256
            or marker.get("dataset_sha256") != dataset_sha256
            or marker.get("observations_sha256") != observations_sha256
            or tuple(marker.get("layers", ())) != layers
        ):
            raise ValueError(f"C55 shard marker mismatch: {shard}")
        markers.append(marker)
    if sum(int(marker["items"]) for marker in markers) != len(required):
        raise ValueError("C55 shard marker item total mismatch")

    sampled = []
    for shard in range(32):
        path = next(
            path
            for path in item_paths
            if int(path.stem.split("_")[1]) % 32 == shard
        )
        item = torch.load(path, map_location="cpu", weights_only=False)
        if (
            item.get("format") != ITEM_FORMAT
            or tuple(item.get("layers", ())) != layers
            or item.get("h3_checkpoint_sha256") != EXPECTED_H3_SHA256
            or item.get("dataset_sha256") != dataset_sha256
            or item.get("observations_sha256") != observations_sha256
        ):
            raise ValueError(f"C55 sampled item identity mismatch: {path}")
        signatures = set()
        for layer in layers:
            for name in ("k", "v"):
                tensor = item["video_kv_cache"][layer][name]
                if tensor.shape != (32, 56, 128) or tensor.dtype != torch.bfloat16:
                    raise ValueError(f"C55 sampled tensor mismatch: {path}")
                signatures.add(tensor.untyped_storage().data_ptr())
                if not torch.isfinite(tensor.float()).all():
                    raise ValueError(f"C55 sampled tensor non-finite: {path}")
        if len(signatures) != 2 * len(layers):
            raise ValueError(f"C55 sampled storage alias: {path}")
        sampled.append({"shard": shard, "file": path.name, "sha256": sha256_file(path)})

    marker_identity = hashlib.sha256(
        "\n".join(
            json.dumps(marker, sort_keys=True, separators=(",", ":"))
            for marker in markers
        ).encode()
    ).hexdigest()
    result = {
        "format": FORMAT,
        "ready": True,
        "dataset_format": dataset["format"],
        "dataset_sha256": dataset_sha256,
        "observations_sha256": observations_sha256,
        "h3_checkpoint_sha256": EXPECTED_H3_SHA256,
        "layers": list(layers),
        "items": len(item_paths),
        "bytes": sum(path.stat().st_size for path in item_paths),
        "missing": 0,
        "extra": 0,
        "partials": 0,
        "shards": 32,
        "marker_identity_sha256": marker_identity,
        "sampled_items": sampled,
        "validation_boundary": (
            "Every item was shape/finite/storage-validated before atomic write by "
            "the extractor; finalizer proves exact filename coverage and reopens one "
            "identity/shape/finite sample per shard."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
