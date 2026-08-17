#!/usr/bin/env python3
"""Real C58/H3/C57-row zero-optimizer gate for C66 block persistence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from safetensors import safe_open
import torch

from fastwam.models.h3wam.c58_online_training import C58OnlineFrozenH3Provider
from fastwam.models.h3wam.c66_lingbot_fastwam_persistent import (
    H3FastWAMLingBotPersistentPolicy,
    prepare_committed_observation_sequence,
)
from fastwam.models.h3wam.fastwam_full_tower import LAYERWISE_H3_50_TO_ACTION_30
from fastwam.models.h3wam.lingbot_persistent_kv import LingBotPersistentKVState
from fastwam.models.h3wam.deployment import minmax_normalize


C58_CHECKPOINT_SHA256 = (
    "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
)
H3_CHECKPOINT_SHA256 = (
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
)
C57_MANIFEST_SHA256 = (
    "8f95005ac66fd89ca3a22a80d75480e9792b09f976e928f2eb70d4f08680049f"
)
C57_AUDIT_SHA256 = (
    "a383bd0d201b8eb9e3ed52b93bad712a96a4e74ecc063d8d71fcc000e46329fb"
)
C57_SOURCE_MANIFEST_SHA256 = (
    "b0d611c21059fa7da6fb08162b03efadd59aff68354bb101be41d3ae20d98eb1"
)
C58_STATS_SHA256 = (
    "6f7e9f4a2232a798e4e30ad26f5748e71aeeda7fa54cb6ea2d0a3ec7d290e814"
)
EXPECTED_LINGBOT_COMMIT = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
LINGBOT_SOURCE_SHA256 = {
    "wan_va/modules/model.py": (
        "f65fdca5f28b327bfc89ca082a7b55eee3fb4ab1ffc469c642a075920e6f4487"
    ),
    "wan_va/wan_va_server.py": (
        "9c2a427611db487fea5cf40f184b713bf2088e533990ee00fdcd020d2668b4bf"
    ),
    "evaluation/libero/client.py": (
        "63a48baa2cfecd924963b8a8f7e7eda2b367785a7e29ad0a6326f80517c7e004"
    ),
}
EXPECTED_FASTWAM_COMMIT = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
EXPECTED_ACTION_DIT_SHA256 = (
    "1301d9224149de43bb701f620a5d41858ecc63c6b19a573ec32edd45a3bdb0a2"
)
DIFFUSERS_H3_REVISION = "huggingface/diffusers PR14355 head f37ab93e621d5ce206c9662e8291ca8b67d9c555"
DIFFUSERS_H3_BEFORE_DENOISE_SHA256 = (
    "530b007c1d689c3ee1fc1690527f5253522d2da6b44dd326bec99faaf9f72fff"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cuda_actiondit_preflight() -> dict[str, Any]:
    """Exercise cuBLAS in the probe process that will execute real H3.

    The cloud CUDA-13 image intermittently fails a newly spawned, tiny GEMM
    even when the following H3 process is healthy.  A launcher-side child is
    therefore not evidence about this process.  Keep the gate here and use the
    real ActionDiT hidden width (3072) for both execution dtypes.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("the real C66 mechanical/data probe requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    shape = (128, 3072, 3072)
    for dtype in (torch.bfloat16, torch.float16):
        left = torch.randn(shape[0], shape[1], device=device, dtype=dtype)
        right = torch.randn(shape[1], shape[2], device=device, dtype=dtype)
        result = left @ right
        if not torch.isfinite(result).all():
            raise RuntimeError(f"non-finite C66 CUDA preflight for {dtype}")
        del left, right, result
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return {
        "process_scope": "same_process_as_h3_and_actiondit",
        "gemm_shape": [shape[0], shape[1], shape[2]],
        "dtypes": ["bfloat16", "float16"],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_lingbot_source(root: Path) -> dict[str, Any]:
    root = root.resolve()
    hashes = {}
    for name, expected in LINGBOT_SOURCE_SHA256.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != expected:
            raise RuntimeError(f"LingBot source hash mismatch for {name}: {digest}")
        hashes[name] = digest
    # Cloud runtime snapshots intentionally omit the outer repository metadata
    # and may not install a git executable. The execution-bearing files are
    # fail-closed by content hash; the corresponding official revision was
    # established in the local source audit and is recorded here as provenance.
    return {
        "root": str(root),
        "revision": EXPECTED_LINGBOT_COMMIT,
        "identity": "execution_file_sha256",
        "sha256": hashes,
    }


def verify_diffusers_h3_source(root: Path) -> dict[str, Any]:
    root = root.resolve()
    before_denoise = (
        root
        / "src/diffusers/modular_pipelines/minimax_h3/before_denoise.py"
    )
    if not before_denoise.is_file():
        raise FileNotFoundError(before_denoise)
    digest = sha256_file(before_denoise)
    if digest != DIFFUSERS_H3_BEFORE_DENOISE_SHA256:
        raise RuntimeError(f"H3 diffusers layout source SHA256 mismatch: {digest}")

    from diffusers.modular_pipelines.minimax_h3 import before_denoise as imported

    imported_path = Path(imported.__file__).resolve()
    if imported_path != before_denoise:
        raise RuntimeError(
            f"H3 diffusers import escaped pinned source: {imported_path}"
        )
    layout = imported.MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=torch.zeros(14, dtype=torch.long),
        num_latent_frames=12,
        latent_height=14,
        latent_width=28,
        num_audio_latents=32,
        patch_size=(1, 2, 2),
        audio_channels=2,
        audio_tag=2,
        video_tag=0,
        keyframe_anchors=("first",),
    )
    positions, _, video_indices, _, _, condition_rows, _ = layout
    condition_positions = positions[video_indices[:condition_rows].long()]
    unique_times = torch.unique(condition_positions[:, 0]).tolist()
    if (
        int(condition_rows) != 98
        or tuple(condition_positions.shape) != (98, 3)
        or unique_times != [14.0]
    ):
        raise RuntimeError("pinned H3 diffusers real-window layout smoke failed")
    return {
        "root": str(root),
        "revision": DIFFUSERS_H3_REVISION,
        "before_denoise": str(before_denoise),
        "before_denoise_sha256": digest,
        "condition_rows": int(condition_rows),
        "condition_position_shape": list(condition_positions.shape),
        "condition_unique_temporal_positions": unique_times,
    }


def validate_c58_parent_payload(payload: dict[str, Any]) -> dict[str, bool]:
    contract = payload.get("contract", {})
    model_spec = contract.get("model_spec", {})
    expected = LAYERWISE_H3_50_TO_ACTION_30
    checks = {
        "schema_version": payload.get("schema_version") == 1,
        "completed_steps": int(payload.get("completed_steps", -1)) == 10_000,
        "candidate": contract.get("candidate") == "C58B_FASTWAM_FULL30_H3_LAYERWISE",
        "classification": contract.get("classification")
        == "action-only-on-frozen-layerwise-h3-kv_backbone_port",
        "fastwam_commit": contract.get("fastwam_commit") == EXPECTED_FASTWAM_COMMIT,
        "action_dit_sha256": contract.get("fastwam_action_dit_sha256")
        == EXPECTED_ACTION_DIT_SHA256,
        "kv_layers": tuple(contract.get("kv_layers", ())) == expected,
        "block_mapping": tuple(contract.get("action_block_to_h3_layer", ()))
        == expected,
        "model_spec_layers": tuple(model_spec.get("carrier_layers", ())) == expected,
        "model_spec_depth": model_spec.get("action_layers") == 30,
        "online_frozen_h3": contract.get("h3_execution")
        == "online_frozen_int8_per_rank_v1",
        "no_disk_kv": contract.get("disk_kv_training_input") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"checkpoint is not the fixed C58 champion: {failed}")
    return checks


def validate_sequence_data(manifest: Path, audit_path: Path) -> tuple[dict, dict]:
    manifest_sha = sha256_file(manifest)
    audit_sha = sha256_file(audit_path)
    if manifest_sha != C57_MANIFEST_SHA256 or audit_sha != C57_AUDIT_SHA256:
        raise RuntimeError(
            f"C57 sequence identity mismatch: {manifest_sha}/{audit_sha}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    checks = {
        "gate": audit.get("gate") == "PASS",
        "schema": audit.get("schema") == "c57_lingbot_replan8_v1",
        "rows": audit.get("rows") == 200_779,
        "episodes": audit.get("episodes") == 1_542,
        "suites": set(audit.get("suite_rows", {}))
        == {"libero_10", "libero_goal", "libero_object", "libero_spatial"},
        "replan": audit.get("replan") == 8,
        "observe_every": audit.get("observe_every") == 4,
        "missing": audit.get("missing_references") == 0,
        "leakage": audit.get("future_leakage") == 0,
        "capacity": audit.get("max_persistent_tokens_per_layer", 541) <= 540,
        "manifest_inside_audit": audit.get("output_manifest_sha256") == manifest_sha,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"C57 AUDIT contract failed: {failed}")
    selected = None
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if int(row.get("history_chunks", -1)) == 1:
                selected = row
                break
    if selected is None:
        raise RuntimeError("C57 manifest contains no one-history sequence row")
    history = selected["history"]
    if len(history) != 1:
        raise RuntimeError("selected C57 row history does not match its count")
    chunk = history[0]
    starts = [int(value) for value in chunk["observation_starts"]]
    action_indices = [int(value) for value in chunk["action_indices"]]
    expected_actions = list(range(int(chunk["decision_start"]), int(chunk["decision_start"]) + 8))
    row_checks = {
        "schema": selected.get("sequence_schema") == "c57_lingbot_replan8_v1",
        "history_frames": int(selected.get("history_observation_frames", -1)) == 3,
        "history_actions": int(selected.get("history_executed_actions", -1)) == 8,
        "observe_every4": len(starts) == 3
        and starts[1] - starts[0] == 4
        and starts[2] - starts[1] == 4,
        "actions_contiguous8": action_indices == expected_actions,
        "current_is_last_feedback": str(selected["current_id"])
        == str(chunk["observation_source_ids"][-1]),
        "same_episode": int(selected["episode"]) >= 0,
    }
    failed = [name for name, passed in row_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"selected C57 row failed: {failed}")
    return selected, {"audit": checks, "selected_row": row_checks}


def source_rows_for_ids(source_manifest: Path, wanted: set[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    with source_manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sample_id = str(row["id"])
            if sample_id in wanted:
                if sample_id in result:
                    raise RuntimeError(f"duplicate source sample ID: {sample_id}")
                result[sample_id] = row
    if set(result) != wanted:
        raise RuntimeError(f"source manifest is missing IDs: {sorted(wanted - set(result))}")
    return result


def build_model(device: torch.device) -> H3FastWAMLingBotPersistentPolicy:
    return H3FastWAMLingBotPersistentPolicy(
        persistent_enabled=True,
        persistent_window_frames=15,
        observation_tokens_per_frame=32,
        action_tokens_per_frame=4,
        enabled=True,
        carrier_layers=LAYERWISE_H3_50_TO_ACTION_30,
        action_dim=7,
        proprio_dim=8,
        context_dim=5120,
        hidden_dim=1024,
        ffn_dim=4096,
        num_heads=56,
        attn_head_dim=128,
        freq_dim=256,
        num_layers=30,
        use_gradient_checkpointing=False,
        action_block_to_h3_layer=LAYERWISE_H3_50_TO_ACTION_30,
    ).to(device=device, dtype=torch.bfloat16)


def exact_flow_batch(actions: torch.Tensor, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """C58's continuous FP32 shifted-flow batch and importance weight."""

    generator = torch.Generator(device=actions.device).manual_seed(seed)
    uniform = torch.rand(
        actions.shape[0], device=actions.device, dtype=torch.float32, generator=generator
    )
    sigma = 5.0 * uniform / (1.0 + 4.0 * uniform)
    timestep = sigma * 1000.0
    noise = torch.randn(
        actions.shape, device=actions.device, dtype=actions.dtype, generator=generator
    )
    expanded = sigma.view(-1, 1, 1).to(actions.dtype)
    noisy = (1.0 - expanded) * actions + expanded * noise
    target = noise - actions
    grid = torch.linspace(1.0, 0.0, 1001, dtype=torch.float64)[:-1]
    grid_t = (5.0 * grid / (1.0 + 4.0 * grid)) * 1000.0
    grid_y = torch.exp(-2.0 * ((grid_t - 500.0) / 1000.0) ** 2)
    minimum = float(grid_y.min())
    normalizer = float((grid_y - minimum).mean())
    y = torch.exp(-2.0 * ((timestep - 500.0) / 1000.0) ** 2)
    weight = (y - minimum) / (normalizer + 1e-10)
    return noisy, target, timestep, weight


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--h3-checkpoint", type=Path, required=True)
    result.add_argument("--lingbot-source", type=Path, required=True)
    result.add_argument("--diffusers-h3-source", type=Path, required=True)
    result.add_argument("--sequence-manifest", type=Path, required=True)
    result.add_argument("--sequence-audit", type=Path, required=True)
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--cache-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--seed", type=int, default=66017)
    return result


def main() -> None:
    args = parser().parse_args()
    started = time.perf_counter()
    cuda_preflight = cuda_actiondit_preflight()
    source_report = verify_lingbot_source(args.lingbot_source)
    diffusers_h3_report = verify_diffusers_h3_source(args.diffusers_h3_source)

    sequence_manifest = args.sequence_manifest.resolve()
    sequence_audit = args.sequence_audit.resolve()
    row, data_checks = validate_sequence_data(sequence_manifest, sequence_audit)
    chunk = row["history"][0]
    observation_ids = [str(value) for value in chunk["observation_source_ids"]]
    action_source_id = str(chunk["action_source_id"])
    wanted_ids = set(observation_ids) | {action_source_id, str(row["current_id"])}
    source_manifest = args.source_manifest.resolve()
    source_manifest_sha = sha256_file(source_manifest)
    if source_manifest_sha != C57_SOURCE_MANIFEST_SHA256:
        raise RuntimeError(
            f"C57 source manifest SHA256 mismatch: {source_manifest_sha}"
        )
    source_rows = source_rows_for_ids(source_manifest, wanted_ids)
    if any(
        str(source_rows[sample_id]["suite"]) != str(row["suite"])
        or int(source_rows[sample_id]["episode"]) != int(row["episode"])
        for sample_id in wanted_ids
    ):
        raise RuntimeError("C57 sequence crosses a source suite/episode boundary")
    for sample_id, start in zip(
        observation_ids, chunk["observation_starts"], strict=True
    ):
        if int(source_rows[sample_id]["start"]) != int(start):
            raise RuntimeError("C57 observation ID/start differs from source manifest")

    cache_root = args.cache_root.resolve()
    stats_path = cache_root / "stats.pt"
    stats_sha = sha256_file(stats_path)
    if stats_sha != C58_STATS_SHA256:
        raise RuntimeError(f"C58 normalization stats SHA256 mismatch: {stats_sha}")
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    windows = {
        sample_id: torch.load(
            cache_root / "windows" / f"{sample_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for sample_id in wanted_ids
    }
    for sample_id, window in windows.items():
        if tuple(window["first_frame_latents"].shape) != (1, 24, 1, 14, 28):
            raise RuntimeError(f"real H3 latent shape mismatch: {sample_id}")
        if tuple(window["actions"].shape) != (32, 7) or tuple(window["state"].shape) != (8,):
            raise RuntimeError(f"real LIBERO action/state shape mismatch: {sample_id}")
    context_id = str(row["context_id"])
    context_payload = torch.load(
        cache_root / "contexts" / f"{context_id}.pt",
        map_location="cpu",
        weights_only=False,
    )
    context_cpu = context_payload["context"].float()
    tags_cpu = context_payload["token_tags"].long()
    if (
        context_payload.get("text_only") is not True
        or context_cpu.ndim != 3
        or context_cpu.shape[0] != 1
        or context_cpu.shape[-1] != 5120
        or tuple(tags_cpu.shape) != (context_cpu.shape[1],)
    ):
        raise RuntimeError("real H3 text context contract mismatch")

    h3_checkpoint = args.h3_checkpoint.resolve()
    h3_sha = sha256_file(h3_checkpoint)
    if h3_sha != H3_CHECKPOINT_SHA256:
        raise RuntimeError(f"H3 checkpoint SHA256 mismatch: {h3_sha}")
    with safe_open(h3_checkpoint, framework="pt", device="cpu") as handle:
        inv_freq = handle.get_tensor("rope.inv_freq").float().clone()
    if tuple(inv_freq.shape) != (16,) or not torch.isfinite(inv_freq).all():
        raise RuntimeError("H3 rope.inv_freq is not finite shape [16]")

    checkpoint = args.checkpoint.resolve()
    c58_sha = sha256_file(checkpoint)
    if c58_sha != C58_CHECKPOINT_SHA256:
        raise RuntimeError(f"C58 checkpoint SHA256 mismatch: {c58_sha}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parent_checks = validate_c58_parent_payload(payload)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = torch.bfloat16
    context = context_cpu.to(device=device, dtype=dtype)
    text_mask = torch.ones(context.shape[:2], device=device, dtype=torch.bool)
    tags = tags_cpu.unsqueeze(0).to(device)

    provider = C58OnlineFrozenH3Provider(
        h3_checkpoint, layers=LAYERWISE_H3_50_TO_ACTION_30
    ).to(device)
    provider.eval()
    observation_sequence = []
    # ``materialize_frozen_kv`` intentionally returns ordinary tensors rather
    # than inference tensors so the C58 action graph can safely consume them.
    with torch.no_grad():
        for sample_id in observation_ids:
            observation_sequence.append(
                provider(
                    {
                        "current_h3_input": windows[sample_id][
                            "first_frame_latents"
                        ].float().to(device),
                        "text_context": context,
                        "text_token_tags": tags,
                    }
                )
            )
    h3_requires_grad = any(parameter.requires_grad for parameter in provider.parameters())
    h3_has_gradient_before = any(
        parameter.grad is not None for parameter in provider.parameters()
    )
    reindexed = prepare_committed_observation_sequence(
        observation_sequence,
        layers=LAYERWISE_H3_50_TO_ACTION_30,
        temporal_inv_freq=inv_freq.to(device),
        frame_start=0,
    )
    first_layer = LAYERWISE_H3_50_TO_ACTION_30[0]
    temporal_key_delta = float(
        (
            reindexed[first_layer]["k"][:, 32:64].float()
            - observation_sequence[1][first_layer]["k"].float()
        )
        .abs()
        .max()
    )
    value_max_abs = max(
        float(
            (
                reindexed[first_layer]["v"][:, index * 32 : (index + 1) * 32].float()
                - item[first_layer]["v"].float()
            )
            .abs()
            .max()
        )
        for index, item in enumerate(observation_sequence)
    )
    if temporal_key_delta <= 0.0 or value_max_abs != 0.0:
        raise RuntimeError(
            f"real H3 temporal reindex failed: K={temporal_key_delta}, V={value_max_abs}"
        )

    model = build_model(device)
    restored = model.load_state_dict(payload["model"], strict=True)
    if restored.missing_keys or restored.unexpected_keys:
        raise RuntimeError(f"C58 strict restore failed: {restored}")
    state_keys_equal = set(model.state_dict()) == set(payload["model"])
    if not state_keys_equal:
        raise RuntimeError("C66 introduced parameters/state keys beyond C58")
    del payload

    action_window = windows[action_source_id]
    executed_actions = minmax_normalize(
        action_window["actions"][:8].float(),
        stats["action_min"].float(),
        stats["action_max"].float(),
    ).clamp(-5.0, 5.0).unsqueeze(0).to(device=device, dtype=dtype)
    history_proprio = minmax_normalize(
        action_window["state"].float(),
        stats["state_min"].float(),
        stats["state_max"].float(),
    ).clamp(-5.0, 5.0).unsqueeze(0).to(device=device, dtype=dtype)
    current_window = windows[str(row["current_id"])]
    current_actions = minmax_normalize(
        current_window["actions"].float(),
        stats["action_min"].float(),
        stats["action_max"].float(),
    ).clamp(-5.0, 5.0).unsqueeze(0).to(device=device, dtype=dtype)
    current_proprio = minmax_normalize(
        current_window["state"].float(),
        stats["state_min"].float(),
        stats["state_max"].float(),
    ).clamp(-5.0, 5.0).unsqueeze(0).to(device=device, dtype=dtype)
    current_pad = current_window.get(
        "action_is_pad", torch.zeros(32, dtype=torch.bool)
    ).bool().unsqueeze(0).to(device)

    noisy, target, timesteps, weight = exact_flow_batch(
        current_actions, seed=args.seed + 1
    )
    model.eval()
    empty = model.new_persistent_state("empty")
    model.persistent_enabled = False
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        parent_prediction = model(
            noisy,
            timesteps,
            text_context=context,
            proprio=current_proprio,
            video_kv_cache=observation_sequence[-1],
            text_mask=text_mask,
        ).float()
    model.persistent_enabled = True
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        empty_prediction = model(
            noisy,
            timesteps,
            text_context=context,
            proprio=current_proprio,
            video_kv_cache=observation_sequence[-1],
            text_mask=text_mask,
            persistent_state=empty,
        ).float()
    empty_parent_max_abs = float((empty_prediction - parent_prediction).abs().max())
    if empty_parent_max_abs != 0.0:
        raise RuntimeError(f"C66 active-empty C58 parity failed: {empty_parent_max_abs}")

    state = model.new_persistent_state(
        f"{row['suite']}:{int(row['episode'])}"
    )
    model.commit_executed_feedback(
        state,
        observation_kv=reindexed,
        observed_frame_count=3,
        executed_actions=executed_actions,
        text_context=context,
        proprio=history_proprio,
        text_mask=text_mask,
    )
    state_audit = state.audit()
    layer_audits = list(state_audit["layers"].values())
    state_contract = {
        "frame_st_id": state_audit["frame_st_id"] == 3,
        "action_st_id": state_audit["action_st_id"] == 8,
        "tokens104": all(item["tokens"] == 104 for item in layer_audits),
        "two_entries": all(item["entries"] == 2 for item in layer_audits),
        "kinds": all(
            item["kinds"] == ["observation", "action"] for item in layer_audits
        ),
        "no_predicted": state_audit["has_predicted"] is False,
    }
    if not all(state_contract.values()):
        raise RuntimeError(f"C66 real feedback state contract failed: {state_contract}")

    # Once feedback is committed, the final current observation is already in
    # state. The public carrier argument is intentionally redundant and inert.
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        history_prediction_a = model(
            noisy,
            timesteps,
            text_context=context,
            proprio=current_proprio,
            video_kv_cache=observation_sequence[-1],
            text_mask=text_mask,
            persistent_state=state,
        ).float()
        history_prediction_b = model(
            noisy,
            timesteps,
            text_context=context,
            proprio=current_proprio,
            video_kv_cache=observation_sequence[0],
            text_mask=text_mask,
            persistent_state=state,
        ).float()
    redundant_current_max_abs = float(
        (history_prediction_a - history_prediction_b).abs().max()
    )
    history_effect_max_abs = float(
        (history_prediction_a - parent_prediction).abs().max()
    )
    if redundant_current_max_abs != 0.0 or history_effect_max_abs <= 0.0:
        raise RuntimeError(
            "C66 duplicate-current/history-effect gate failed: "
            f"{redundant_current_max_abs}/{history_effect_max_abs}"
        )

    runtime_snapshot = state.snapshot()
    restored_state = LingBotPersistentKVState.from_snapshot(
        runtime_snapshot, device=device, dtype=dtype
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        restored_prediction = model(
            noisy,
            timesteps,
            text_context=context,
            proprio=current_proprio,
            video_kv_cache=observation_sequence[0],
            text_mask=text_mask,
            persistent_state=restored_state,
        ).float()
    runtime_restore_max_abs = float(
        (history_prediction_a - restored_prediction).abs().max()
    )
    if runtime_restore_max_abs != 0.0:
        raise RuntimeError(f"C66 runtime restore failed: {runtime_restore_max_abs}")

    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=dtype):
        prediction = model(
            noisy,
            timesteps,
            text_context=context,
            proprio=current_proprio,
            video_kv_cache=observation_sequence[-1],
            text_mask=text_mask,
            persistent_state=state,
        )
        valid = (~current_pad).unsqueeze(-1).expand_as(prediction)
        per_sample = (
            ((prediction.float() - target.float()).square() * valid).sum((1, 2))
            / valid.sum((1, 2)).clamp(min=1)
        )
        action_loss = (per_sample * weight).mean()
    action_loss.backward()
    block_k_gradients = [
        float(block.self_attn.k.weight.grad.float().norm())
        if block.self_attn.k.weight.grad is not None
        else 0.0
        for block in model.action_expert.blocks
    ]
    gradients_finite_positive = all(
        math.isfinite(value) and value > 0.0 for value in block_k_gradients
    )
    h3_has_gradient_after = any(
        parameter.grad is not None for parameter in provider.parameters()
    )
    if not gradients_finite_positive or h3_requires_grad or h3_has_gradient_before or h3_has_gradient_after:
        raise RuntimeError("C66 30-block action/H3 gradient boundary failed")

    report = {
        "event": "h3_c66_lingbot_c58_block_persistent_real_mechanical_data_gate",
        "status": "PASS_MECHANICAL_DATA_GATE",
        "effect_status": "NOT_EVIDENCE_READY",
        "permission": "NO_GO_OPTIMIZER",
        "classification": "source_aligned_block_internal_committed_context_port",
        "cuda_actiondit_preflight": cuda_preflight,
        "source": source_report,
        "diffusers_h3_source": diffusers_h3_report,
        "c58_checkpoint": str(checkpoint),
        "c58_checkpoint_sha256": c58_sha,
        "c58_parent_identity_checks": parent_checks,
        "c58_parent_strict_restore": True,
        "c66_state_keys_exact_c58": state_keys_equal,
        "h3_checkpoint": str(h3_checkpoint),
        "h3_checkpoint_sha256": h3_sha,
        "h3_rope_inv_freq": inv_freq.tolist(),
        "sequence_manifest": str(sequence_manifest),
        "sequence_manifest_sha256": C57_MANIFEST_SHA256,
        "sequence_audit": str(sequence_audit),
        "sequence_audit_sha256": C57_AUDIT_SHA256,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": source_manifest_sha,
        "stats_sha256": stats_sha,
        "data_checks": data_checks,
        "selected_sequence": {
            "id": row["id"],
            "suite": row["suite"],
            "episode": row["episode"],
            "task": row["task"],
            "current_id": row["current_id"],
            "observation_ids": observation_ids,
            "observation_starts": chunk["observation_starts"],
            "action_source_id": action_source_id,
            "action_indices": chunk["action_indices"],
        },
        "real_h3_temporal_reindex": {
            "merged_tokens_per_layer": int(reindexed[first_layer]["k"].shape[1]),
            "second_frame_key_max_abs_from_local_phase": temporal_key_delta,
            "all_frame_value_max_abs": value_max_abs,
        },
        "active_empty_parent_max_abs": empty_parent_max_abs,
        "state_contract": state_contract,
        "state_audit": state_audit,
        "redundant_current_carrier_max_abs": redundant_current_max_abs,
        "history_vs_current_only_max_abs": history_effect_max_abs,
        "runtime_restore_max_abs": runtime_restore_max_abs,
        "action_flow_loss": float(action_loss.detach()),
        "action_block_k_gradient_norms": block_k_gradients,
        "action_blocks_with_positive_gradient": sum(
            value > 0.0 for value in block_k_gradients
        ),
        "h3_requires_grad": h3_requires_grad,
        "h3_has_gradient": h3_has_gradient_after,
        "optimizer_steps": 0,
        "training_checkpoints_written": 0,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
        "boundary": (
            "This report proves source/checkpoint/data identity, real H3 temporal semantics, "
            "C58 parity, block-internal gradient reachability, committed feedback lifecycle "
            "and restore with zero optimizer steps. It does not authorize training, fusion "
            "or a LIBERO effect claim."
        ),
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
