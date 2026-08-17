from pathlib import Path

import torch

from fastwam.models.h3wam.c66_lingbot_fastwam_persistent import (
    H3FastWAMLingBotPersistentPolicy,
    prepare_committed_observation_sequence,
)
from fastwam.models.h3wam.fastwam_full_tower import (
    H3FastWAMFullTowerPolicy,
    LAYERWISE_H3_50_TO_ACTION_30,
)
from fastwam.models.h3wam.lingbot_persistent_kv import LingBotPersistentKVState


INV_FREQ = torch.logspace(0, -3.75, 16, base=10.0)


def _arguments():
    return {
        "enabled": True,
        "carrier_layers": LAYERWISE_H3_50_TO_ACTION_30,
        "action_block_to_h3_layer": LAYERWISE_H3_50_TO_ACTION_30,
        "action_dim": 2,
        "proprio_dim": 3,
        "context_dim": 16,
        "hidden_dim": 128,
        "ffn_dim": 256,
        "num_heads": 1,
        "attn_head_dim": 128,
        "freq_dim": 16,
        "num_layers": 30,
    }


def _kv(seed: int, tokens: int = 3):
    generator = torch.Generator().manual_seed(seed)
    return {
        layer: {
            name: torch.randn(1, tokens, 1, 128, generator=generator)
            for name in ("k", "v")
        }
        for layer in LAYERWISE_H3_50_TO_ACTION_30
    }


def _inputs():
    return {
        "noisy_actions": torch.randn(1, 4, 2),
        "timestep": torch.tensor([500.0]),
        "text_context": torch.randn(1, 3, 16),
        "proprio": torch.randn(1, 3),
        "text_mask": torch.ones(1, 3, dtype=torch.bool),
    }


def test_disabled_path_and_state_dict_are_exact_c58():
    torch.manual_seed(1)
    parent = H3FastWAMFullTowerPolicy(**_arguments()).eval()
    model = H3FastWAMLingBotPersistentPolicy(
        persistent_enabled=False, **_arguments()
    ).eval()
    model.load_state_dict(parent.state_dict(), strict=True)
    assert set(model.state_dict()) == set(parent.state_dict())
    values = _inputs()
    current = _kv(2)
    with torch.no_grad():
        expected = parent(video_kv_cache=current, **values)
        actual = model(video_kv_cache=current, **values)
    assert torch.equal(actual, expected)

    # Enabling the lifecycle with an empty episode state changes no math: the
    # current H3 carrier is still consumed once at the same thirty blocks.
    active = H3FastWAMLingBotPersistentPolicy(
        persistent_enabled=True, **_arguments()
    ).eval()
    active.load_state_dict(parent.state_dict(), strict=True)
    with torch.no_grad():
        active_actual = active(
            video_kv_cache=current,
            persistent_state=active.new_persistent_state("empty"),
            **values,
        )
    assert torch.equal(active_actual, expected)


def test_training_reindex_is_framewise_and_value_exact():
    sequence = [_kv(3), _kv(4), _kv(5)]
    merged = prepare_committed_observation_sequence(
        sequence,
        layers=LAYERWISE_H3_50_TO_ACTION_30,
        temporal_inv_freq=INV_FREQ,
        frame_start=7,
    )
    layer = LAYERWISE_H3_50_TO_ACTION_30[0]
    assert merged[layer]["k"].shape[1] == 9
    assert torch.equal(merged[layer]["v"][:, :3], sequence[0][layer]["v"])
    assert torch.equal(merged[layer]["v"][:, 3:6], sequence[1][layer]["v"])
    assert not torch.equal(merged[layer]["k"][:, :3], sequence[0][layer]["k"])


def test_committed_observation_is_not_duplicated_and_action_kv_is_block_internal():
    torch.manual_seed(6)
    model = H3FastWAMLingBotPersistentPolicy(
        persistent_enabled=True,
        persistent_window_frames=3,
        observation_tokens_per_frame=3,
        action_tokens_per_frame=4,
        use_gradient_checkpointing=True,
        **_arguments(),
    )
    state = model.new_persistent_state("episode")
    values = _inputs()
    history_observation = _kv(7)
    executed = torch.randn(1, 4, 2)
    model.commit_executed_feedback(
        state,
        observation_kv=history_observation,
        observed_frame_count=1,
        executed_actions=executed,
        text_context=values["text_context"],
        proprio=values["proprio"],
        text_mask=values["text_mask"],
    )
    audit = state.audit()
    assert audit["frame_st_id"] == 1
    assert audit["action_st_id"] == 4
    assert all(
        item["kinds"] == ["observation", "action"]
        for item in audit["layers"].values()
    )

    # The post-feedback current observation is already in committed state;
    # changing the redundant API argument cannot change the result.
    with torch.no_grad():
        first = model(video_kv_cache=_kv(8), persistent_state=state, **values)
        second = model(video_kv_cache=_kv(9), persistent_state=state, **values)
    assert torch.equal(first, second)

    model.zero_grad(set_to_none=True)
    prediction = model(video_kv_cache=_kv(10), persistent_state=state, **values)
    prediction.square().mean().backward()
    gradients = [
        block.self_attn.k.weight.grad
        for block in model.action_expert.blocks
    ]
    assert all(gradient is not None and float(gradient.norm()) > 0 for gradient in gradients)


def test_runtime_strict_restore_and_fifo_are_shared_with_c57_state_contract():
    model = H3FastWAMLingBotPersistentPolicy(
        persistent_enabled=True,
        persistent_window_frames=2,
        observation_tokens_per_frame=3,
        action_tokens_per_frame=4,
        **_arguments(),
    )
    state = model.new_persistent_state("restore")
    values = _inputs()
    for index in range(3):
        model.commit_executed_feedback(
            state,
            observation_kv=_kv(20 + index),
            observed_frame_count=1,
            executed_actions=torch.randn(1, 4, 2),
            text_context=values["text_context"],
            proprio=values["proprio"],
            text_mask=values["text_mask"],
        )
    restored = LingBotPersistentKVState.from_snapshot(state.snapshot())
    assert restored.audit() == state.audit()
    assert all(item["tokens"] <= 14 for item in state.audit()["layers"].values())
    clone = H3FastWAMLingBotPersistentPolicy(
        persistent_enabled=True,
        persistent_window_frames=2,
        observation_tokens_per_frame=3,
        action_tokens_per_frame=4,
        **_arguments(),
    )
    clone.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    clone.eval()
    with torch.no_grad():
        expected = model(video_kv_cache=_kv(30), persistent_state=state, **values)
        actual = clone(video_kv_cache=_kv(31), persistent_state=restored, **values)
    assert torch.equal(actual, expected)


def test_cloud_launcher_exposes_pinned_h3_diffusers_layout_source():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts/h3wam/launch_c66_lingbot_c58_mechanical.sh"
    ).read_text(encoding="utf-8")
    assert "${workspace}/runtime/h3-int8-native/bin/python" in launcher
    assert "${project_root}/third_party/diffusers_h3" in launcher
    assert "export PYTHONPATH=" in launcher
    assert "/usr/local/nvidia/lib:/usr/local/nvidia/lib64" in launcher
    assert '"${python_bin}" -c' not in launcher
    assert '--diffusers-h3-source "${diffusers_h3_root}"' in launcher

    probe = (
        Path(__file__).resolve().parents[1]
        / "scripts/h3wam/probe_c66_lingbot_c58_persistent.py"
    ).read_text(encoding="utf-8")
    assert "cuda_actiondit_preflight" not in probe
    assert "torch.randn(256,256" not in probe
    assert "C58OnlineFrozenH3Provider" in probe


def test_bf16_feedback_commit_keeps_fp32_timestep_but_matches_linear_dtype():
    model = H3FastWAMLingBotPersistentPolicy(
        persistent_enabled=True,
        persistent_window_frames=2,
        observation_tokens_per_frame=3,
        action_tokens_per_frame=4,
        **_arguments(),
    ).to(dtype=torch.bfloat16)
    state = model.new_persistent_state("bf16-feedback")
    values = {
        name: value.to(dtype=torch.bfloat16) if value.is_floating_point() else value
        for name, value in _inputs().items()
    }
    observation = {
        layer: {
            name: value.to(dtype=torch.bfloat16)
            for name, value in item.items()
        }
        for layer, item in _kv(91).items()
    }
    model.commit_executed_feedback(
        state,
        observation_kv=observation,
        observed_frame_count=1,
        executed_actions=torch.randn(1, 4, 2, dtype=torch.bfloat16),
        text_context=values["text_context"],
        proprio=values["proprio"],
        text_mask=values["text_mask"],
    )
    assert state.frame_st_id == 1
    assert state.action_st_id == 4
    assert all(
        entry["tokens"] == 7 for entry in state.audit()["layers"].values()
    )


def test_c66_canary_keeps_full_lingbot_history_and_three_paired_arms():
    root = Path(__file__).resolve().parents[1]
    freezer = (root / "scripts/h3wam/freeze_c66_lingbot_c58_canary.py").read_text()
    trainer = (
        root / "scripts/h3wam/train_c66_lingbot_c58_persistent_canary.py"
    ).read_text()
    launcher = (
        root / "scripts/h3wam/launch_c66_lingbot_c58_canary_8gpu.sh"
    ).read_text()
    assert '"train_rows": 800' in freezer
    assert '"heldout_rows": 64' in freezer
    assert '"history_chunks": 7' in freezer
    assert '"history_observation_frames": 15' in freezer
    assert '"history_executed_actions": 56' in freezer
    assert '"clean_context", "history_action_shuffle", "context_off"' in freezer
    assert 'if args.steps != 100' in trainer
    assert 'world_size != 8' in trainer
    assert 'shuffle_actions=True' in trainer
    assert 'predict_off(raw.policy' in trainer
    assert 'clean_beats_shuffle_by_1pct' in trainer
    assert 'clean_vs_off_regression_at_most_5pct' in trainer
    assert 'dba327cd41f26596cec23228eb5d4be67ff2fa6a4c354b198271ef48cd87468e' in launcher
    assert '${workspace}/runtime/conda-py311/bin/python' in launcher
    assert '${workspace}/.venv/lib/python3.11/site-packages' in launcher
    assert '${workspace}/runtime/c66-conda-deps' in launcher
    assert 'm.version("comfy-kitchen")=="0.2.26"' in launcher
    assert '--nproc_per_node=8' in launcher
