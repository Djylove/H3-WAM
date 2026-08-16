from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import torch

from fastwam.models.h3wam.dreamwam_kv_carrier import (
    H3DreamWAMKVCarrierPolicy,
    REPEAT_LAYER49_CARRIER_SOURCE,
)


ROOT = Path(__file__).resolve().parents[1]


def load_trainer():
    path = ROOT / "scripts/h3wam/train_h3_fastwam_full_tower.py"
    spec = importlib.util.spec_from_file_location("_test_c58_matched_trainer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parent_model():
    return H3DreamWAMKVCarrierPolicy(
        enabled=True,
        carrier_layers=(9, 19, 29, 39, 49),
        carrier_source_mode=REPEAT_LAYER49_CARRIER_SOURCE,
        action_dim=2,
        proprio_dim=3,
        context_dim=6,
        hidden_dim=8,
        ffn_dim=16,
        num_heads=2,
        attn_head_dim=4,
        freq_dim=8,
    )


def test_matched_control_strictly_loads_parent_with_fresh_optimizer():
    trainer = load_trainer()
    parent = parent_model().eval()
    spec = trainer.ModelSpec(
        action_dim=2,
        proprio_dim=3,
        context_dim=6,
        hidden_dim=8,
        ffn_dim=16,
        num_heads=2,
        attn_head_dim=4,
        freq_dim=8,
        action_layers=5,
        carrier_layers=(9, 19, 29, 39, 49),
        carrier_source_mode=trainer.REPEAT_LAYER49_MODE,
    )
    control = trainer.build_model(
        spec, device=torch.device("cpu"), dtype=torch.float32
    ).eval()
    control.load_state_dict(parent.state_dict(), strict=True)
    optimizer = torch.optim.AdamW(
        control.parameters(), lr=1.0e-4, weight_decay=1.0e-2, betas=(0.9, 0.95)
    )
    assert optimizer.state == {}
    inputs = {
        "noisy_actions": torch.randn(2, 4, 2),
        "timestep": torch.tensor([100.0, 700.0]),
        "text_context": torch.randn(2, 5, 6),
        "proprio": torch.randn(2, 3),
        "text_mask": torch.ones(2, 5, dtype=torch.bool),
        "video_kv_cache": {
            layer: {
                "k": torch.randn(2, 5, 2, 4),
                "v": torch.randn(2, 5, 2, 4),
            }
            for layer in (9, 19, 29, 39, 49)
        },
    }
    with torch.no_grad():
        expected = parent(**inputs)
        actual = control(**inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_matched_control_and_c58_share_lr_and_flow_seed_schedules():
    trainer = load_trainer()
    control_parameter = torch.nn.Parameter(torch.zeros(()))
    candidate_parameter = torch.nn.Parameter(torch.zeros(()))
    control_optimizer = torch.optim.AdamW(
        [control_parameter], lr=1.0e-4, weight_decay=1.0e-2, betas=(0.9, 0.95)
    )
    candidate_optimizer = torch.optim.AdamW(
        [candidate_parameter], lr=1.0e-4, weight_decay=1.0e-2, betas=(0.9, 0.95)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        control_scheduler = trainer.PARENT.PARENT.build_lr_scheduler(
            control_optimizer,
            warmup_steps=1000,
            scheduler_horizon=10000,
            min_learning_rate=1.0e-6,
        )
        candidate_scheduler = trainer.PARENT.PARENT.build_lr_scheduler(
            candidate_optimizer,
            warmup_steps=1000,
            scheduler_horizon=10000,
            min_learning_rate=1.0e-6,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for step in (1, 2, 999, 1000, 1001, 9999, 10000):
            while control_scheduler.last_epoch < step:
                control_optimizer.step()
                candidate_optimizer.step()
                control_scheduler.step()
                candidate_scheduler.step()
            assert (
                control_optimizer.param_groups[0]["lr"]
                == candidate_optimizer.param_groups[0]["lr"]
            )
            for rank in range(8):
                control_seed = trainer.PARENT.PARENT.distributed_flow_seed(
                    base_seed=42,
                    completed_step=step,
                    accumulation_index=0,
                    rank=rank,
                )
                candidate_seed = trainer.PARENT.PARENT.distributed_flow_seed(
                    base_seed=42,
                    completed_step=step,
                    accumulation_index=0,
                    rank=rank,
                )
                assert control_seed == candidate_seed


def test_long_launchers_share_every_non_depth_training_argument():
    candidate = (ROOT / "scripts/h3wam/launch_c58_fastwam_full30_long.sh").read_text()
    control = (ROOT / "scripts/h3wam/launch_c58_matched_d0_control_long.sh").read_text()
    shared_fragments = (
        "BASE_OFFSET=112000",
        "STAGE_STEPS=1000",
        "GLOBAL_BATCH=8",
        "TOTAL_STAGES=10",
        "--per-device-batch-size 1",
        "--gradient-accumulation-steps 1",
        "--num-workers 0",
        "--learning-rate 1e-4",
        "--weight-decay 0.01",
        "--warmup-steps 1000",
        "--scheduler-horizon 10000",
        "--min-learning-rate 1e-6",
        "--action-horizon 32",
        "--action-shift 5",
    )
    for fragment in shared_fragments:
        assert fragment in candidate
        assert fragment in control
    assert "--matched-d0-control" not in candidate
    assert "--matched-d0-control" in control
