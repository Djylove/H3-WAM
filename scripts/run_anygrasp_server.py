from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.anygrasp.deploy_policy import (
    FastWAMAnyGraspPolicy,
    compose_task_config,
    infer_dataset_stats_path,
)
from experiments.anygrasp.server_client import PolicyServer


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    if str(value).strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a FastWAM AnyGrasp policy server.")
    parser.add_argument("--ckpt", required=True, help="Path to FastWAM weights checkpoint, e.g. checkpoints/weights/step_xxxxxx.pt.")
    parser.add_argument(
        "--task",
        default="real_anygrasp_v2_hierarchical_1cam_384_1e-4",
        help=(
            "Hydra task config under configs/task. Use "
            "real_anygrasp_v2_uncond_1cam_384_1e-4 for native FastWAM."
        ),
    )
    parser.add_argument("--dataset-stats-path", default=None, help="Path to dataset_stats.json. If omitted, inferred from checkpoint parents.")
    parser.add_argument("--host", default="0.0.0.0", help="ZMQ bind host.")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ bind port.")
    parser.add_argument("--api-token", default=None, help="Optional token required from clients.")
    parser.add_argument("--device", default="cuda:0", help="Inference device.")
    parser.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"], help="Model dtype.")
    parser.add_argument("--action-horizon", type=int, default=None, help="Action horizon returned by each get_action call.")
    parser.add_argument("--replan-steps", type=int, default=None, help="Recommended client-side execution steps per returned chunk.")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--high-video-inference-steps", type=int, default=None)
    parser.add_argument("--low-video-inference-steps", type=int, default=None)
    parser.add_argument("--high-denoise-step", type=int, default=None)
    parser.add_argument("--low-denoise-step", type=int, default=None)
    parser.add_argument("--high-reuse-step", type=int, default=None)
    parser.add_argument("--low-reuse-step", type=int, default=None)
    parser.add_argument("--action-inference-steps", type=int, default=None)
    parser.add_argument("--joint-denoise", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sigma-shift", default=None, help="Optional scheduler sigma shift.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rand-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Extra Hydra override, repeatable. Example: --config-override model.hierarchical_dynamic_skip=false",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = Path(args.ckpt).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    stats_path = infer_dataset_stats_path(ckpt, explicit=args.dataset_stats_path)
    cfg = compose_task_config(args.task, overrides=args.config_override)

    policy = FastWAMAnyGraspPolicy(
        cfg=cfg,
        checkpoint_path=ckpt,
        dataset_stats_path=stats_path,
        device=args.device,
        mixed_precision=args.mixed_precision,
        action_horizon=args.action_horizon,
        replan_steps=args.replan_steps,
        num_inference_steps=args.num_inference_steps,
        high_video_inference_steps=args.high_video_inference_steps,
        low_video_inference_steps=args.low_video_inference_steps,
        high_denoise_step=args.high_denoise_step,
        low_denoise_step=args.low_denoise_step,
        high_reuse_step=args.high_reuse_step,
        low_reuse_step=args.low_reuse_step,
        action_inference_steps=args.action_inference_steps,
        joint_denoise=args.joint_denoise,
        sigma_shift=_optional_float(args.sigma_shift),
        seed=args.seed,
        rand_device=args.rand_device,
        tiled=args.tiled,
    )

    print("Starting FastWAM AnyGrasp policy server...")
    print(f"  task: {args.task}")
    print(f"  ckpt: {ckpt}")
    print(f"  dataset_stats: {stats_path}")
    print(f"  model_variant: {'hierarchical' if policy.is_hierarchical_model else 'native'}")
    print(f"  device: {policy.device}")
    print(f"  host: {args.host}")
    print(f"  port: {args.port}")
    print("  observation format: {'video': {'top': uint8 HWC/THWC/BTHWC}, 'state': {'default': float32 D/TD/BTD}, 'language': {'task': [[str]]}}")
    print("  action format: {'default': float32 [B,T,31]} by default; set options.action_space='raw' for [B,T,37].")

    with PolicyServer(policy=policy, host=args.host, port=args.port, api_token=args.api_token) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down FastWAM AnyGrasp policy server.")


if __name__ == "__main__":
    main()
