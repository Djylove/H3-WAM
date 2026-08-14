#!/usr/bin/env python3
"""Run closed-loop LIBERO episodes against the isolated H3-WAM policy server."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.policy_environment import standalone_policy_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=("h3", "h3_feature", "h3_feature_int8", "baseline"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--h3-feature-ensemble-checkpoint",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--h3-feature-ensemble-mode",
        choices=("mean", "switch", "disagreement_switch", "learned_switch"),
        default="mean",
    )
    parser.add_argument("--h3-feature-switch-step", type=int)
    parser.add_argument("--h3-feature-disagreement-threshold", type=float)
    parser.add_argument("--h3-feature-switch-consecutive", type=int, default=1)
    parser.add_argument("--h3-feature-switch-gate-checkpoint", type=Path)
    parser.add_argument("--h3-feature-gate-threshold", type=float)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--task-languages",
        nargs="+",
        help=(
            "Resolve benchmark tasks by exact language and override --task-ids. "
            "Prefer this when dataset task_index values are not benchmark IDs."
        ),
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--trial-indices",
        type=int,
        nargs="+",
        help="Explicit benchmark init-state indices; overrides --trials.",
    )
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--wait-steps", type=int, default=30)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=32,
        help="Number of actions requested from the policy at each replan.",
    )
    parser.add_argument("--model-evaluations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help=(
            "Save raw dual-camera observations, proprioception and policy actions "
            "at every replan for offline teacher relabeling."
        ),
    )
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--comfy-root", type=Path)
    parser.add_argument("--h3-checkpoint", type=Path)
    parser.add_argument("--h3-model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-latent-frames", type=int, default=12)
    parser.add_argument("--h3-feature-audio-horizon", type=int)
    parser.add_argument(
        "--h3-tail-delta",
        type=Path,
        help="Feature-only BF16 tail delta applied over the quantized Comfy H3 base.",
    )
    parser.add_argument("--h3-video-lora-checkpoint", type=Path)
    parser.add_argument("--video-vae", type=Path)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-last-blocks", type=int, default=50)
    parser.add_argument(
        "--context-mode",
        choices=("cached", "online_episode", "online_replan"),
        default="cached",
    )
    parser.add_argument("--text-encoder", type=Path)
    parser.add_argument(
        "--binarize-gripper", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fixed-replan-noise", action="store_true")
    parser.add_argument(
        "--fixed-noise-seed",
        type=int,
        help="Use one exact diffusion noise seed for every replan and trial.",
    )
    parser.add_argument("--action-median-window", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--sample-ensemble-size", type=int, default=1)
    parser.add_argument(
        "--h3-feature-ablation", choices=("none", "zero"), default="none"
    )
    parser.add_argument("--h3-action-mode", type=int)
    parser.add_argument(
        "--lock-h3-action-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use-action-ensembler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Average overlapping action chunks at each absolute timestep, "
            "matching FastWAM's LIBERO temporal ensemble."
        ),
    )
    return parser.parse_args()


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def policy_command(args: argparse.Namespace, port: int, ready_file: Path) -> list[str]:
    command = [
        # Do not resolve this symlink: uv virtualenv interpreters rely on the
        # venv path to discover their site-packages.
        str(args.policy_python.absolute()),
        str(Path(__file__).with_name("serve_rollout_policy.py")),
        "--policy",
        args.policy,
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--cache-root",
        str(args.cache_root.resolve()),
        "--port",
        str(port),
        "--ready-file",
        str(ready_file),
        "--model-evaluations",
        str(args.model_evaluations),
        "--action-horizon",
        str(args.action_horizon),
        "--seed",
        str(args.seed),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-last-blocks",
        str(args.lora_last_blocks),
        "--context-mode",
        args.context_mode,
        "--binarize-gripper" if args.binarize_gripper else "--no-binarize-gripper",
        "--action-median-window",
        str(args.action_median_window),
        "--action-scale",
        str(args.action_scale),
        "--sample-ensemble-size",
        str(args.sample_ensemble_size),
        "--h3-feature-ablation",
        args.h3_feature_ablation,
        "--h3-feature-switch-consecutive",
        str(args.h3_feature_switch_consecutive),
        "--lock-h3-action-mode"
        if args.lock_h3_action_mode
        else "--no-lock-h3-action-mode",
    ]
    if args.policy in ("h3", "h3_feature"):
        for flag, value in (
            ("--comfy-root", args.comfy_root),
            ("--h3-checkpoint", args.h3_checkpoint),
            ("--video-vae", args.video_vae),
        ):
            if value is None:
                raise ValueError(f"H3 rollout requires {flag}")
            command.extend((flag, str(value.resolve())))
        if args.h3_video_lora_checkpoint is not None:
            command.extend(
                (
                    "--h3-video-lora-checkpoint",
                    str(args.h3_video_lora_checkpoint.resolve()),
                )
            )
        if args.h3_tail_delta is not None:
            command.extend(
                ("--h3-tail-delta", str(args.h3_tail_delta.resolve()))
            )
        if args.context_mode in ("online_episode", "online_replan"):
            if args.text_encoder is None:
                raise ValueError("online context modes require --text-encoder")
            command.extend(("--text-encoder", str(args.text_encoder.resolve())))
    elif args.policy == "h3_feature_int8":
        for flag, value in (
            ("--h3-checkpoint", args.h3_checkpoint),
            ("--h3-model", args.h3_model),
        ):
            if value is None:
                raise ValueError(f"standalone INT8 H3 rollout requires {flag}")
            command.extend((flag, str(value.resolve())))
        command.extend(
            (
                "--device",
                args.device,
                "--target-latent-frames",
                str(args.target_latent_frames),
            )
        )
        if args.h3_feature_audio_horizon is not None:
            command.extend(
                (
                    "--h3-feature-audio-horizon",
                    str(args.h3_feature_audio_horizon),
                )
            )
        if args.h3_video_lora_checkpoint is not None:
            command.extend(
                (
                    "--h3-video-lora-checkpoint",
                    str(args.h3_video_lora_checkpoint.resolve()),
                )
            )
        if args.h3_tail_delta is not None:
            raise ValueError("standalone INT8 rollout does not support H3 tail delta")
    if args.h3_action_mode is not None:
        command.extend(("--h3-action-mode", str(args.h3_action_mode)))
    for checkpoint in args.h3_feature_ensemble_checkpoint:
        command.extend(
            ("--h3-feature-ensemble-checkpoint", str(checkpoint.resolve()))
        )
    command.extend(("--h3-feature-ensemble-mode", args.h3_feature_ensemble_mode))
    if args.h3_feature_switch_step is not None:
        command.extend(
            ("--h3-feature-switch-step", str(args.h3_feature_switch_step))
        )
    if args.h3_feature_disagreement_threshold is not None:
        command.extend(
            (
                "--h3-feature-disagreement-threshold",
                str(args.h3_feature_disagreement_threshold),
            )
        )
    if args.h3_feature_switch_gate_checkpoint is not None:
        command.extend(
            (
                "--h3-feature-switch-gate-checkpoint",
                str(args.h3_feature_switch_gate_checkpoint.resolve()),
            )
        )
    if args.h3_feature_gate_threshold is not None:
        command.extend(
            ("--h3-feature-gate-threshold", str(args.h3_feature_gate_threshold))
        )
    return command


def policy_environment() -> dict[str, str]:
    """Keep simulator-only wheels out of the standalone policy process.

    ``run_cloud_libero.sh`` prepends the LIBERO wheel directory to PYTHONPATH
    for the simulator.  The policy uses its own virtualenv and must not import
    NumPy (or any other package) from that directory; mixing the imported
    NumPy with virtualenv package metadata breaks accelerate's version gates.
    """

    return standalone_policy_environment()


def wait_for_server(process: subprocess.Popen, ready_file: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_file.exists():
            return
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"policy server exited during startup with code {exit_code}")
        time.sleep(0.25)
    raise TimeoutError(f"policy server was not ready after {timeout:.1f}s")


def frame_from_observation(obs: dict) -> np.ndarray:
    agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return np.concatenate((agent, wrist), axis=1)


def object_joint_state(env) -> dict[str, list[float]]:
    """Capture compact simulator state for task-progress diagnostics."""

    state = {}
    for name in env.sim.model.joint_names:
        if name.startswith(("robot", "gripper")):
            continue
        value = np.asarray(env.sim.data.get_joint_qpos(name), dtype=np.float64)
        state[name] = value.reshape(-1).tolist()
    return state


def predict(
    connection,
    obs: dict,
    task: str,
    seed: int,
    episode_key: str,
    step: int,
    previous_action: np.ndarray,
    executed_action_history: np.ndarray | None = None,
) -> tuple[np.ndarray, dict, float]:
    agentview = np.ascontiguousarray(obs["agentview_image"], dtype=np.uint8)
    wristview = np.ascontiguousarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)
    request = {
        "command": "predict",
        "task": task,
        "seed": int(seed),
        "episode_key": episode_key,
        "step": int(step),
        "previous_environment_action": np.asarray(
            previous_action, dtype=np.float32
        ).tolist(),
        "executed_action_history": (
            []
            if executed_action_history is None
            else np.asarray(executed_action_history, dtype=np.float32).tolist()
        ),
        # NumPy 1.x and 2.x array pickles are not mutually importable. Send a
        # stable bytes/list wire format across the two virtual environments.
        "agentview_bytes": agentview.tobytes(),
        "agentview_shape": agentview.shape,
        "wristview_bytes": wristview.tobytes(),
        "wristview_shape": wristview.shape,
        "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32).tolist(),
        "eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32).tolist(),
        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).tolist(),
    }
    started = time.perf_counter()
    connection.send(request)
    response = connection.recv()
    roundtrip_seconds = time.perf_counter() - started
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "policy server failed"))
    return np.asarray(response["actions"], dtype=np.float32), response["metadata"], roundtrip_seconds


def run_episode(
    env,
    initial_state,
    task: str,
    connection,
    *,
    episode_seed: int,
    max_steps: int,
    wait_steps: int,
    replan_steps: int,
    save_video: bool,
    save_trajectory: bool,
    fixed_replan_noise: bool,
    fixed_noise_seed: int | None,
    episode_key: str,
    use_action_ensembler: bool,
) -> tuple[dict, list[np.ndarray], dict[str, np.ndarray] | None]:
    env.reset()
    obs = env.set_init_state(initial_state)
    frames = []
    trajectory: dict[str, list] | None = (
        {
            "step": [],
            "agentview_image": [],
            "wristview_image": [],
            "eef_pos": [],
            "eef_quat": [],
            "gripper_qpos": [],
            "previous_action": [],
            "policy_actions": [],
            # Full MuJoCo state makes an exact cross-process teacher takeover
            # possible without keeping H3 and FastWAM resident together.
            "sim_state": [],
        }
        if save_trajectory
        else None
    )
    done = False
    for _ in range(wait_steps):
        obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        if done:
            break

    initial_object_joints = object_joint_state(env)
    max_object_joint_delta = {name: 0.0 for name in initial_object_joints}

    step = 0
    replans = 0
    policy_seconds = []
    model_seconds = []
    vae_seconds = []
    context_seconds = []
    peak_allocated_gib = 0.0
    saturation_fractions = []
    mean_action_magnitudes = []
    context_id = None
    first_environment_action = None
    first_environment_action_chunk = None
    replan_first_actions = []
    selected_action_modes = []
    selected_action_mode_sources = []
    action_mode_probabilities = []
    action_head_selected_indices = []
    action_head_motion_disagreements = []
    action_head_gripper_disagreements = []
    action_head_switch_gate_probabilities = []
    if use_action_ensembler:
        from fastwam.models.h3wam import ActionEnsembler

        action_ensembler = ActionEnsembler()
    else:
        action_ensembler = None
    ensemble_prediction_counts = []
    done = bool(done)
    previous_action = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    executed_actions: list[np.ndarray] = []
    while step < max_steps and not done:
        if trajectory is not None:
            trajectory["step"].append(step)
            trajectory["agentview_image"].append(
                np.ascontiguousarray(obs["agentview_image"], dtype=np.uint8)
            )
            trajectory["wristview_image"].append(
                np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"], dtype=np.uint8
                )
            )
            trajectory["eef_pos"].append(
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
            )
            trajectory["eef_quat"].append(
                np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
            )
            trajectory["gripper_qpos"].append(
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
            )
            trajectory["previous_action"].append(previous_action.copy())
            trajectory["sim_state"].append(
                np.asarray(env.get_sim_state(), dtype=np.float64)
            )
        actions, metadata, roundtrip = predict(
            connection,
            obs,
            task,
            fixed_noise_seed
            if fixed_noise_seed is not None
            else (episode_seed if fixed_replan_noise else episode_seed + replans),
            episode_key,
            step,
            previous_action,
            np.asarray(executed_actions[-16:], dtype=np.float32),
        )
        if trajectory is not None:
            trajectory["policy_actions"].append(actions.copy())
        context_id = metadata.get("context_id")
        if first_environment_action is None:
            first_environment_action = metadata.get("first_environment_action")
            first_environment_action_chunk = metadata.get("environment_action_chunk")
        replan_first_actions.append(metadata.get("first_environment_action"))
        selected_action_modes.append(metadata.get("selected_action_mode"))
        selected_action_mode_sources.append(
            metadata.get("selected_action_mode_source")
        )
        action_mode_probabilities.append(metadata.get("action_mode_probabilities"))
        action_head_selected_indices.append(
            metadata.get("action_head_selected_index")
        )
        action_head_motion_disagreements.append(
            metadata.get("action_head_motion_disagreement")
        )
        action_head_gripper_disagreements.append(
            metadata.get("action_head_gripper_disagreement")
        )
        action_head_switch_gate_probabilities.append(
            metadata.get("action_head_switch_gate_probability")
        )
        policy_seconds.append(roundtrip)
        model_seconds.append(float(metadata.get("inference_seconds", 0.0)))
        vae_seconds.append(float(metadata.get("vae_encode_seconds", 0.0)))
        context_seconds.append(float(metadata.get("context_encode_seconds", 0.0)))
        peak_allocated_gib = max(
            peak_allocated_gib, float(metadata.get("peak_allocated_gib", 0.0))
        )
        saturation_fractions.append(float((np.abs(actions[..., :6]) >= 0.999).mean()))
        mean_action_magnitudes.append(float(np.abs(actions[..., :6]).mean()))
        replans += 1
        if action_ensembler is not None:
            action_ensembler.add_actions(actions, step)
            execution_horizon = min(replan_steps, max_steps - step)
            ensemble_prediction_counts.extend(
                action_ensembler.prediction_count(step + offset)
                for offset in range(execution_horizon)
            )
            actions_to_execute = np.stack(
                [
                    action_ensembler.get_action(timestamp)
                    for timestamp in range(step, step + execution_horizon)
                ],
                axis=0,
            )
            action_ensembler.cleanup(step)
        else:
            actions_to_execute = actions[:replan_steps]
        for action in actions_to_execute:
            if save_video:
                frames.append(frame_from_observation(obs))
            obs, _, done, _ = env.step(action)
            previous_action = np.asarray(action, dtype=np.float32)
            executed_actions.append(previous_action.copy())
            current_object_joints = object_joint_state(env)
            for name, initial in initial_object_joints.items():
                current = np.asarray(current_object_joints[name])
                baseline = np.asarray(initial)
                max_object_joint_delta[name] = max(
                    max_object_joint_delta[name],
                    float(np.abs(current - baseline).max()),
                )
            step += 1
            if done or step >= max_steps:
                break
    if save_video:
        frames.append(frame_from_observation(obs))
    packed_trajectory = None
    if trajectory is not None:
        packed_trajectory = {
            key: np.asarray(values)
            for key, values in trajectory.items()
        }
    return (
        {
            "success": bool(done),
            "steps": step,
            "replans": replans,
            "context_id": context_id,
            "first_environment_action": first_environment_action,
            "first_environment_action_chunk": first_environment_action_chunk,
            "replan_first_actions": replan_first_actions,
            "selected_action_modes": selected_action_modes,
            "selected_action_mode_sources": selected_action_mode_sources,
            "action_mode_probabilities": action_mode_probabilities,
            "action_head_selected_indices": action_head_selected_indices,
            "action_head_motion_disagreements": action_head_motion_disagreements,
            "action_head_gripper_disagreements": action_head_gripper_disagreements,
            "action_head_switch_gate_probabilities": (
                action_head_switch_gate_probabilities
            ),
            "mean_policy_roundtrip_seconds": sum(policy_seconds) / max(len(policy_seconds), 1),
            "mean_model_inference_seconds": sum(model_seconds) / max(len(model_seconds), 1),
            "mean_vae_encode_seconds": sum(vae_seconds) / max(len(vae_seconds), 1),
            "total_context_encode_seconds": sum(context_seconds),
            "peak_allocated_gib": peak_allocated_gib,
            "action_saturation_fraction": sum(saturation_fractions)
            / max(len(saturation_fractions), 1),
            "mean_action_magnitude": sum(mean_action_magnitudes)
            / max(len(mean_action_magnitudes), 1),
            "use_action_ensembler": use_action_ensembler,
            "mean_temporal_ensemble_predictions": sum(ensemble_prediction_counts)
            / max(len(ensemble_prediction_counts), 1),
            "initial_object_joints": initial_object_joints,
            "final_object_joints": object_joint_state(env),
            "max_object_joint_delta": max_object_joint_delta,
        },
        frames,
        packed_trajectory,
    )


def main() -> None:
    args = parse_args()
    if (
        args.trials <= 0
        or args.max_steps <= 0
        or args.replan_steps <= 0
        or args.action_horizon <= 0
    ):
        raise ValueError(
            "trials, max-steps, replan-steps and action-horizon must be positive"
        )
    if args.replan_steps > args.action_horizon:
        raise ValueError("replan-steps cannot exceed action-horizon")
    from libero.libero import benchmark

    from experiments.libero.libero_utils import get_libero_env

    trial_indices = (
        list(args.trial_indices)
        if args.trial_indices is not None
        else list(range(args.trials))
    )
    if not trial_indices or any(index < 0 for index in trial_indices):
        raise ValueError("trial indices must be non-negative")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if args.task_languages:
        language_to_id = {
            suite.get_task(index).language: index
            for index in range(suite.get_num_tasks())
        }
        missing = [
            language for language in args.task_languages if language not in language_to_id
        ]
        if missing:
            raise ValueError(f"task languages not found in {args.suite}: {missing}")
        task_ids = [language_to_id[language] for language in args.task_languages]
    else:
        task_ids = list(args.task_ids)

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ready_file = args.output_dir / "policy_ready.json"
    ready_file.unlink(missing_ok=True)
    server_log_path = args.output_dir / "policy_server.log"
    port = free_local_port()
    server_log = server_log_path.open("w", encoding="utf-8")
    server = subprocess.Popen(
        policy_command(args, port, ready_file),
        cwd=REPO_ROOT,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=policy_environment(),
    )
    connection = None
    all_results = {
        "policy": args.policy,
        "checkpoint": str(args.checkpoint.resolve()),
        "suite": args.suite,
        "task_ids": task_ids,
        "task_languages": args.task_languages,
        "trial_indices": trial_indices,
        "trials_per_task": len(trial_indices),
        "max_steps": args.max_steps,
        "replan_steps": args.replan_steps,
        "action_horizon": args.action_horizon,
        "wait_steps": args.wait_steps,
        "binarize_gripper": args.binarize_gripper,
        "fixed_replan_noise": args.fixed_replan_noise,
        "fixed_noise_seed": args.fixed_noise_seed,
        "action_median_window": args.action_median_window,
        "action_scale": args.action_scale,
        "sample_ensemble_size": args.sample_ensemble_size,
        "h3_feature_ablation": args.h3_feature_ablation,
        "h3_video_lora_checkpoint": (
            None
            if args.h3_video_lora_checkpoint is None
            else str(args.h3_video_lora_checkpoint.resolve())
        ),
        "h3_tail_delta": (
            None
            if args.h3_tail_delta is None
            else str(args.h3_tail_delta.resolve())
        ),
        "h3_action_mode": args.h3_action_mode,
        "h3_feature_ensemble_checkpoints": [
            str(path.resolve()) for path in args.h3_feature_ensemble_checkpoint
        ],
        "h3_feature_ensemble_mode": args.h3_feature_ensemble_mode,
        "h3_feature_switch_step": args.h3_feature_switch_step,
        "h3_feature_disagreement_threshold": (
            args.h3_feature_disagreement_threshold
        ),
        "h3_feature_switch_consecutive": args.h3_feature_switch_consecutive,
        "h3_feature_switch_gate_checkpoint": (
            None
            if args.h3_feature_switch_gate_checkpoint is None
            else str(args.h3_feature_switch_gate_checkpoint.resolve())
        ),
        "h3_feature_gate_threshold": args.h3_feature_gate_threshold,
        "lock_h3_action_mode": args.lock_h3_action_mode,
        "use_action_ensembler": args.use_action_ensembler,
        "model_evaluations": args.model_evaluations,
        "context_mode": args.context_mode,
        "save_trajectories": args.save_trajectories,
        "tasks": [],
    }
    started = time.perf_counter()
    try:
        wait_for_server(server, ready_file, args.startup_timeout)
        connection = Client(("127.0.0.1", port), authkey=b"h3wam-local-rollout")
        for task_id in task_ids:
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            if max(trial_indices) >= len(initial_states):
                raise ValueError(
                    f"task {task_id} has {len(initial_states)} initial states, "
                    f"but trial index {max(trial_indices)} was requested"
                )
            env, task_description = get_libero_env(task, 256, args.seed)
            task_result = {
                "task_id": task_id,
                "task": task_description,
                "episodes": [],
            }
            try:
                for trial in trial_indices:
                    episode, frames, trajectory = run_episode(
                        env,
                        initial_states[trial],
                        task_description,
                        connection,
                        episode_seed=args.seed + task_id * 100_000 + trial * 1_000,
                        max_steps=args.max_steps,
                        wait_steps=args.wait_steps,
                        replan_steps=args.replan_steps,
                        save_video=args.save_video,
                        save_trajectory=args.save_trajectories,
                        fixed_replan_noise=args.fixed_replan_noise,
                        fixed_noise_seed=args.fixed_noise_seed,
                        episode_key=f"task{task_id}:trial{trial}",
                        use_action_ensembler=args.use_action_ensembler,
                    )
                    episode["trial"] = trial
                    if trajectory is not None:
                        trajectory_path = args.output_dir / (
                            f"task{task_id:02d}_trial{trial:02d}_trajectory.npz"
                        )
                        temporary_path = trajectory_path.with_suffix(".npz.tmp")
                        with temporary_path.open("wb") as handle:
                            np.savez_compressed(handle, **trajectory)
                        temporary_path.replace(trajectory_path)
                        episode["trajectory"] = str(trajectory_path)
                        episode["trajectory_replans"] = int(
                            trajectory["step"].shape[0]
                        )
                    task_result["episodes"].append(episode)
                    if args.save_video:
                        video_path = args.output_dir / (
                            f"task{task_id:02d}_trial{trial:02d}_success{int(episode['success'])}.mp4"
                        )
                        imageio.mimsave(video_path, frames, fps=20)
                        episode["video"] = str(video_path)
                    print(
                        json.dumps(
                            {
                                "task": task_id,
                                "trial": trial,
                                "success": episode["success"],
                                "steps": episode["steps"],
                                "replans": episode["replans"],
                                "mean_policy_roundtrip_seconds": episode[
                                    "mean_policy_roundtrip_seconds"
                                ],
                                "peak_allocated_gib": episode["peak_allocated_gib"],
                                "max_object_joint_delta": episode[
                                    "max_object_joint_delta"
                                ],
                            }
                        ),
                        flush=True,
                    )
            finally:
                env.close()
            task_result["successes"] = sum(
                int(episode["success"]) for episode in task_result["episodes"]
            )
            task_result["success_rate"] = task_result["successes"] / len(trial_indices)
            all_results["tasks"].append(task_result)
            (args.output_dir / "results.json").write_text(
                json.dumps(all_results, indent=2), encoding="utf-8"
            )
        episodes = [episode for task in all_results["tasks"] for episode in task["episodes"]]
        all_results["successes"] = sum(int(episode["success"]) for episode in episodes)
        all_results["episodes"] = len(episodes)
        all_results["success_rate"] = all_results["successes"] / max(len(episodes), 1)
        all_results["duration_seconds"] = time.perf_counter() - started
        (args.output_dir / "results.json").write_text(
            json.dumps(all_results, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "summary": {
                        "successes": all_results["successes"],
                        "episodes": all_results["episodes"],
                        "success_rate": all_results["success_rate"],
                        "duration_seconds": all_results["duration_seconds"],
                        "results": str(args.output_dir / "results.json"),
                    }
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        if connection is not None:
            try:
                connection.send({"command": "close"})
                connection.recv()
            except (BrokenPipeError, EOFError):
                pass
            connection.close()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.terminate()
            server.wait(timeout=10)
        server_log.close()
        ready_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
