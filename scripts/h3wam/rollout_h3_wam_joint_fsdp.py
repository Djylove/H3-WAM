#!/usr/bin/env python3
"""Launch the distributed full-H3 policy and evaluate real LIBERO episodes."""

from __future__ import annotations

import argparse
import json
import os
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rollout_libero import (  # noqa: E402
    free_local_port,
    predict,
    run_episode,
    wait_for_server,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--action-checkpoint", type=Path)
    parser.add_argument("--h3-tail-checkpoint", type=Path)
    parser.add_argument("--h3-lora-checkpoint", type=Path)
    parser.add_argument("--h3-lora-recovery-only", action="store_true")
    parser.add_argument("--h3-lora-recovery-max-trigger-step", type=int)
    parser.add_argument(
        "--h3-lora-rerun-on-recovery-trigger", action="store_true"
    )
    parser.add_argument("--last-blocks", type=int, default=50)
    parser.add_argument("--action-recovery-checkpoint", type=Path)
    parser.add_argument("--action-recovery-task")
    parser.add_argument("--action-recovery-after-step", type=int, default=64)
    parser.add_argument("--action-recovery-gate-checkpoint", type=Path)
    parser.add_argument("--action-recovery-gate-threshold", type=float)
    parser.add_argument(
        "--action-ensemble-checkpoint", type=Path, action="append", default=[]
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-contexts", type=Path, required=True)
    parser.add_argument("--torchrun", type=Path, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--task-languages", nargs="+")
    parser.add_argument(
        "--policy-task-language",
        help=(
            "Use this instruction for policy conditioning while keeping the "
            "selected LIBERO task and initial state unchanged. This is intended "
            "for fixed-observation language counterfactual probes."
        ),
    )
    parser.add_argument(
        "--probe-policy-task-languages",
        nargs="+",
        help=(
            "Before each rollout, query these alternative instructions plus "
            "the rollout instruction against the exact same initial observation."
        ),
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--trial-indices", type=int, nargs="+")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--wait-steps", type=int, default=30)
    parser.add_argument(
        "--env-resolution",
        type=int,
        default=256,
        help="LIBERO camera render resolution before the policy's 224px resize.",
    )
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--flow-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument(
        "--binarize-gripper", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fixed-noise-seed", type=int)
    parser.add_argument("--action-median-window", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--history-adapter-scale", type=float, default=1.0)
    parser.add_argument(
        "--use-action-ensembler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Average overlapping action chunks at each absolute timestep, "
            "matching FastWAM temporal ensembling."
        ),
    )
    parser.add_argument(
        "--event-stage-routing",
        choices=("learned", "monotonic"),
        default="learned",
    )
    return parser.parse_args()


def server_command(args: argparse.Namespace, port: int, ready_file: Path) -> list[str]:
    command = [
        str(args.torchrun.absolute()),
        "--standalone",
        f"--nproc-per-node={args.nproc_per_node}",
        str(Path(__file__).with_name("serve_h3_wam_joint_fsdp.py")),
        "--model",
        str(args.model.resolve()),
        "--cache-root",
        str(args.cache_root.resolve()),
        "--task-contexts",
        str(args.task_contexts.resolve()),
        "--port",
        str(port),
        "--ready-file",
        str(ready_file),
        "--action-horizon",
        str(args.action_horizon),
        "--flow-steps",
        str(args.flow_steps),
        "--last-blocks",
        str(args.last_blocks),
        "--action-median-window",
        str(args.action_median_window),
        "--action-scale",
        str(args.action_scale),
        "--history-adapter-scale",
        str(args.history_adapter_scale),
        "--event-stage-routing",
        args.event_stage_routing,
        "--binarize-gripper" if args.binarize_gripper else "--no-binarize-gripper",
    ]
    if args.checkpoint is not None:
        command.extend(("--checkpoint", str(args.checkpoint.resolve())))
    if args.action_checkpoint is not None:
        command.extend(
            ("--action-checkpoint", str(args.action_checkpoint.resolve()))
        )
    if args.h3_tail_checkpoint is not None:
        command.extend(("--h3-tail-checkpoint", str(args.h3_tail_checkpoint.resolve())))
    if args.h3_lora_checkpoint is not None:
        command.extend(("--h3-lora-checkpoint", str(args.h3_lora_checkpoint.resolve())))
    if args.h3_lora_recovery_only:
        command.append("--h3-lora-recovery-only")
    if args.h3_lora_recovery_max_trigger_step is not None:
        command.extend(
            (
                "--h3-lora-recovery-max-trigger-step",
                str(args.h3_lora_recovery_max_trigger_step),
            )
        )
    if args.h3_lora_rerun_on_recovery_trigger:
        command.append("--h3-lora-rerun-on-recovery-trigger")
    if args.action_recovery_checkpoint is not None:
        command.extend(
            (
                "--action-recovery-checkpoint",
                str(args.action_recovery_checkpoint.resolve()),
                "--action-recovery-task",
                args.action_recovery_task,
                "--action-recovery-after-step",
                str(args.action_recovery_after_step),
            )
        )
        if args.action_recovery_gate_checkpoint is not None:
            command.extend(
                (
                    "--action-recovery-gate-checkpoint",
                    str(args.action_recovery_gate_checkpoint.resolve()),
                )
            )
        if args.action_recovery_gate_threshold is not None:
            command.extend(
                (
                    "--action-recovery-gate-threshold",
                    str(args.action_recovery_gate_threshold),
                )
            )
    for path in args.action_ensemble_checkpoint:
        command.extend(("--action-ensemble-checkpoint", str(path.resolve())))
    return command


def write_results(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if min(
        args.nproc_per_node,
        args.trials,
        args.max_steps,
        args.replan_steps,
        args.action_horizon,
        args.flow_steps,
    ) <= 0:
        raise ValueError("positive rollout arguments are required")
    if args.replan_steps > args.action_horizon:
        raise ValueError("replan-steps cannot exceed action-horizon")
    if (args.checkpoint is None) == (args.action_checkpoint is None):
        raise ValueError("set exactly one of --checkpoint and --action-checkpoint")
    if args.action_ensemble_checkpoint and args.action_checkpoint is None:
        raise ValueError("action ensembles require --action-checkpoint")
    if (args.h3_tail_checkpoint is not None or args.h3_lora_checkpoint is not None) and args.action_checkpoint is None:
        raise ValueError("H3 adapters require --action-checkpoint")
    if args.h3_tail_checkpoint is not None and args.h3_lora_checkpoint is not None:
        raise ValueError("H3 tail and LoRA checkpoints are mutually exclusive")
    if args.h3_lora_recovery_only and args.h3_lora_checkpoint is None:
        raise ValueError("recovery-only LoRA requires --h3-lora-checkpoint")
    if args.h3_lora_rerun_on_recovery_trigger and not args.h3_lora_recovery_only:
        raise ValueError("trigger rerun requires --h3-lora-recovery-only")
    if (args.action_recovery_checkpoint is None) != (args.action_recovery_task is None):
        raise ValueError("recovery checkpoint and recovery task must be set together")
    if args.action_recovery_after_step < 0:
        raise ValueError("action-recovery-after-step cannot be negative")
    if (
        args.action_recovery_gate_checkpoint is not None
        and args.action_recovery_checkpoint is None
    ):
        raise ValueError("recovery gate requires a recovery action checkpoint")

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
        missing = [task for task in args.task_languages if task not in language_to_id]
        if missing:
            raise ValueError(f"task languages absent from suite: {missing}")
        task_ids = [language_to_id[task] for task in args.task_languages]
    else:
        task_ids = list(args.task_ids)
    if any(index < 0 or index >= suite.get_num_tasks() for index in task_ids):
        raise ValueError("task id lies outside benchmark suite")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ready_file = output_dir / "policy_ready.json"
    ready_file.unlink(missing_ok=True)
    port = free_local_port()
    log_path = output_dir / "policy_server.log"
    server_log = log_path.open("w", encoding="utf-8")
    server = subprocess.Popen(
        server_command(args, port, ready_file),
        cwd=REPO_ROOT,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    connection = None
    results = {
        "policy": "h3_wam_joint_fsdp",
        "model": str(args.model.resolve()),
        "checkpoint": (
            str(args.checkpoint.resolve()) if args.checkpoint is not None else None
        ),
        "action_checkpoint": (
            str(args.action_checkpoint.resolve())
            if args.action_checkpoint is not None
            else None
        ),
        "h3_tail_checkpoint": (
            str(args.h3_tail_checkpoint.resolve())
            if args.h3_tail_checkpoint is not None
            else None
        ),
        "h3_lora_checkpoint": (
            str(args.h3_lora_checkpoint.resolve())
            if args.h3_lora_checkpoint is not None
            else None
        ),
        "history_adapter_scale": args.history_adapter_scale,
        "h3_lora_recovery_only": args.h3_lora_recovery_only,
        "h3_lora_recovery_max_trigger_step": args.h3_lora_recovery_max_trigger_step,
        "h3_lora_rerun_on_recovery_trigger": (
            args.h3_lora_rerun_on_recovery_trigger
        ),
        "action_ensemble_checkpoints": [
            str(path.resolve()) for path in args.action_ensemble_checkpoint
        ],
        "action_recovery_checkpoint": (
            str(args.action_recovery_checkpoint.resolve())
            if args.action_recovery_checkpoint is not None
            else None
        ),
        "action_recovery_task": args.action_recovery_task,
        "action_recovery_after_step": args.action_recovery_after_step,
        "action_recovery_gate_checkpoint": (
            str(args.action_recovery_gate_checkpoint.resolve())
            if args.action_recovery_gate_checkpoint is not None
            else None
        ),
        "action_recovery_gate_threshold": args.action_recovery_gate_threshold,
        "suite": args.suite,
        "task_ids": task_ids,
        "policy_task_language": args.policy_task_language,
        "trial_indices": trial_indices,
        "max_steps": args.max_steps,
        "wait_steps": args.wait_steps,
        "env_resolution": args.env_resolution,
        "replan_steps": args.replan_steps,
        "action_horizon": args.action_horizon,
        "flow_steps": args.flow_steps,
        "fixed_noise_seed": args.fixed_noise_seed,
        "use_action_ensembler": args.use_action_ensembler,
        "tasks": [],
    }
    started = time.perf_counter()
    try:
        wait_for_server(server, ready_file, args.startup_timeout)
        ready = json.loads(ready_file.read_text(encoding="utf-8"))
        results["server"] = ready
        connection = Client(("127.0.0.1", port), authkey=b"h3wam-local-rollout")
        for task_id in task_ids:
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            if max(trial_indices) >= len(initial_states):
                raise ValueError(f"task {task_id} has too few initial states")
            env, task_description = get_libero_env(
                task, args.env_resolution, args.seed
            )
            task_result = {
                "task_id": task_id,
                "task": task_description,
                "policy_task": args.policy_task_language or task_description,
                "episodes": [],
            }
            try:
                for trial in trial_indices:
                    language_counterfactuals = []
                    if args.probe_policy_task_languages:
                        env.reset()
                        probe_obs = env.set_init_state(initial_states[trial])
                        for _ in range(args.wait_steps):
                            probe_obs, _, probe_done, _ = env.step(
                                [0, 0, 0, 0, 0, 0, -1]
                            )
                            if probe_done:
                                break
                        probe_languages = [
                            args.policy_task_language or task_description,
                            *args.probe_policy_task_languages,
                        ]
                        for probe_index, probe_language in enumerate(
                            dict.fromkeys(probe_languages)
                        ):
                            probe_actions, probe_metadata, probe_roundtrip = predict(
                                connection,
                                probe_obs,
                                probe_language,
                                args.seed + task_id * 100_000 + trial * 1_000,
                                f"task{task_id}:trial{trial}:language_probe",
                                0,
                                np.asarray(
                                    [0, 0, 0, 0, 0, 0, -1], dtype=np.float32
                                ),
                            )
                            language_counterfactuals.append(
                                {
                                    "task": probe_language,
                                    "context_id": probe_metadata.get("context_id"),
                                    "first_environment_action": probe_metadata.get(
                                        "first_environment_action"
                                    ),
                                    "environment_action_chunk": probe_actions.tolist(),
                                    "roundtrip_seconds": probe_roundtrip,
                                    "probe_index": probe_index,
                                }
                            )
                    episode, frames, trajectory = run_episode(
                        env,
                        initial_states[trial],
                        args.policy_task_language or task_description,
                        connection,
                        episode_seed=args.seed + task_id * 100_000 + trial * 1_000,
                        max_steps=args.max_steps,
                        wait_steps=args.wait_steps,
                        replan_steps=args.replan_steps,
                        save_video=args.save_video,
                        save_trajectory=args.save_trajectories,
                        fixed_replan_noise=args.fixed_noise_seed is not None,
                        fixed_noise_seed=args.fixed_noise_seed,
                        episode_key=f"task{task_id}:trial{trial}",
                        use_action_ensembler=args.use_action_ensembler,
                    )
                    episode["trial"] = trial
                    if language_counterfactuals:
                        episode["language_counterfactuals"] = (
                            language_counterfactuals
                        )
                    if trajectory is not None:
                        trajectory_path = output_dir / (
                            f"task{task_id:02d}_trial{trial:02d}_trajectory.npz"
                        )
                        with trajectory_path.with_suffix(".npz.tmp").open("wb") as handle:
                            np.savez_compressed(handle, **trajectory)
                        trajectory_path.with_suffix(".npz.tmp").replace(trajectory_path)
                        episode["trajectory"] = str(trajectory_path)
                    if args.save_video:
                        video_path = output_dir / (
                            f"task{task_id:02d}_trial{trial:02d}_success"
                            f"{int(episode['success'])}.mp4"
                        )
                        imageio.mimsave(video_path, frames, fps=20)
                        episode["video"] = str(video_path)
                    task_result["episodes"].append(episode)
                    print(
                        json.dumps(
                            {
                                "task": task_id,
                                "trial": trial,
                                "success": episode["success"],
                                "steps": episode["steps"],
                                "replans": episode["replans"],
                                "mean_roundtrip_seconds": episode[
                                    "mean_policy_roundtrip_seconds"
                                ],
                                "max_object_joint_delta": episode[
                                    "max_object_joint_delta"
                                ],
                            }
                        ),
                        flush=True,
                    )
                    write_results(output_dir / "results.json", results)
            finally:
                env.close()
            task_result["successes"] = sum(
                int(episode["success"]) for episode in task_result["episodes"]
            )
            task_result["success_rate"] = task_result["successes"] / len(
                task_result["episodes"]
            )
            results["tasks"].append(task_result)
        episodes = [episode for task in results["tasks"] for episode in task["episodes"]]
        results["successes"] = sum(int(episode["success"]) for episode in episodes)
        results["episodes"] = len(episodes)
        results["success_rate"] = results["successes"] / max(len(episodes), 1)
        results["duration_seconds"] = time.perf_counter() - started
        write_results(output_dir / "results.json", results)
        print(json.dumps({"summary": results}, indent=2), flush=True)
    finally:
        if connection is not None:
            try:
                connection.send({"command": "close"})
                connection.recv()
            except (BrokenPipeError, EOFError):
                pass
            connection.close()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.terminate()
            server.wait(timeout=30)
        server_log.close()
        ready_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
