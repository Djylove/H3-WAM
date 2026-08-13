#!/usr/bin/env python3
"""Launch H3-DreamWAM on 8 GPUs and run real closed-loop LIBERO episodes."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "h3wam"))

from rollout_libero import free_local_port, run_episode, wait_for_server  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--action-stage", type=Path)
    parser.add_argument("--lingbot-shared-stage", type=Path)
    parser.add_argument("--lingbot-shared", action="store_true")
    parser.add_argument("--h3-joint-stage", type=Path)
    parser.add_argument("--dot", action="store_true")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--torchrun", type=Path, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--task-languages", nargs="+")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--trial-indices", type=int, nargs="+")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--wait-steps", type=int, default=30)
    parser.add_argument("--env-resolution", type=int, default=256)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--actions-per-chunk", type=int, default=4)
    parser.add_argument("--target-latent-frames", type=int, default=12)
    parser.add_argument("--last-trainable-layers", type=int, default=2)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--video-sample-steps", type=int, default=0)
    parser.add_argument("--action-sample-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument(
        "--fp32-model-storage", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--binarize-gripper", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--clip-normalized-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--require-text-only-context", action="store_true")
    parser.add_argument("--dreamwam-exact-action-norm", action="store_true")
    parser.add_argument("--action-init-alpha-scaling", action="store_true")
    parser.add_argument("--fixed-noise-seed", type=int)
    parser.add_argument("--action-median-window", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=1.0)
    return parser.parse_args()


def server_command(args: argparse.Namespace, port: int, ready_file: Path) -> list[str]:
    if args.lingbot_shared:
        return [
            str(args.torchrun.absolute()),
            "--standalone",
            f"--nproc-per-node={args.nproc_per_node}",
            str(Path(__file__).with_name("serve_h3_lingbot_shared_fsdp.py")),
            "--model", str(args.model.resolve()),
            "--stage", str(args.lingbot_shared_stage.resolve()),
            "--cache-root", str(args.cache_root.resolve()),
            "--manifest", str(args.manifest.resolve()),
            "--port", str(port),
            "--ready-file", str(ready_file),
            "--action-horizon", str(args.action_horizon),
            "--actions-per-chunk", str(args.actions_per_chunk),
            "--target-latent-frames", str(args.target_latent_frames),
            "--sample-steps", str(args.sample_steps),
            "--video-sample-steps", str(args.video_sample_steps),
            "--action-sample-steps", str(args.action_sample_steps),
            "--last-trainable-layers", str(args.last_trainable_layers),
            "--binarize-gripper" if args.binarize_gripper else "--no-binarize-gripper",
            "--clip-normalized-actions" if args.clip_normalized_actions else "--no-clip-normalized-actions",
        ]
    return [
        str(args.torchrun.absolute()),
        "--standalone",
        f"--nproc-per-node={args.nproc_per_node}",
        str(
            Path(__file__).with_name(
                "serve_h3dotwam_fsdp.py" if args.dot else "serve_h3dreamwam_fsdp.py"
            )
        ),
        "--model",
        str(args.model.resolve()),
        "--action-stage",
        str(args.action_stage.resolve()),
        "--cache-root",
        str(args.cache_root.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--port",
        str(port),
        "--ready-file",
        str(ready_file),
        "--action-horizon",
        str(args.action_horizon),
        "--sample-steps",
        str(args.sample_steps),
        "--action-median-window",
        str(args.action_median_window),
        "--action-scale",
        str(args.action_scale),
        "--binarize-gripper" if args.binarize_gripper else "--no-binarize-gripper",
        "--clip-normalized-actions"
        if args.clip_normalized_actions
        else "--no-clip-normalized-actions",
    ] + (
        ["--h3-joint-stage", str(args.h3_joint_stage.resolve())]
        if args.dot and args.h3_joint_stage is not None
        else []
    ) + ([] if args.dot else [
        "--fp32-model-storage"
        if args.fp32_model_storage
        else "--no-fp32-model-storage"
    ]) + (["--require-text-only-context"] if args.require_text_only_context else []) + (
        ["--dreamwam-exact-action-norm"]
        if args.dreamwam_exact_action_norm and not args.dot
        else []
    ) + (
        ["--action-init-alpha-scaling"]
        if args.action_init_alpha_scaling and not args.dot
        else []
    )


def write_results(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.lingbot_shared:
        if args.lingbot_shared_stage is None:
            raise ValueError("lingbot-shared requires --lingbot-shared-stage")
        if args.action_median_window != 1 or args.action_scale != 1.0:
            raise ValueError(
                "shared-H3 does not implement action-median-window/action-scale; "
                "refusing a silently ineffective sweep"
            )
    elif args.action_stage is None:
        raise ValueError("non-LingBot rollout requires --action-stage")
    if min(
        args.nproc_per_node,
        args.trials,
        args.max_steps,
        args.replan_steps,
        args.action_horizon,
        args.actions_per_chunk,
        args.target_latent_frames,
        args.last_trainable_layers,
        args.sample_steps,
    ) <= 0:
        raise ValueError("positive rollout arguments are required")
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
    server_log = (output_dir / "policy_server.log").open("w", encoding="utf-8")
    server_env = os.environ.copy()
    # The rollout parent may run from a simulator-only environment whose
    # site-packages contains a CPU torch build. Keep those packages available
    # to LIBERO in this process, but never leak them into the CUDA policy
    # server launched by the main training torchrun.
    server_env["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT), str(REPO_ROOT / "src"))
    )
    server = subprocess.Popen(
        server_command(args, port, ready_file),
        cwd=REPO_ROOT,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=server_env,
    )
    connection = None
    results = {
        "policy": (
            "h3_lingbot_shared_fsdp"
            if args.lingbot_shared
            else ("h3dotwam_fsdp" if args.dot else "h3dreamwam_fsdp")
        ),
        "model": str(args.model.resolve()),
        "action_stage": (
            None if args.action_stage is None else str(args.action_stage.resolve())
        ),
        "lingbot_shared_stage": (
            None
            if args.lingbot_shared_stage is None
            else str(args.lingbot_shared_stage.resolve())
        ),
        "h3_joint_stage": (
            None
            if args.h3_joint_stage is None
            else str(args.h3_joint_stage.resolve())
        ),
        "manifest": str(args.manifest.resolve()),
        "suite": args.suite,
        "task_ids": task_ids,
        "trial_indices": trial_indices,
        "max_steps": args.max_steps,
        "wait_steps": args.wait_steps,
        "env_resolution": args.env_resolution,
        "replan_steps": args.replan_steps,
        "action_horizon": args.action_horizon,
        "sample_steps": args.sample_steps,
        "video_sample_steps": args.video_sample_steps or args.sample_steps,
        "action_sample_steps": args.action_sample_steps or args.sample_steps,
        "clip_normalized_actions": args.clip_normalized_actions,
        "fixed_noise_seed": args.fixed_noise_seed,
        "status": "running",
        "expected_tasks": len(task_ids),
        "expected_episodes": len(task_ids) * len(trial_indices),
        "finished_episodes": 0,
        "tasks": [],
    }
    started = time.perf_counter()
    try:
        wait_for_server(server, ready_file, args.startup_timeout)
        results["server"] = json.loads(ready_file.read_text(encoding="utf-8"))
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
                        fixed_replan_noise=args.fixed_noise_seed is not None,
                        fixed_noise_seed=args.fixed_noise_seed,
                        episode_key=f"task{task_id}:trial{trial}",
                        use_action_ensembler=False,
                    )
                    episode["trial"] = trial
                    if trajectory is not None:
                        import numpy as np

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
                    results["finished_episodes"] += 1
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
                    write_results(output_dir / "results.partial.json", results)
            finally:
                env.close()
            task_result["successes"] = sum(
                int(episode["success"]) for episode in task_result["episodes"]
            )
            task_result["success_rate"] = task_result["successes"] / len(
                task_result["episodes"]
            )
            results["tasks"].append(task_result)
            write_results(output_dir / "results.partial.json", results)
        episodes = [
            episode for task_result in results["tasks"]
            for episode in task_result["episodes"]
        ]
        results["successes"] = sum(int(episode["success"]) for episode in episodes)
        results["episodes"] = len(episodes)
        results["success_rate"] = results["successes"] / max(len(episodes), 1)
        results["duration_seconds"] = time.perf_counter() - started
        if len(results["tasks"]) != results["expected_tasks"] or (
            len(episodes) != results["expected_episodes"]
        ):
            raise RuntimeError("rollout finished with an incomplete task/episode count")
        results["status"] = "complete"
        write_results(output_dir / "results.json", results)
        (output_dir / "results.partial.json").unlink(missing_ok=True)
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
