#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch FastWAM multi-node training via Ray by running one train shell process per node."
        )
    )
    parser.add_argument("--address", type=str, default="auto", help="Ray address, default=auto")
    parser.add_argument("--num-nodes", type=int, required=True, help="Number of nodes to use")
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        required=True,
        help="GPUs per node, passed as the first arg to train_zero*.sh",
    )
    parser.add_argument("--master-port", type=int, default=29500, help="MASTER_PORT for torch distributed")
    parser.add_argument(
        "--train-script",
        type=str,
        default="scripts/train_zero1.sh",
        help="Training shell entrypoint, e.g. scripts/train_zero1.sh or scripts/train_zero2.sh",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=".",
        help="Working directory where training command is executed",
    )
    parser.add_argument(
        "--conda-sh",
        type=str,
        default=None,
        help="Path to conda.sh (e.g. ~/miniconda3/etc/profile.d/conda.sh)",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default=None,
        help="Conda env name to activate before launching train script (e.g. fastwam)",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Hydra overrides for training script, place after '--'",
    )
    return parser.parse_args()


def _alive_gpu_nodes(min_gpus: int) -> list[dict[str, Any]]:
    nodes = []
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        resources = node.get("Resources", {})
        gpus = int(resources.get("GPU", 0))
        if gpus >= min_gpus:
            nodes.append(node)
    nodes.sort(key=lambda n: n.get("NodeManagerAddress", ""))
    return nodes


@ray.remote(max_retries=0)
def _run_train_on_node(
    node_rank: int,
    nnodes: int,
    master_addr: str,
    master_port: int,
    command: list[str],
    workdir: str,
    conda_sh: str | None,
    conda_env: str | None,
) -> int:
    env = os.environ.copy()
    env["NNODES"] = str(nnodes)
    env["NODE_RANK"] = str(node_rank)
    env["MASTER_ADDR"] = str(master_addr)
    env["MASTER_PORT"] = str(master_port)

    cmd_str = " ".join(shlex.quote(part) for part in command)
    if conda_env is not None:
        if conda_sh is None:
            raise RuntimeError("`conda_env` is set but `conda_sh` is missing.")
        shell_cmd = (
            f"source {shlex.quote(conda_sh)} && "
            f"conda activate {shlex.quote(conda_env)} && "
            f"{cmd_str}"
        )
        run_command = ["bash", "-lc", shell_cmd]
    else:
        run_command = command

    print(
        f"[ray-launch] rank={node_rank}/{nnodes} master={master_addr}:{master_port} "
        f"cmd={' '.join(run_command)} workdir={workdir}",
        flush=True,
    )
    proc = subprocess.Popen(run_command, cwd=workdir, env=env)
    return int(proc.wait())


def main() -> None:
    args = _parse_args()
    if args.num_nodes <= 0:
        raise ValueError("--num-nodes must be > 0")
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be > 0")
    if (args.conda_sh is None) ^ (args.conda_env is None):
        raise ValueError("`--conda-sh` and `--conda-env` must be provided together.")

    ray.init(address=args.address, log_to_driver=True)

    candidate_nodes = _alive_gpu_nodes(min_gpus=args.nproc_per_node)
    if len(candidate_nodes) < args.num_nodes:
        raise RuntimeError(
            f"Not enough alive GPU nodes. Need {args.num_nodes}, found {len(candidate_nodes)} "
            f"with GPU>={args.nproc_per_node}."
        )

    selected_nodes = candidate_nodes[: args.num_nodes]
    master_addr = str(selected_nodes[0]["NodeManagerAddress"])

    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    command = ["bash", args.train_script, str(args.nproc_per_node), *train_args]

    refs = []
    for rank, node in enumerate(selected_nodes):
        node_id = node["NodeID"]
        ref = _run_train_on_node.options(
            num_gpus=float(args.nproc_per_node),
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote(
            node_rank=rank,
            nnodes=args.num_nodes,
            master_addr=master_addr,
            master_port=args.master_port,
            command=command,
            workdir=args.workdir,
            conda_sh=(os.path.expanduser(args.conda_sh) if args.conda_sh is not None else None),
            conda_env=args.conda_env,
        )
        refs.append(ref)

    exit_codes = ray.get(refs)
    failed = [i for i, code in enumerate(exit_codes) if int(code) != 0]
    if failed:
        raise SystemExit(f"Training failed on node ranks {failed}, exit_codes={exit_codes}")

    print(f"All node ranks completed successfully: exit_codes={exit_codes}")


if __name__ == "__main__":
    main()
