#!/usr/bin/env python3
"""Record a fixed 12-hour, 30-minute H3-WAM cluster heartbeat."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


NODES = {
    "history": 32611,
    "shared_long": 30907,
    "evaluator": 32409,
    "frameindexed": 30234,
}
HOST = "117.50.181.177"
USER = "dev"
ERROR_PATTERN = "Traceback|Error|FAILED|Killed|NCCL|out of memory|CUDA error"

PROBES = {
    "history": r'''set +e
date -Is
tail -n 1 /mnt/h3-wam/logs/history16_from_s5000_s3000_train.log 2>/dev/null
find /mnt/h3-wam/outputs/h3-lingbot-history -maxdepth 2 -type f \( -name 'history16_from_s5000_s3000*.pt' -o -name '*complete.json' \) -printf '%T@ %s %p\n' 2>/dev/null | sort -n | tail -n 8
rg -n '"'"''' + ERROR_PATTERN + r'''"'"' /mnt/h3-wam/logs/history16_from_s5000_s3000_train.log 2>/dev/null | tail -n 5
pgrep -fc 'verify_h3_lingbot_four_stream_fsdp.py.*history16_from_s5000_s3000'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
df -B1 --output=avail /mnt/h3-wam | tail -n 1''',
    "shared_long": r'''set +e
date -Is
tail -n 1 /mnt/h3-wam/logs/cluster-30907/quantile_flowweight_lr1e5_tail2_s10000.log 2>/dev/null
find /mnt/h3-wam/outputs/h3-lingbot-shared -maxdepth 1 -name 'quantile_flowweight_lr1e5_tail2_s10000*.pt' -printf '%T@ %s %p\n' 2>/dev/null | sort -n | tail -n 8
rg -n '"'"''' + ERROR_PATTERN + r'''"'"' /mnt/h3-wam/logs/cluster-30907/quantile_flowweight_lr1e5_tail2_s10000.log 2>/dev/null | tail -n 5
pgrep -fc 'verify_h3_lingbot_four_stream_fsdp.py.*quantile_flowweight_lr1e5_tail2_s10000'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
df -B1 --output=avail /mnt/h3-wam | tail -n 1''',
    "evaluator": r'''set +e
date -Is
find /mnt/h3-wam/outputs/eval-lingbot-shared/scale-s10000 /mnt/h3-wam/outputs/h3-lingbot-history/eval-s3000 -type f -name '*complete.json' -printf '%T@ %s %p\n' 2>/dev/null | sort -n | tail -n 12
rg -n '"'"''' + ERROR_PATTERN + r'''"'"' /mnt/h3-wam/logs/watch_eval_shared_tail2_s10000.log /mnt/h3-wam/logs/history16_s3000_eval_watcher_32409.log 2>/dev/null | tail -n 5
pgrep -fc 'watch_eval_(shared_tail2_s10000|executed_history_s3000)|verify_h3_lingbot_four_stream_fsdp.py'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
df -B1 --output=avail /mnt/h3-wam | tail -n 1''',
    "frameindexed": r'''set +e
date -Is
tail -n 1 /mnt/h3-wam/logs/cluster-30234/m36_frameindexed_head_epoch2_s2170.log 2>/dev/null
find /mnt/h3-wam/outputs/h3dotwam-frameindexed -maxdepth 1 -name 'm36*' -printf '%T@ %s %p\n' 2>/dev/null | sort -n | tail -n 8
rg -n '"'"''' + ERROR_PATTERN + r'''"'"' /mnt/h3-wam/logs/cluster-30234/m36_frameindexed_head_epoch2_s2170.log 2>/dev/null | tail -n 5
pgrep -fc 'train_h3dotwam_fsdp.py.*m36_frameindexed_head_epoch2'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
df -B1 --output=avail /mnt/h3-wam | tail -n 1''',
}


def ssh_probe(name: str, timeout: int) -> dict:
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        "-p", str(NODES[name]), f"{USER}@{HOST}", PROBES[name],
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
        output = result.stdout
        status = "ok" if result.returncode == 0 else "ssh_error"
        return {
            "status": status,
            "returncode": result.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "latest_step": latest_step(output),
            "error_match_count": sum(
                bool(re.search(ERROR_PATTERN, line, re.IGNORECASE))
                for line in output.splitlines()
            ),
            "output": output,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "status": "timeout",
            "returncode": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "latest_step": None,
            "error_match_count": 0,
            "output": (error.stdout or "") if isinstance(error.stdout, str) else "",
        }


def latest_step(output: str) -> int | None:
    matches = re.findall(r'"step"\s*:\s*(\d+)', output)
    return int(matches[-1]) if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=25)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cycles < 1 or args.interval_seconds < 1 or args.timeout_seconds < 1:
        parser.error("cycles, interval-seconds and timeout-seconds must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    monotonic_start = time.monotonic()
    previous_steps: dict[str, int | None] = {}
    for cycle in range(args.cycles):
        scheduled = monotonic_start + cycle * args.interval_seconds
        delay = scheduled - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        record = {
            "cycle": cycle,
            "cycles": args.cycles,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "scheduled_elapsed_seconds": cycle * args.interval_seconds,
            "nodes": {},
        }
        for name in NODES:
            probe = ssh_probe(name, args.timeout_seconds)
            step = probe["latest_step"]
            prior = previous_steps.get(name)
            probe["step_advanced"] = (
                None if step is None or prior is None else step > prior
            )
            if step is not None:
                previous_steps[name] = step
            record["nodes"][name] = probe
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({
            "cycle": cycle,
            "captured_at_utc": record["captured_at_utc"],
            "status": {
                name: {
                    "status": probe["status"],
                    "latest_step": probe["latest_step"],
                    "step_advanced": probe["step_advanced"],
                    "error_match_count": probe["error_match_count"],
                }
                for name, probe in record["nodes"].items()
            },
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
