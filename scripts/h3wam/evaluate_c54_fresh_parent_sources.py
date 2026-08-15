#!/usr/bin/env python3
"""Audit wholly new C54 parent sources before freezing d3/d5 branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


TRIALS = (29, 30, 31, 32)
SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    rows=[]; eligible=[]; counts=Counter()
    for trial in TRIALS:
        for suite in SUITES:
            for task in range(10):
                result = args.workspace / "outputs/eval-dense-d0-long" / f"d0_h32_s14000_{suite[7:]}_task{task}_trial{trial}_replan8/results.json"
                payload=json.loads(result.read_text()); episode=payload["tasks"][0]["episodes"][0]
                trajectory=Path(episode["trajectory"]); archive=np.load(trajectory)
                replans=int(archive["step"].shape[0])
                if replans != int(episode["replans"]) or archive["policy_actions"].shape != (replans,32,7):
                    raise ValueError(f"C54 trajectory mismatch: {trajectory}")
                success=bool(episode["success"]); is_eligible=success and replans>5
                row={"trial":trial,"suite":suite,"task":task,"success":success,"eligible":is_eligible,"steps":int(episode["steps"]),"replans":replans,"result":str(result),"trajectory":str(trajectory),"result_sha256":sha256_file(result),"trajectory_sha256":sha256_file(trajectory)}
                rows.append(row); counts["success" if success else "failure"]+=1
                if is_eligible: eligible.append(row); counts[f"eligible_{suite}"]+=1
    eligible_suites=sorted(suite for suite in SUITES if counts[f"eligible_{suite}"]>0)
    gate={"all_160_sources_complete":len(rows)==160,"at_least_40_eligible_successes":len(eligible)>=40,"eligible_successes_cover_at_least_3_suites":len(eligible_suites)>=3,"all_trajectory_action_rows_are_32x7":True}; gate["passed"]=all(gate.values())
    report={"format":"h3wam-c54-fresh-parent-sources-v1","status":"PASS_C54_FRESH_PARENT_SOURCES" if gate["passed"] else "FAIL_C54_FRESH_PARENT_SOURCES","permission":"GO_C54_FREEZE_STATE_ALIGNED_BRANCHES" if gate["passed"] else "NO_GO_C54_BRANCHES","counts":dict(counts),"eligible_sources":len(eligible),"eligible_suites":eligible_suites,"gate":gate,"rows":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True); temporary=args.output.with_name(f".{args.output.name}.{os.getpid()}.partial"); temporary.write_text(json.dumps(report,indent=2)+"\n"); os.replace(temporary,args.output); print(json.dumps({k:report[k] for k in ("status","permission","counts","eligible_sources","eligible_suites","gate")},indent=2))


if __name__ == "__main__": main()
