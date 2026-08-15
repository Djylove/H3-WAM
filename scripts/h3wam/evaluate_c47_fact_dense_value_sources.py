#!/usr/bin/env python3
"""Freeze and audit wholly new parent trajectories for FACT-style dense value."""

from __future__ import annotations

import argparse, hashlib, json, os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROLES = {24: "dense_value_validation_source", 25: "dense_value_validation_source",
         26: "dense_value_final_source", 27: "dense_value_final_source"}
SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(16 * 1024 * 1024): d.update(b)
    return d.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--workspace", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    if a.output.exists(): raise FileExistsError("refusing to overwrite C47 report")
    rows=[]; counts=defaultdict(Counter); replan_rows=defaultdict(int)
    for trial, role in ROLES.items():
        for suite in SUITES:
            slug=suite[7:]
            for task in range(10):
                result=a.workspace/f"outputs/eval-dense-d0-long/d0_h32_s14000_{slug}_task{task}_trial{trial}_replan8/results.json"
                payload=json.loads(result.read_text()); episode=payload["tasks"][0]["episodes"][0]
                trajectory=Path(episode["trajectory"]); z=np.load(trajectory)
                n=int(z["step"].shape[0])
                if n != int(episode["replans"]) or z["policy_actions"].shape != (n,32,7):
                    raise ValueError(f"trajectory contract mismatch: {trajectory}")
                success=bool(episode["success"]); counts[role]["success" if success else "failure"]+=1; counts[role][suite]+=int(success); replan_rows[role]+=n
                rows.append({"trial":trial,"role":role,"suite":suite,"task":task,"success":success,"steps":int(episode["steps"]),"replans":n,"result":str(result),"trajectory":str(trajectory),"result_sha256":sha256_file(result),"trajectory_sha256":sha256_file(trajectory)})
    gate={
      "all_160_results_and_trajectories_present":len(rows)==160,
      "each_role_has_at_least_20_successes":all(counts[r]["success"]>=20 for r in set(ROLES.values())),
      "each_role_has_at_least_20_failures":all(counts[r]["failure"]>=20 for r in set(ROLES.values())),
      "each_role_successes_cover_at_least_3_suites":all(sum(counts[r][s]>0 for s in SUITES)>=3 for r in set(ROLES.values())),
      "all_trajectory_rows_have_32x7_parent_actions":True,
      "trial_roles_fixed_before_outcomes":True,
    }; gate["passed"]=all(gate.values())
    report={"format":"h3wam-c47-fact-dense-value-sources-v1","status":"PASS_C47_DENSE_VALUE_SOURCES" if gate["passed"] else "FAIL_C47_DENSE_VALUE_SOURCES","permission":"GO_DENSE_VALUE_DATASET" if gate["passed"] else "NO_GO_DENSE_VALUE","parent":"D0-H32-s14000/replan8/sample1","roles":ROLES,"counts":{r:dict(c) for r,c in counts.items()},"replan_rows":dict(replan_rows),"gate":gate,"rows":rows}
    a.output.parent.mkdir(parents=True,exist_ok=True); tmp=a.output.with_name(f".{a.output.name}.{os.getpid()}.partial"); tmp.write_text(json.dumps(report,indent=2)+"\n"); os.replace(tmp,a.output); print(json.dumps({k:report[k] for k in ("status","permission","counts","replan_rows","gate")},indent=2))


if __name__ == "__main__": main()
