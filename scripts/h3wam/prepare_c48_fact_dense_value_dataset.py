#!/usr/bin/env python3
"""Freeze C42/C47 trajectories into FACT-style dense value windows."""

from __future__ import annotations

import argparse, hashlib, json, os, sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from fastwam.models.h3wam import libero_observation_state


FORMAT = "h3wam-c48-fact-dense-value-dataset-v1"


def sha256_file(path: Path) -> str:
    d=hashlib.sha256()
    with path.open("rb") as f:
        while b:=f.read(16*1024*1024): d.update(b)
    return d.hexdigest()


def proprio(archive, prefix: str, index: int | None = None) -> torch.Tensor:
    def get(name):
        value=archive[f"{prefix}{name}"]
        return value if index is None else value[index]
    return torch.as_tensor(libero_observation_state({
        "eef_pos":get("eef_pos"), "eef_quat":get("eef_quat"),
        "gripper_qpos":get("gripper_qpos"),
    }),dtype=torch.float32)


def episode_rows(c42: dict, c47: dict) -> list[dict]:
    rows=[]
    for row in c42["rows"]:
        rows.append({**row,"dense_value_split":"train"})
    for row in c47["rows"]:
        split="validation" if int(row["trial"]) in (24,25) else "final"
        rows.append({**row,"dense_value_split":split})
    return rows


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--c42",type=Path,required=True); p.add_argument("--c47",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args()
    if a.output_root.exists(): raise FileExistsError("refusing to overwrite C48 root")
    c42=json.loads(a.c42.read_text()); c47=json.loads(a.c47.read_text())
    if c42.get("status")!="PASS_C42_FRESH_PARENT_SOURCE_EXPANSION" or c47.get("status")!="PASS_C47_DENSE_VALUE_SOURCES": raise ValueError("C48 source gate is not PASS")
    observations=[]; samples=[]; counts={s:{"episodes":0,"samples":0,"success_samples":0,"failure_samples":0} for s in ("train","validation","final")}
    for episode_id,row in enumerate(episode_rows(c42,c47)):
        result_path=Path(row["path"] if "path" in row else row["result"])
        result=json.loads(result_path.read_text()); episode=result["tasks"][0]["episodes"][0]
        trajectory=Path(episode["trajectory"]); z=np.load(trajectory)
        n=int(z["step"].shape[0]); terminal_step=int(z["terminal_step"])
        if z["policy_actions"].shape!=(n,32,7) or int(episode["replans"])!=n: raise ValueError(f"bad trajectory {trajectory}")
        split=row["dense_value_split"]; counts[split]["episodes"]+=1
        obs_ids=[]
        for i in range(n):
            oid=len(observations); obs_ids.append(oid); observations.append({"observation_id":oid,"episode_id":episode_id,"split":split,"trajectory":str(trajectory),"kind":"row","row_index":i,"step":int(z["step"][i]),"task_language":result["tasks"][0]["task"]})
        terminal_id=len(observations); observations.append({"observation_id":terminal_id,"episode_id":episode_id,"split":split,"trajectory":str(trajectory),"kind":"terminal","row_index":None,"step":terminal_step,"task_language":result["tasks"][0]["task"]})
        for i in range(n):
            start=int(z["step"][i]); executed=[]
            for j in range(i,n):
                segment_end=int(z["step"][j+1]) if j+1<n else terminal_step
                take=max(0,min(8,segment_end-int(z["step"][j])))
                executed.append(torch.as_tensor(z["policy_actions"][j,:take],dtype=torch.float32))
                if sum(len(x) for x in executed)>=32: break
            actions=torch.cat(executed,dim=0)[:32] if executed else torch.empty(0,7)
            executed_steps=len(actions); padded=torch.zeros(32,7); padded[:executed_steps]=actions
            future_target=min(start+32,terminal_step)
            future_rows=np.flatnonzero(z["step"]==future_target)
            if len(future_rows):
                fi=int(future_rows[0]); future_id=obs_ids[fi]; future_state=proprio(z,"",fi)
            else:
                future_id=terminal_id; future_state=proprio(z,"terminal_",None)
            success=bool(episode["success"])
            base=max(0.0,float(terminal_step-future_target)/max(float(terminal_step),1.0))
            value_target=base+(0.0 if success else 1.0)
            samples.append({"sample_id":len(samples),"episode_id":episode_id,"split":split,"suite":row["suite"],"task":int(row["task"]),"trial":int(row["trial"]),"success":success,"current_observation_id":obs_ids[i],"future_observation_id":future_id,"current_step":start,"future_step":future_target,"terminal_step":terminal_step,"current_proprio":proprio(z,"",i),"future_proprio":future_state,"executed_actions":padded,"action_is_pad":torch.arange(32)>=executed_steps,"executed_action_steps":executed_steps,"value_target":value_target,"failure_penalty":0.0 if success else 1.0})
            counts[split]["samples"]+=1; counts[split]["success_samples" if success else "failure_samples"]+=1
    if [x["observation_id"] for x in observations]!=list(range(len(observations))) or [x["sample_id"] for x in samples]!=list(range(len(samples))): raise RuntimeError("C48 ids are not contiguous")
    a.output_root.mkdir(parents=True); obs_path=a.output_root/"observations.jsonl"; obs_path.write_text("".join(json.dumps(x)+"\n" for x in observations))
    dataset={"format":FORMAT,"sources":{"c42_sha256":sha256_file(a.c42),"c47_sha256":sha256_file(a.c47)},"value_contract":"FACT normalized time-to-go after the clean 32-step future window; +1 penalty for all failed parent episodes","action_contract":"concatenate actually executed replan8 prefixes; zero-mask only terminal tail","counts":counts,"observations":len(observations),"samples":samples}
    tmp=a.output_root/".dataset.pt.partial"; torch.save(dataset,tmp); os.replace(tmp,a.output_root/"dataset.pt")
    report={"format":FORMAT,"status":"PASS_C48_DENSE_VALUE_DATASET","dataset_sha256":sha256_file(a.output_root/"dataset.pt"),"observations_sha256":sha256_file(obs_path),"counts":counts,"observations":len(observations),"samples":len(samples)}
    (a.output_root/"COMPLETED").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))


if __name__=="__main__": main()
