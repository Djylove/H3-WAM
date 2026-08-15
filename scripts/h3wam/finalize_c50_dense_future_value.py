#!/usr/bin/env python3
"""Select on C47 validation and evaluate C47 final exactly once."""

from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import torch

REPO_ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(REPO_ROOT/"src")); sys.path.insert(0,str(REPO_ROOT/"scripts"/"h3wam"))
from fastwam.models.h3wam import DenseTemporalFutureValueModel
from train_c50_dense_future_value_expert import metrics, sha


def main():
 p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--features",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda:0"); a=p.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 candidates=[]
 for completed in sorted(a.root.glob("*/COMPLETED")):
  report=json.loads(completed.read_text())
  for row in report["history"]:
   m=row["validation"]; margin=m["failure_mean_value"]-m["success_mean_value"]
   passed=m["value_rank_correlation"]>=.5 and margin>=.2 and m["shuffled_value_mse"]>=m["value_mse"]*1.01
   candidates.append({"arm":report["arm"],"seed":report["seed"],"step":row["step"],"checkpoint":str(completed.parent/"checkpoints"/f"step{row['step']}.pt"),"validation":m,"failure_success_margin":margin,"passed":passed})
 eligible=[x for x in candidates if x["passed"]]
 if not eligible:
  result={"format":"h3wam-c51-dense-future-value-final-v1","status":"FAIL_C50_VALIDATION_GATE","permission":"NO_GO_DENSE_VALUE_FINAL","claim_boundary":"C47 final was not read because no validation arm passed.","candidates":candidates}
 else:
  selected=min(eligible,key=lambda x:(x["validation"]["value_mse"],x["arm"]!="frozen_consequence",x["step"],x["seed"]))
  data=torch.load(a.dataset,map_location="cpu",weights_only=False); feat=torch.load(a.features,map_location="cpu",weights_only=False); cp=torch.load(selected["checkpoint"],map_location="cpu",weights_only=False); samples=data["samples"]; final=[i for i,r in enumerate(samples) if r["split"]=="final"]
  device=torch.device(a.device); model=DenseTemporalFutureValueModel(**cp["model_kwargs"]).to(device); model.load_state_dict(cp["model"],strict=True); mean=cp["normalization"]["state_mean"].to(device); std=cp["normalization"]["state_std"].to(device); m=metrics(model,samples,final,feat["fact_layer49_projected"].float(),mean,std,device)
  targets=torch.tensor([samples[i]["value_target"]-1.0 for i in final]); train_targets=torch.tensor([r["value_target"]-1.0 for r in samples if r["split"]=="train"]); baseline=float(((targets-train_targets.mean())**2).mean()); margin=m["failure_mean_value"]-m["success_mean_value"]
  gate={"final_value_mse_beats_train_mean_by_20_percent":m["value_mse"]<=baseline*.8,"final_value_rank_correlation_at_least_0_5":m["value_rank_correlation"]>=.5,"final_failure_success_margin_at_least_0_2":margin>=.2,"final_shuffled_action_mse_at_least_1_percent_worse":m["shuffled_value_mse"]>=m["value_mse"]*1.01,"strict_checkpoint_restore":True}; gate["passed"]=all(gate.values())
  result={"format":"h3wam-c51-dense-future-value-final-v1","status":"PASS_C51_DENSE_VALUE_FINAL" if gate["passed"] else "FAIL_C51_DENSE_VALUE_FINAL","permission":"GO_FRESH_COUNTERFACTUAL_VALUE_RANKING" if gate["passed"] else "NO_GO_DENSE_VALUE_RANKING","claim_boundary":"Dense held-out trajectory value only; alternative-action ranking and online improvement remain unproven.","selected":selected,"final":m,"final_failure_success_margin":margin,"train_mean_baseline_mse":baseline,"gate":gate,"sources":{"dataset_sha256":sha(a.dataset),"features_sha256":sha(a.features),"checkpoint_sha256":sha(Path(selected["checkpoint"]))},"validation_candidates":candidates}
 tmp=a.output.with_name(f".{a.output.name}.{os.getpid()}.partial"); tmp.write_text(json.dumps(result,indent=2)+"\n"); os.replace(tmp,a.output); print(json.dumps({k:result[k] for k in result if k not in ("validation_candidates","candidates")},indent=2))


if __name__=="__main__": main()
