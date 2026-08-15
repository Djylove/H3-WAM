#!/usr/bin/env python3
"""Train one preregistered FACT-style dense future/value expert arm."""

from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(REPO_ROOT/"src"))
from fastwam.models.h3wam import DenseTemporalFutureValueModel


def sha(path):
 d=hashlib.sha256()
 with Path(path).open("rb") as f:
  while b:=f.read(16*1024*1024): d.update(b)
 return d.hexdigest()


def metrics(model,samples,indices,features,mean,std,device,batch=128):
 sums={k:0.0 for k in ("future_h3_mse","future_state_mse","value_mse","shuffled_value_mse","action_sensitivity")}; pred=[]; target=[]; success=[]
 model.eval()
 with torch.inference_mode():
  for start in range(0,len(indices),batch):
   ids=indices[start:start+batch]; rows=[samples[i] for i in ids]
   cur=torch.stack([features[r["current_observation_id"]] for r in rows]).to(device); fut=torch.stack([features[r["future_observation_id"]] for r in rows]).to(device)
   state=(torch.stack([r["current_proprio"] for r in rows]).to(device)-mean)/std; fstate=(torch.stack([r["future_proprio"] for r in rows]).to(device)-mean)/std; actions=torch.stack([r["executed_actions"] for r in rows]).to(device); value=torch.tensor([r["value_target"]-1.0 for r in rows],device=device)
   out=model.forward_projected(state,cur,actions); sh=model.forward_projected(state,cur,actions.roll(1,0))
   n=len(rows); sums["future_h3_mse"]+=float(F.mse_loss(out["future_h3"],fut,reduction="sum"))/256; sums["future_state_mse"]+=float(F.mse_loss(out["future_state"],fstate,reduction="sum"))/8; sums["value_mse"]+=float(F.mse_loss(out["value"],value,reduction="sum")); sums["shuffled_value_mse"]+=float(F.mse_loss(sh["value"],value,reduction="sum")); sums["action_sensitivity"]+=float((sh["value"]-out["value"]).abs().sum()); pred.append(out["value"].cpu()); target.append(value.cpu()); success.extend(r["success"] for r in rows)
 n=len(indices); result={k:v/n for k,v in sums.items()}; p=torch.cat(pred); t=torch.cat(target); pr=torch.argsort(torch.argsort(p)).float(); tr=torch.argsort(torch.argsort(t)).float(); result["value_rank_correlation"]=float(torch.corrcoef(torch.stack((pr,tr)))[0,1]); s=torch.tensor(success,dtype=torch.bool); result["success_mean_value"]=float(p[s].mean()); result["failure_mean_value"]=float(p[~s].mean()); result["samples"]=n
 return result


def main():
 p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--features",type=Path,required=True); p.add_argument("--c38-checkpoint",type=Path,required=True); p.add_argument("--arm",choices=("joint","frozen_consequence"),required=True); p.add_argument("--steps",type=int,default=10000); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--seed",type=int,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--device",default="cuda:0"); a=p.parse_args()
 if a.output_root.exists(): raise FileExistsError(a.output_root)
 data=torch.load(a.dataset,map_location="cpu",weights_only=False); feat=torch.load(a.features,map_location="cpu",weights_only=False); c38=torch.load(a.c38_checkpoint,map_location="cpu",weights_only=False)
 if data.get("format")!="h3wam-c48-fact-dense-value-dataset-v1" or feat.get("format")!="h3wam-c49-dense-value-projected-features-v1" or c38.get("model_variant")!="temporal": raise ValueError("C50 input identity mismatch")
 features=feat["fact_layer49_projected"].float(); samples=data["samples"]; train=[i for i,r in enumerate(samples) if r["split"]=="train"]; val=[i for i,r in enumerate(samples) if r["split"]=="validation"]
 positive=[i for i in train if samples[i]["success"]]; negative=[i for i in train if not samples[i]["success"]]
 device=torch.device(a.device); torch.cuda.set_device(device); torch.manual_seed(a.seed)
 kwargs=c38["model_kwargs"]; model=DenseTemporalFutureValueModel(**kwargs).to(device); restored=model.load_state_dict(c38["models"]["conditioned"],strict=False)
 expected={"future_state_decoder.0.weight","future_state_decoder.0.bias","future_state_decoder.1.weight","future_state_decoder.1.bias","future_state_decoder.3.weight","future_state_decoder.3.bias","value_decoder.0.weight","value_decoder.0.bias","value_decoder.1.weight","value_decoder.1.bias","value_decoder.3.weight","value_decoder.3.bias"}
 if set(restored.missing_keys)!=expected or restored.unexpected_keys: raise ValueError(f"C38 partial restore mismatch: {restored}")
 if a.arm=="frozen_consequence":
  for name,param in model.named_parameters(): param.requires_grad_(name.startswith("future_state_decoder") or name.startswith("value_decoder"))
 params=[x for x in model.parameters() if x.requires_grad]; opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-2)
 mean=c38["normalization"]["state_mean"].to(device); std=c38["normalization"]["state_std"].to(device)
 g=torch.Generator(device="cpu").manual_seed(a.seed); a.output_root.mkdir(parents=True); (a.output_root/"checkpoints").mkdir(); history=[]; started=time.perf_counter()
 model.train()
 for step in range(1,a.steps+1):
  half=a.batch_size//2; ids=torch.cat((torch.randint(len(positive),(half,),generator=g),torch.randint(len(negative),(a.batch_size-half,),generator=g))); chosen=[positive[int(x)] if j<half else negative[int(x)] for j,x in enumerate(ids)]
  rows=[samples[i] for i in chosen]; cur=torch.stack([features[r["current_observation_id"]] for r in rows]).to(device); fut=torch.stack([features[r["future_observation_id"]] for r in rows]).to(device); state=(torch.stack([r["current_proprio"] for r in rows]).to(device)-mean)/std; fstate=(torch.stack([r["future_proprio"] for r in rows]).to(device)-mean)/std; actions=torch.stack([r["executed_actions"] for r in rows]).to(device); value=torch.tensor([r["value_target"]-1.0 for r in rows],device=device)
  out=model.forward_projected(state,cur,actions); h3=F.mse_loss(out["future_h3"],fut); state_loss=F.mse_loss(out["future_state"],fstate); value_loss=F.mse_loss(out["value"],value); loss=h3+.4*state_loss+.4*value_loss
  opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
  if step%1000==0 or step==a.steps:
   val_metric=metrics(model,samples,val,features,mean,std,device); row={"step":step,"train_loss":float(loss.detach()),"train_future_h3_mse":float(h3.detach()),"train_future_state_mse":float(state_loss.detach()),"train_value_mse":float(value_loss.detach()),"validation":val_metric,"elapsed_seconds":time.perf_counter()-started}; history.append(row)
   payload={"format":"h3wam-c50-dense-future-value-expert-v1","arm":a.arm,"seed":a.seed,"completed_steps":step,"model_kwargs":kwargs,"model":model.state_dict(),"optimizer":opt.state_dict(),"normalization":{"state_mean":mean.cpu(),"state_std":std.cpu(),"value":"raw_0_to_2_minus_1"},"sources":{"dataset_sha256":sha(a.dataset),"features_sha256":sha(a.features),"c38_sha256":sha(a.c38_checkpoint)},"history":history}
   path=a.output_root/"checkpoints"/f"step{step}.pt"; tmp=path.with_name(f".{path.name}.{os.getpid()}.partial"); torch.save(payload,tmp); os.replace(tmp,path); print(json.dumps(row),flush=True); model.train()
 report={"format":"h3wam-c50-dense-future-value-expert-v1","status":"COMPLETED_C50_TRAINING_ARM","claim_boundary":"Validation only; C47 final remains unread.","arm":a.arm,"seed":a.seed,"steps":a.steps,"train_samples":len(train),"validation_samples":len(val),"history":history}; (a.output_root/"COMPLETED").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__": main()
