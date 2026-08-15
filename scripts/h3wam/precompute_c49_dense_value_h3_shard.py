#!/usr/bin/env python3
"""Extract one immutable shard of live INT8 H3 features for C48 observations."""

from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fastwam.models.h3wam import H3Int8FeatureBackbone, H3Int8OnlineFeatureContract, H3Int8OnlineFeatureProvider, encode_h3_vae_condition_standalone, preprocess_libero_cameras


EXPECTED_H3_SHA256="e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
EXPECTED_SOURCE_SHA256="cab8876f067114dce41d16ca52cb0bafddf17da33c92d0adde5f11d7ac9555b9"


def sha(path):
 d=hashlib.sha256()
 with Path(path).open("rb") as f:
  while b:=f.read(16*1024*1024): d.update(b)
 return d.hexdigest()


def contexts_for(source,tasks):
 out=defaultdict(set)
 for line in Path(source).read_text().splitlines():
  row=json.loads(line)
  if row["task"] in tasks: out[row["task"]].add(row["context_id"])
 if set(out)!=tasks or any(len(x)!=1 for x in out.values()): raise ValueError("task/context mapping invalid")
 return {k:next(iter(v)) for k,v in out.items()}


def main():
 p=argparse.ArgumentParser(); p.add_argument("--observations",type=Path,required=True); p.add_argument("--cache-root",type=Path,required=True); p.add_argument("--source-manifest",type=Path,required=True); p.add_argument("--h3-checkpoint",type=Path,required=True); p.add_argument("--h3-model",type=Path,required=True); p.add_argument("--shard",type=int,required=True); p.add_argument("--num-shards",type=int,default=32); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda:0"); a=p.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 if not 0<=a.shard<a.num_shards: raise ValueError("invalid shard")
 if sha(a.h3_checkpoint)!=EXPECTED_H3_SHA256 or sha(a.source_manifest)!=EXPECTED_SOURCE_SHA256: raise ValueError("H3/source identity mismatch")
 all_rows=[json.loads(x) for x in a.observations.read_text().splitlines()]
 rows=[x for x in all_rows if int(x["observation_id"])%a.num_shards==a.shard]
 if not rows: raise ValueError("empty shard")
 device=torch.device(a.device); torch.cuda.set_device(device)
 task_map=contexts_for(a.source_manifest,{x["task_language"] for x in rows}); ctx={}
 for task,cid in task_map.items():
  q=torch.load(a.cache_root/"contexts"/f"{cid}.pt",map_location="cpu",weights_only=False); ctx[task]={"id":cid,"context":q["context"].to(device=device,dtype=torch.bfloat16),"tags":q["token_tags"].to(device)}
 from diffusers import AutoencoderKLMiniMaxH3
 backbone=H3Int8FeatureBackbone.from_checkpoint(a.h3_checkpoint).to(device).eval(); backbone.requires_grad_(False)
 provider=H3Int8OnlineFeatureProvider(backbone,H3Int8OnlineFeatureContract(layers=(49,),action_horizon=32,target_latent_frames=12,video_timestep=1.0,condition_video_timestep=1.0,capture_compatibility="none"))
 vae=AutoencoderKLMiniMaxH3.from_pretrained(a.h3_model,subfolder="vae",torch_dtype=torch.float32,low_cpu_mem_usage=True).to(device).eval(); vae.requires_grad_(False)
 ids=[]; hidden=[]; context_ids=[]; last_path=None; archive=None; started=time.perf_counter()
 for pos,row in enumerate(rows):
  if row["trajectory"]!=last_path:
   if archive is not None: archive.close()
   archive=np.load(row["trajectory"]); last_path=row["trajectory"]
  if row["kind"]=="row":
   i=int(row["row_index"]); agent=archive["agentview_image"][i]; wrist=archive["wristview_image"][i]
  else: agent=archive["terminal_agentview_image"]; wrist=archive["terminal_wristview_image"]
  c=ctx[row["task_language"]]; pixels=preprocess_libero_cameras(agent,wrist); video=pixels.mul(255).round().to(torch.uint8).permute(0,3,1,2).unsqueeze(2).to(device)
  with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.float16): frame=encode_h3_vae_condition_standalone(vae,video,(.485,.456,.406),(.229,.224,.225)).to(device=device,dtype=torch.float32)
  with torch.inference_mode(): h=provider(frame,c["context"],c["tags"])[0]
  if h.shape[1]>32: h=F.adaptive_avg_pool1d(h.transpose(1,2),32).transpose(1,2)
  h=h.to(torch.bfloat16).cpu()
  if h.shape!=(1,32,5376) or not torch.isfinite(h.float()).all(): raise RuntimeError(f"bad H3 feature {row['observation_id']}")
  ids.append(int(row["observation_id"])); hidden.append(h); context_ids.append(c["id"])
  if (pos+1)%50==0 or pos+1==len(rows): print(json.dumps({"shard":a.shard,"completed":pos+1,"total":len(rows),"seconds":round(time.perf_counter()-started,2)}),flush=True)
 if archive is not None: archive.close()
 result={"format":"h3wam-c49-dense-value-h3-feature-shard-v1","observations_sha256":sha(a.observations),"shard":a.shard,"num_shards":a.num_shards,"observation_ids":torch.tensor(ids),"context_ids":context_ids,"fact_layer49_hidden":torch.stack(hidden),"h3_checkpoint_sha256":EXPECTED_H3_SHA256}
 a.output.parent.mkdir(parents=True,exist_ok=True); tmp=a.output.with_name(f".{a.output.name}.{os.getpid()}.partial"); torch.save(result,tmp); os.replace(tmp,a.output)


if __name__=="__main__": main()
