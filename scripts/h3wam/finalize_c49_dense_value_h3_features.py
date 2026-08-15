#!/usr/bin/env python3
"""Audit exact coverage and identity of the 32 C49 feature shards."""

from __future__ import annotations

import argparse, hashlib, json, os
from pathlib import Path

import torch


def sha(path):
 d=hashlib.sha256()
 with Path(path).open("rb") as f:
  while b:=f.read(16*1024*1024): d.update(b)
 return d.hexdigest()


def main():
 p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--observations",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 rows=a.observations.read_text().splitlines(); expected_sha=sha(a.observations); ids=[]; shards=[]
 for i in range(32):
  if not (a.root/f"node{i//8}.COMPLETED").is_file(): raise RuntimeError(f"node{i//8} incomplete")
  path=a.root/"shards"/f"shard{i}.pt"; q=torch.load(path,map_location="cpu",weights_only=False)
  if q.get("format")!="h3wam-c49-dense-value-h3-feature-shard-v1" or q["observations_sha256"]!=expected_sha or q["shard"]!=i or q["num_shards"]!=32: raise ValueError(f"shard{i} identity mismatch")
  this=q["observation_ids"].tolist(); h=q["fact_layer49_hidden"]
  if h.shape!=(len(this),1,32,5376) or not torch.isfinite(h.float()).all(): raise ValueError(f"shard{i} feature mismatch")
  if any(x%32!=i for x in this): raise ValueError(f"shard{i} assignment mismatch")
  ids.extend(this); shards.append({"shard":i,"observations":len(this),"sha256":sha(path),"bytes":path.stat().st_size})
 gate={"all_nodes_complete":True,"all_observations_exactly_once":sorted(ids)==list(range(len(rows))),"no_duplicate_observations":len(ids)==len(set(ids)),"all_features_finite_exact_shape":True}; gate["passed"]=all(gate.values())
 report={"format":"h3wam-c49-dense-value-h3-features-v1","status":"PASS_C49_DENSE_VALUE_H3_FEATURES" if gate["passed"] else "FAIL_C49_DENSE_VALUE_H3_FEATURES","permission":"GO_DENSE_FUTURE_VALUE_TRAINING" if gate["passed"] else "NO_GO_DENSE_VALUE_TRAINING","observations":len(rows),"observations_sha256":expected_sha,"total_bytes":sum(x["bytes"] for x in shards),"shards":shards,"gate":gate}
 tmp=a.output.with_name(f".{a.output.name}.{os.getpid()}.partial"); tmp.write_text(json.dumps(report,indent=2)+"\n"); os.replace(tmp,a.output); print(json.dumps({k:report[k] for k in ("status","permission","observations","total_bytes","gate")},indent=2))


if __name__=="__main__": main()
