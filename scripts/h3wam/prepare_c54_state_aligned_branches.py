#!/usr/bin/env python3
"""Freeze every eligible C54 parent into d3/d5 paired branch jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


DISTANCES = (3, 5)
OFFSETS = (0, 1_000_000, 2_000_000, 3_000_000)


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        while chunk:=stream.read(16*1024*1024): digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--sources",type=Path,required=True); parser.add_argument("--root",type=Path,required=True); args=parser.parse_args()
    sources=json.loads(args.sources.read_text())
    if sources.get("permission")!="GO_C54_FREEZE_STATE_ALIGNED_BRANCHES": raise ValueError("C54 source gate did not pass")
    if not (args.root/"preregistration.json").is_file(): raise ValueError("C54 preregistration is absent")
    selection=args.root/"selection.jsonl"
    if selection.exists(): raise FileExistsError(selection)
    groups=[]; jobs=[]; suite_counts=Counter()
    eligible=sorted((row for row in sources["rows"] if row["eligible"]),key=lambda row:(row["suite"],row["task"],row["trial"]))
    for source in eligible:
        trajectory=Path(source["trajectory"]); archive=np.load(trajectory); state_count=int(archive["step"].shape[0])
        for distance in DISTANCES:
            index=state_count-distance; start_step=int(archive["step"][index]); group_id=len(groups)
            continuation=254_000_000+group_id*100_000
            base_seed=42+int(source["task"])*100_000+int(source["trial"])*1_000+index
            group={"group_id":group_id,"source_episode":f"{source['suite']}:task{source['task']}:trial{source['trial']}","suite":source["suite"],"task":int(source["task"]),"trial":int(source["trial"]),"distance_replans":distance,"trajectory":str(trajectory),"index":index,"start_step":start_step,"first_policy_noise_seed":base_seed,"continuation_policy_noise_seed_base":continuation}
            groups.append(group); suite_counts[source["suite"]]+=1
            for arm in ("candidate0","dense_bestof4"):
                jobs.append({"ordinal":len(jobs),**group,"arm":arm})
    if len(groups)!=2*len(eligible) or len(jobs)!=2*len(groups): raise RuntimeError("C54 branch inventory mismatch")
    if len(groups)<80 or len(suite_counts)<3: raise ValueError("C54 frozen branch yield below preregistration")
    selection.write_text("".join(json.dumps(row)+"\n" for row in jobs))
    frozen={"format":"h3wam-c54-state-aligned-branch-selection-v1","status":"PASS_C54_BRANCH_SELECTION","sources":len(eligible),"groups":len(groups),"branches":len(jobs),"suite_group_counts":dict(suite_counts),"offsets":list(OFFSETS),"sources_sha256":sha256_file(args.sources),"selection_sha256":sha256_file(selection),"groups_detail":groups}
    temporary=(args.root/"BRANCH_SELECTION.json").with_name(f".BRANCH_SELECTION.json.{os.getpid()}.partial"); temporary.write_text(json.dumps(frozen,indent=2)+"\n"); os.replace(temporary,args.root/"BRANCH_SELECTION.json"); (args.root/"runs").mkdir(exist_ok=True); (args.root/"logs").mkdir(exist_ok=True); print(json.dumps({k:frozen[k] for k in ("status","sources","groups","branches","suite_group_counts","selection_sha256")},indent=2))


if __name__=="__main__": main()
