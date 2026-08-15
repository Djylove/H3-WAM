#!/usr/bin/env python3
"""Finalize C54 fresh-source state-aligned paired causal replication."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def one_sided_mcnemar(wins: int, losses: int) -> float:
    n=wins+losses
    if not n: return 1.0
    return sum(math.comb(n,k) for k in range(wins,n+1))/(2**n)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not all((args.root/f"shard{i}.COMPLETED").is_file() for i in range(4)): raise RuntimeError("C54 shards incomplete")
    selection=json.loads((args.root/"BRANCH_SELECTION.json").read_text()); jobs=[json.loads(x) for x in (args.root/"selection.jsonl").read_text().splitlines()]
    by_group=defaultdict(dict)
    for job in jobs:
        name=f"{job['ordinal']}_g{job['group_id']}_{job['arm']}_{job['suite'][7:]}_task{job['task']}_trial{job['trial']}_d{job['distance_replans']}"
        directory=args.root/"runs"/name; payload=json.loads((directory/"results.json").read_text()); episode=payload["tasks"][0]["episodes"][0]
        by_group[job["group_id"]][job["arm"]]=(job,payload,episode)
    expected_sha="d2f3a812eb1d4921efd6f2f9ee6f7f4f2736c516d338168b8110df281960907c"
    exact_state=exact_candidate0=seeds_valid=ranker_valid=score_valid=True; selected_nonzero=0; pairs=[]; suite=defaultdict(lambda:{"groups":0,"candidate0":0,"dense_bestof4":0})
    for group_id,arms in sorted(by_group.items()):
        if set(arms)!={"candidate0","dense_bestof4"}: raise ValueError(f"C54 group {group_id} arms mismatch")
        cj,cp,ce=arms["candidate0"]; dj,dp,de=arms["dense_bestof4"]
        ct=np.load(ce["trajectory"]); dt=np.load(de["trajectory"]); exact_state &= np.array_equal(ct["sim_state"][0],dt["sim_state"][0])
        exact_candidate0 &= np.array_equal(ct["policy_actions"][0],np.asarray(de["first_consequence_candidate0_chunk"],dtype=np.float32))
        expected=[int(cj["first_policy_noise_seed"])]+[int(cj["continuation_policy_noise_seed_base"])+i for i in range(max(0,len(ce["replan_noise_seeds"])-1))]
        expected_dense=[int(dj["first_policy_noise_seed"])]+[int(dj["continuation_policy_noise_seed_base"])+i for i in range(max(0,len(de["replan_noise_seeds"])-1))]
        seeds_valid &= ce["replan_noise_seeds"]==expected and de["replan_noise_seeds"]==expected_dense and de["consequence_candidate_seeds"]==[[int(dj["first_policy_noise_seed"])+x for x in (0,1_000_000,2_000_000,3_000_000)]]
        ranker_valid &= de["action_ranker_types"]==["c51_dense_value"] and de["consequence_ranker_sha256s"]==[expected_sha] and len(de["consequence_selected_indices"])==1
        score_valid &= len(de["consequence_score_ranges"])==1 and de["consequence_score_ranges"][0]>1e-6
        selected_nonzero+=int(de["consequence_selected_indices"][0]!=0)
        c=bool(ce["success"]); d=bool(de["success"]); suite[cj["suite"]]["groups"]+=1; suite[cj["suite"]]["candidate0"]+=c; suite[cj["suite"]]["dense_bestof4"]+=d
        pairs.append({"group_id":group_id,"suite":cj["suite"],"task":cj["task"],"trial":cj["trial"],"distance_replans":cj["distance_replans"],"candidate0_success":c,"dense_bestof4_success":d,"selected_index":de["consequence_selected_indices"][0]})
    groups=len(pairs); c_success=sum(p["candidate0_success"] for p in pairs); d_success=sum(p["dense_bestof4_success"] for p in pairs); wins=sum(p["dense_bestof4_success"] and not p["candidate0_success"] for p in pairs); losses=sum(p["candidate0_success"] and not p["dense_bestof4_success"] for p in pairs); gain=(d_success-c_success)/groups; pvalue=one_sided_mcnemar(wins,losses)
    suite_safety=all(v["dense_bestof4"]/v["groups"]>=v["candidate0"]/v["groups"]-.05 for v in suite.values())
    gate={"all_start_states_exact":exact_state,"all_candidate0_chunks_exact":exact_candidate0,"all_seed_schedules_exact":seeds_valid,"all_ranker_identities_exact":ranker_valid,"all_scores_vary":score_valid,"nonzero_selected_at_least_20_percent":selected_nonzero/groups>=.20,"absolute_success_gain_at_least_0_05":gain>=.05,"paired_net_wins_at_least_5":wins-losses>=5,"one_sided_exact_mcnemar_p_at_most_0_05":pvalue<=.05,"no_suite_regresses_by_more_than_0_05":suite_safety}; gate["passed"]=all(gate.values())
    report={"format":"h3wam-c54-state-aligned-replication-v1","status":"PASS_C54_STATE_ALIGNED_REPLICATION" if gate["passed"] else "FAIL_C54_STATE_ALIGNED_REPLICATION","permission":"GO_PROGRESS_TRIGGER_RESEARCH" if gate["passed"] else "NO_GO_DENSE_VALUE_ONLINE","claim_boundary":"Fresh visual sources and state-aligned causal branches; branch timing still uses parent hindsight and is not deployable.","results":{"sources":selection["sources"],"groups":groups,"candidate0_successes":c_success,"dense_bestof4_successes":d_success,"absolute_gain":gain,"wins":wins,"losses":losses,"ties":groups-wins-losses,"one_sided_exact_mcnemar_p":pvalue,"selected_nonzero":selected_nonzero,"per_suite":dict(suite),"pairs":pairs},"gate":gate}
    temporary=args.output.with_name(f".{args.output.name}.{os.getpid()}.partial"); temporary.write_text(json.dumps(report,indent=2)+"\n"); os.replace(temporary,args.output); print(json.dumps({"status":report["status"],"permission":report["permission"],"results":{k:v for k,v in report["results"].items() if k!="pairs"},"gate":gate},indent=2))


if __name__=="__main__": main()
