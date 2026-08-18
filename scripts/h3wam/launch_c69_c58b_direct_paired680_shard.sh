#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT must be an immutable rollout snapshot}"
root="${C69_C58B_ROLLOUT_ROOT:?C69_C58B_ROLLOUT_ROOT is required}"
shard="${ROLLOUT_SHARD_INDEX:?ROLLOUT_SHARD_INDEX is required}"
total="${ROLLOUT_NUM_SHARDS:?ROLLOUT_NUM_SHARDS is required}"
auth="${root}/AUTHORIZATION.json"
manifest="${root}/jobs.jsonl"
policy_python="${workspace}/runtime/h3-int8-native/bin/python"
sim_python="${workspace}/runtime/conda-py311/bin/python"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
h3_model="${workspace}/models/MiniMax-H3"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
marker="${root}/SHARD_$(printf '%02d' "${shard}")_COMPLETE.json"

for path in "${auth}" "${manifest}" "${project}/SOURCE_FREEZE.json" "${policy_python}" "${sim_python}" \
  "${h3_checkpoint}" "${h3_model}" "${source_manifest}" "${cache_root}/stats.pt"; do
  [[ -e "${path}" ]] || { echo "missing direct-pair input: ${path}" >&2; exit 2; }
done
[[ ! -e "${marker}" && ! -e "${root}/COMPLETED.json" ]] || { echo "shard already closed" >&2; exit 2; }

readarray -t gate_data < <("${sim_python}" - "${auth}" "${manifest}" "${project}" "${shard}" "${total}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as f:
  while c:=f.read(16*1024*1024): d.update(c)
 return d.hexdigest()
a=json.load(open(sys.argv[1])); jobs=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
project=Path(sys.argv[3]); shard,total=map(int,sys.argv[4:6])
assert a['format']=='h3wam-c69-c58b-direct-paired680-authorization-v1'
assert a['status']=='AUTHORIZED_DIRECT_PAIRED_RECHECK'
assert a['permission']=='GO_1360_FRESH_PROCESSES_NO_INTERMEDIATE_PROMOTION'
assert len(jobs)==1360 and sha(sys.argv[2])==a['manifest_sha256']
assert sha(project/'SOURCE_FREEZE.json')==a['source_freeze_sha256']
assert not (os.stat(project).st_mode & 0o222)
for row in a['endpoints'].values(): assert sha(row['checkpoint'])==row['checkpoint_sha256']
for row in a['inner_gates'].values(): assert sha(row['path'])==row['sha256']
assert 0<=shard<total
print(a['inner_gates']['c69']['path'])
print(a['inner_gates']['c58b']['path'])
PY
)
c69_gate="${gate_data[0]}"; c58_gate="${gate_data[1]}"

lock="${root}/.launcher-shard$(printf '%02d' "${shard}").lock"
mkdir "${lock}" 2>/dev/null || { echo "launcher lock already held" >&2; exit 75; }
trap 'rmdir "${lock}" 2>/dev/null || true' EXIT
mkdir -p "${root}/logs"
cd "${project}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PROJECT_ROOT="${project}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export H3WAM_FASTWAM_SOURCE_ROOT="${project}/third_party/FastWAM/src/fastwam/models/wan22"

run_worker() {
  local gpu="$1" job_id pair_id arm suite task trial assigned checkpoint output log
  while IFS=$'\t' read -r job_id pair_id arm suite task trial assigned checkpoint output; do
    [[ "${assigned}" == "${gpu}" ]] || continue
    [[ ! -e "${output}" ]] || { echo "refusing reused output ${output}" >&2; return 1; }
    mkdir -p "${output}"
    log="${root}/logs/shard$(printf '%02d' "${shard}")_job$(printf '%04d' "${job_id}")_${arm}.log"
    common=(--policy-python "${policy_python}" --checkpoint "${checkpoint}" --cache-root "${cache_root}"
      --h3-checkpoint "${h3_checkpoint}" --h3-model "${h3_model}" --dreamwam-source-manifest "${source_manifest}"
      --device cuda:0 --suite "${suite}" --task-ids "${task}" --trial-indices "${trial}"
      --max-steps 400 --wait-steps 30 --replan-steps 8 --action-horizon 32
      --h3-feature-audio-horizon 32 --target-latent-frames 12 --model-evaluations 10 --seed 42
      --normalized-action-pre-clamp --save-trajectories --output-dir "${output}")
    if [[ "${arm}" == c69_action_only ]]; then
      policy=(--policy h3_fact_online_int8 --c67-c69-attribution-authorization "${c69_gate}")
    else
      policy=(--policy h3_fastwam_online_int8 --c58b-balanced80-ready "${c58_gate}")
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHON_BIN="${sim_python}" SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site" \
      bash "${project}/scripts/h3wam/run_cloud_libero.sh" "${sim_python}" "${project}/scripts/h3wam/rollout_libero.py" \
      "${policy[@]}" "${common[@]}" >"${log}" 2>&1
  done < <("${sim_python}" - "${manifest}" "${shard}" "${total}" <<'PY'
import json,sys
s,n=map(int,sys.argv[2:4])
for line in open(sys.argv[1]):
 r=json.loads(line)
 if r['pair_id']%n==s:
  gpu=(r['pair_id']//n)%8
  print('\t'.join(map(str,(r['job_id'],r['pair_id'],r['arm'],r['suite'],r['tasks'][0],r['trials'][0],gpu,r['checkpoint'],r['output']))))
PY
)
}

pids=(); for gpu in 0 1 2 3 4 5 6 7; do run_worker "${gpu}" & pids+=("$!"); done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "one or more workers failed" >&2; exit 1; }

"${sim_python}" - "${manifest}" "${marker}" "${shard}" "${total}" <<'PY'
import json,os,sys
from pathlib import Path
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
out=Path(sys.argv[2]); shard,total=map(int,sys.argv[3:5]); selected=[r for r in rows if r['pair_id']%total==shard]
for row in selected:
 d=json.load(open(Path(row['output'])/'results.json'))
 eps=[e for t in d.get('tasks',[]) for e in t.get('episodes',[])]
 assert len(eps)==1 and d['task_ids']==row['tasks'] and d['trial_indices']==row['trials']
tmp=out.with_name('.'+out.name+'.partial'); tmp.write_text(json.dumps({'status':'COMPLETE','shard':shard,'jobs':len(selected),'pairs':len(selected)//2},indent=2)+'\n'); os.replace(tmp,out)
PY
