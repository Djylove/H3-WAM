#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT must be the complete immutable rollout snapshot}"
root="${C67_C69_ROLLOUT_ROOT:?Set the prepared C67/C69 rollout root}"
shard_index="${ROLLOUT_SHARD_INDEX:?Set zero-based rollout shard index}"
num_shards="${ROLLOUT_NUM_SHARDS:?Set total rollout shards}"
authorization="${root}/AUTHORIZATION.json"
manifest="${root}/jobs.jsonl"
policy_python="${POLICY_PYTHON:-${workspace}/runtime/h3-int8-native/bin/python}"
sim_python="${SIM_PYTHON:-${workspace}/runtime/conda-py311/bin/python}"
h3_checkpoint="${H3_CHECKPOINT:-${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
h3_model="${H3_MODEL:-${workspace}/models/MiniMax-H3}"
source_manifest="${SOURCE_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl}"
train_manifest="${TRAIN_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}"
cache_root="${CACHE_ROOT:-${workspace}/data/v7_dense_h3_cache}"
c48_dataset="${C48_DATASET:-${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt}"
c48_observations="${C48_OBSERVATIONS:-${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl}"
c59_root="${C59_ROOT:-${workspace}/eval/c59-fact-failure-active-overlay-v1}"
marker="${root}/SHARD_$(printf '%02d' "${shard_index}")_COMPLETE.json"

[[ "${shard_index}" =~ ^[0-9]+$ && "${num_shards}" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid rollout shard integers" >&2; exit 2;
}
(( shard_index < num_shards )) || { echo "rollout shard index out of range" >&2; exit 2; }
for path in "${project}/SOURCE_FREEZE.json" "${authorization}" "${manifest}" \
  "${policy_python}" "${sim_python}" "${h3_checkpoint}" "${h3_model}" \
  "${source_manifest}" "${train_manifest}" "${cache_root}/stats.pt" \
  "${c48_dataset}" "${c48_observations}" "${c59_root}/COMPLETED.json" \
  "${c59_root}/sample_labels.jsonl"; do
  [[ -e "${path}" ]] || { echo "missing C67/C69 rollout input: ${path}" >&2; exit 2; }
done
[[ ! -e "${root}/COMPLETED.json" && ! -e "${root}/INVALID.json" && ! -e "${marker}" ]] || {
  echo "C67/C69 rollout shard cannot start from completed/invalid/reused root" >&2; exit 2;
}

freeze_sha="$(${sim_python} - "${authorization}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get("source_freeze",{}).get("sha256",""))
PY
)"
"${sim_python}" "${project}/scripts/h3wam/freeze_c67_rollout_source.py" \
  --verify --snapshot "${project}" --expected-manifest-sha256 "${freeze_sha}"

"${sim_python}" - "${authorization}" "${manifest}" "${project}" \
  "${source_manifest}" "${train_manifest}" "${cache_root}/stats.pt" \
  "${c48_dataset}" "${c48_observations}" "${c59_root}/COMPLETED.json" \
  "${c59_root}/sample_labels.jsonl" "${h3_checkpoint}" "${shard_index}" "${num_shards}" <<'PY'
import hashlib,json,stat,sys
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        while chunk:=stream.read(16*1024*1024): digest.update(chunk)
    return digest.hexdigest()

auth_path,manifest_path,project=map(Path,sys.argv[1:4])
data_paths=list(map(Path,sys.argv[4:11])); h3=Path(sys.argv[11])
shard_index,num_shards=map(int,sys.argv[12:14])
auth=json.load(open(auth_path)); jobs=[json.loads(x) for x in open(manifest_path) if x.strip()]
expected_data={
 "source_manifest_sha256":"cab8876f067114dce41d16ca52cb0bafddf17da33c92d0adde5f11d7ac9555b9",
 "demo_manifest_sha256":"b0d611c21059fa7da6fb08162b03efadd59aff68354bb101be41d3ae20d98eb1",
 "demo_stats_sha256":"6f7e9f4a2232a798e4e30ad26f5748e71aeeda7fa54cb6ea2d0a3ec7d290e814",
 "c48_dataset_sha256":"d416d86c09ba334fae449a131510b84fa1d111e665a77eabfb248f1c79a5bc61",
 "c48_observations_sha256":"399d93f31a8f26297145942387a233b9667049efc60ac1f46514a3f7ce77a638",
 "c59_completed_sha256":"4e67bb95b69ada2a854d3b2bf4ba434c6b3072c2bba11a91df2c30c6de5eeb99",
 "c59_sample_labels_sha256":"f2be6801cac2f1c5b680b30c5e089f47e2bf428f179ee13c1ae283e2d47a9d53",
}
ordered=("source_manifest_sha256","demo_manifest_sha256","demo_stats_sha256",
 "c48_dataset_sha256","c48_observations_sha256","c59_completed_sha256",
 "c59_sample_labels_sha256")
if (auth.get("format")!="h3wam-c67-c69-paired-rollout-authorization-v1" or
 auth.get("status")!="AUTHORIZED_C67_C69_FIXED_S20_PAIRED_680" or
 auth.get("permission")!="GO_C67_C69_1360_FRESH_PROCESSES_NO_INTERMEDIATE_STOP" or
 auth.get("release_signed") is not False or
 auth.get("historical_c60_data_sha256")!=expected_data):
    raise SystemExit("C67/C69 authorization contract mismatch")
if [sha(path.resolve()) for path in data_paths] != [expected_data[key] for key in ordered]:
    raise SystemExit("C67/C69 historical seven-data SHA fail-close")
if sha(h3.resolve())!="e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a":
    raise SystemExit("C67/C69 H3 checkpoint SHA mismatch")
if sha(manifest_path.resolve())!=auth.get("manifest_sha256") or len(jobs)!=1360:
    raise SystemExit("C67/C69 job manifest mismatch")
expected={(a,s,t,r) for r in range(33,50) for s in
 ("libero_spatial","libero_object","libero_goal","libero_10") for t in range(10)
 for a in ("c67_fact_joint","c69_action_only")}
actual={(j.get("arm"),j.get("suite"),j.get("tasks",[None])[0],j.get("trials",[None])[0])
 for j in jobs if j.get("episodes")==1}
if actual!=expected or len(actual)!=len(jobs): raise SystemExit("C67/C69 1360-grid mismatch")
selected=[j for j in jobs if j["pair_id"]%num_shards==shard_index]
if not selected or any(j["pair_id"]%num_shards!=shard_index for j in selected):
    raise SystemExit("C67/C69 shard selection failed")
for endpoint in auth["endpoints"].values():
    path=Path(endpoint["checkpoint"])
    if not path.is_file() or sha(path)!=endpoint["checkpoint_sha256"]:
        raise SystemExit("C67/C69 endpoint bytes mismatch")
if stat.S_IMODE(project.stat().st_mode)&0o222:
    raise SystemExit("C67/C69 source snapshot root is writable")
PY

lock="${root}/.launcher-shard$(printf '%02d' "${shard_index}").lock"
mkdir "${lock}" 2>/dev/null || { echo "another launcher owns ${lock}" >&2; exit 75; }
trap 'rmdir "${lock}" 2>/dev/null || true' EXIT
mkdir -p "${root}/logs"
cd "${project}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PROJECT_ROOT="${project}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export H3WAM_FASTWAM_SOURCE_ROOT="${project}/third_party/FastWAM/src/fastwam/models/wan22"
authorization_sha="$(sha256sum "${authorization}" | awk '{print $1}')"

run_worker() {
  local gpu="$1" job_id pair_id arm suite task trial assigned_gpu checkpoint output log
  while IFS=$'\t' read -r job_id pair_id arm suite task trial assigned_gpu checkpoint output; do
    [[ "${assigned_gpu}" == "${gpu}" ]] || continue
    [[ ! -e "${output}" ]] || {
      echo "refusing partial/reused C67/C69 output: ${output}" >&2; return 1;
    }
    mkdir -p "${output}"
    log="${root}/logs/shard$(printf '%02d' "${shard_index}")_job$(printf '%04d' "${job_id}")_${arm}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHON_BIN="${sim_python}" \
    SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site" \
    bash "${project}/scripts/h3wam/run_cloud_libero.sh" \
      "${sim_python}" "${project}/scripts/h3wam/rollout_libero.py" \
      --policy h3_fact_online_int8 --policy-python "${policy_python}" \
      --checkpoint "${checkpoint}" \
      --c67-c69-attribution-authorization "${authorization}" \
      --cache-root "${cache_root}" --h3-checkpoint "${h3_checkpoint}" \
      --h3-model "${h3_model}" --dreamwam-source-manifest "${source_manifest}" \
      --device cuda:0 --suite "${suite}" --task-ids "${task}" \
      --trial-indices "${trial}" --max-steps 400 --wait-steps 30 \
      --replan-steps 8 --action-horizon 32 --h3-feature-audio-horizon 32 \
      --target-latent-frames 12 --model-evaluations 10 --seed 42 \
      --normalized-action-pre-clamp --save-trajectories --output-dir "${output}" \
      >"${log}" 2>&1
  done < <("${sim_python}" - "${manifest}" "${shard_index}" "${num_shards}" <<'PY'
import json,sys
shard,total=map(int,sys.argv[2:4])
for line in open(sys.argv[1]):
 row=json.loads(line)
 if row["pair_id"]%total!=shard: continue
 gpu=(row["pair_id"]//total)%8
 print("\t".join(map(str,(row["job_id"],row["pair_id"],row["arm"],row["suite"],
  row["tasks"][0],row["trials"][0],gpu,row["checkpoint"],row["output"]))))
PY
)
}

pids=()
for gpu in 0 1 2 3 4 5 6 7; do run_worker "${gpu}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "one or more C67/C69 shard workers failed" >&2; exit 1; }

"${sim_python}" - "${authorization}" "${manifest}" "${marker}" \
  "${authorization_sha}" "${shard_index}" "${num_shards}" <<'PY'
import json,os,sys
from pathlib import Path
auth=json.load(open(sys.argv[1])); rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
out,auth_sha=Path(sys.argv[3]),sys.argv[4]; shard,total=map(int,sys.argv[5:7])
selected=[row for row in rows if row["pair_id"]%total==shard]
seen=set()
for row in selected:
 path=Path(row["output"])/"results.json"; data=json.load(open(path))
 found=[e for task in data.get("tasks",[]) for e in task.get("episodes",[])]
 key=(row["arm"],row["suite"],row["tasks"][0],row["trials"][0])
 if key in seen or len(found)!=1 or data.get("task_ids")!=row["tasks"] or data.get("trial_indices")!=row["trials"]:
  raise ValueError(f"invalid/duplicate C67/C69 shard job: {path}")
 if data.get("c67_c69_attribution_authorization_sha256")!=auth_sha:
  raise ValueError(f"C67/C69 result authorization mismatch: {path}")
 seen.add(key)
tmp=out.with_name(f".{out.name}.{os.getpid()}.partial")
tmp.write_text(json.dumps({
 "format":"h3wam-c67-c69-paired680-shard-complete-v1","status":"COMPLETE",
 "shard_index":shard,"num_shards":total,"jobs":len(selected),
 "pairs":len(selected)//2,"authorization_sha256":auth_sha,
 "manifest_sha256":auth["manifest_sha256"],
},indent=2)+"\n")
os.replace(tmp,out)
PY
