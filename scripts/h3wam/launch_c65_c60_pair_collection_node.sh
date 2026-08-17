#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT must be an immutable C65 snapshot}"
root="${C65_OUTPUT_ROOT:-${workspace}/eval/c65-c60-deployment-pair-collection-v1}"
suites_csv="${C65_SUITES:?C65_SUITES must contain comma-separated suites for this node}"
node_tag="${C65_NODE_TAG:?C65_NODE_TAG is required}"
prepared="${root}/PREPARED.json"
manifest="${root}/jobs.jsonl"
checkpoint="${workspace}/outputs/c56b-fact-online-v1/online-long10000-v1/checkpoints/c56b_online_s10000.pt"
gate="${workspace}/outputs/c56b-fact-online-v1/paired-final-eval-v2/balanced80/PAIRED_BALANCED80.json"
policy_python="${workspace}/runtime/h3-int8-native/bin/python"
sim_python="${workspace}/runtime/conda-py311/bin/python"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
h3_model="${workspace}/models/MiniMax-H3"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
source_root="${workspace}/upstream-readonly/FastWAM-45d8e145/wan22"
marker="${root}/node-${node_tag}.COMPLETED"

case ",${suites_csv}," in
  *,libero_spatial,*|*,libero_object,*|*,libero_goal,*|*,libero_10,*) ;;
  *) echo "C65_SUITES contains no valid suite" >&2; exit 2 ;;
esac
[[ "${node_tag}" =~ ^[a-z0-9-]+$ ]]
for path in "${prepared}" "${manifest}" "${checkpoint}" "${gate}" \
  "${policy_python}" "${sim_python}" "${h3_checkpoint}" "${h3_model}" \
  "${source_manifest}" "${cache_root}/stats.pt"; do
  [[ -e "${path}" ]] || { echo "missing C65 input: ${path}" >&2; exit 2; }
done
[[ "$(stat -c '%A' "${project}/scripts/h3wam/rollout_libero.py")" != *w* ]] || {
  echo "PROJECT_ROOT is not read-only" >&2; exit 2;
}
[[ ! -e "${marker}" ]] || { echo "C65 node already complete: ${marker}"; exit 0; }

mapfile -t jobs < <(
  "${sim_python}" - "${prepared}" "${manifest}" "${suites_csv}" <<'PY'
import hashlib,json,sys
p=json.load(open(sys.argv[1])); suites=set(sys.argv[3].split(','))
rows=[json.loads(line) for line in open(sys.argv[2]) if line.strip()]
assert p["status"] == "PASS_C65_COLLECTION_FROZEN_NOT_EXECUTED"
assert p["effect_status"] == "NOT_EVALUATED"
assert p["jobs"] == 3072 and p["groups"] == 384 and p["sources"] == 96
assert hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest() == p["jobs_sha256"]
allowed={"libero_spatial","libero_object","libero_goal","libero_10"}
assert suites and suites <= allowed
for row in rows:
 if row["suite"] in suites:
  print("\t".join(str(row[k]) for k in (
   "ordinal","group_id","candidate","suite","task","trial","trajectory",
   "index","start_step","first_policy_noise_seed",
   "continuation_policy_noise_seed_base",
  )))
PY
)
(( ${#jobs[@]} > 0 ))

cd "${project}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
mkdir -p "${root}/logs" "${root}/runs"

worker() {
  local gpu="$1" i row ordinal group candidate suite task trial trajectory index start_step first_seed continuation out log
  for ((i=gpu;i<${#jobs[@]};i+=8)); do
    row="${jobs[$i]}"
    IFS=$'\t' read -r ordinal group candidate suite task trial trajectory index start_step first_seed continuation <<<"${row}"
    out="${root}/runs/${ordinal}_g${group}_c${candidate}_${suite#libero_}_task${task}_trial${trial}"
    log="${root}/logs/${ordinal}_g${group}_c${candidate}.log"
    if [[ -s "${out}/results.json" ]] && compgen -G "${out}/*trajectory.npz" >/dev/null; then
      continue
    fi
    [[ ! -e "${out}" ]] || { echo "refusing partial C65 output: ${out}" >&2; return 1; }
    mkdir -p "${out}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHON_BIN="${sim_python}" \
    SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site" \
    bash "${project}/scripts/h3wam/run_cloud_libero.sh" \
      "${sim_python}" "${project}/scripts/h3wam/rollout_libero.py" \
      --policy h3_fact_online_int8 --policy-python "${policy_python}" \
      --checkpoint "${checkpoint}" --c56b-paired-ready "${gate}" \
      --cache-root "${cache_root}" --h3-checkpoint "${h3_checkpoint}" \
      --h3-model "${h3_model}" --dreamwam-source-manifest "${source_manifest}" \
      --device cuda:0 --suite "${suite}" --task-ids "${task}" \
      --trial-indices "${trial}" --max-steps 400 --wait-steps 0 \
      --replan-steps 8 --first-replan-steps 8 --action-horizon 32 \
      --h3-feature-audio-horizon 32 --target-latent-frames 12 \
      --model-evaluations 10 --seed 42 --environment-seed 42 \
      --normalized-action-pre-clamp --start-trajectory "${trajectory}" \
      --start-index "${index}" --first-policy-noise-seed "${first_seed}" \
      --continuation-policy-noise-seed-base "${continuation}" \
      --save-trajectories --output-dir "${out}" >"${log}" 2>&1
  done
}

started="$(date +%s)"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do worker "${gpu}" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 ))
printf '{"node_tag":"%s","suites":"%s","jobs":%s,"duration_seconds":%s}\n' \
  "${node_tag}" "${suites_csv}" "${#jobs[@]}" "$(( $(date +%s)-started ))" >"${marker}"
