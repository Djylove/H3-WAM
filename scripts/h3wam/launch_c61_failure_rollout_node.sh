#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
root="${C61_OUTPUT_ROOT:-${workspace}/eval/c61-failure-rollout-expansion-v1}"
checkpoint="${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"
python_bin="${workspace}/runtime/conda-py311/bin/python"
policy_python="${workspace}/runtime/h3-int8-native/bin/python"
node="${C61_NODE:-0}"
num_nodes="${C61_NUM_NODES:-1}"
[[ "${node}" =~ ^[0-9]+$ && "${num_nodes}" =~ ^[1-9][0-9]*$ ]]
(( node < num_nodes ))
test -f "${root}/FROZEN.json"
test -f "${checkpoint}"
marker="${root}/node${node}-of-${num_nodes}.COMPLETED"
test ! -e "${marker}"

mapfile -t jobs < <(
  "${python_bin}" - "${root}/jobs.jsonl" "${node}" "${num_nodes}" <<'PY'
import json,sys
node,total=map(int,sys.argv[2:])
for line in open(sys.argv[1]):
 row=json.loads(line)
 if row["ordinal"]%total==node:
  print("\t".join(str(row[k]) for k in ("ordinal","group_id","candidate","suite","task","trial","trajectory","index","start_step","first_policy_noise_seed","continuation_policy_noise_seed_base")))
PY
)
(( ${#jobs[@]} > 0 ))

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
    test ! -e "${out}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64" \
    PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}" \
    PYTHON_BIN="${python_bin}" SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site" \
    bash "${project}/scripts/h3wam/run_cloud_libero.sh" \
      "${python_bin}" "${project}/scripts/h3wam/rollout_libero.py" \
      --policy h3_dreamwam_kv_int8 --policy-python "${policy_python}" \
      --checkpoint "${checkpoint}" --cache-root "${workspace}/data/v7_dense_h3_cache" \
      --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
      --h3-model "${workspace}/models/MiniMax-H3" \
      --dreamwam-source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
      --device cuda:0 --suite "${suite}" --task-ids "${task}" --trial-indices "${trial}" \
      --max-steps 400 --wait-steps 0 --replan-steps 8 --action-horizon 32 \
      --h3-feature-audio-horizon 32 --target-latent-frames 12 --model-evaluations 10 \
      --seed 42 --normalized-action-pre-clamp \
      --start-trajectory "${trajectory}" --start-index "${index}" --environment-seed 42 \
      --first-policy-noise-seed "${first_seed}" \
      --continuation-policy-noise-seed-base "${continuation}" --first-replan-steps 32 \
      --output-dir "${out}" --save-trajectories >"${log}" 2>&1
  done
}

started="$(date +%s)"
pids=()
for gpu in {0..7}; do worker "${gpu}" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 ))
printf '{"node":%s,"num_nodes":%s,"jobs":%s,"duration_seconds":%s}\n' \
  "${node}" "${num_nodes}" "${#jobs[@]}" "$(( $(date +%s)-started ))" >"${marker}"
