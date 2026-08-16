#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
policy_python="${POLICY_PYTHON:-${workspace}/runtime/h3-int8-native/bin/python}"
sim_python="${SIM_PYTHON:-${workspace}/runtime/conda-py311/bin/python}"
main_root="${C56B_MAIN_ROOT:-${workspace}/outputs/c56b-fact-online-v1/online-long10000-v1}"
c61_root="${C56B_C61_ROOT:-${workspace}/outputs/c56b-fact-online-v1/online-long10000-c61-matched-v1}"
eval_root="${C56B_PAIRED_EVAL_ROOT:-${workspace}/outputs/c56b-fact-online-v1/paired-final-eval-v1}"
main_ready="${main_root}/READY.json"
c61_ready="${c61_root}/READY.json"
paired_report="${eval_root}/balanced80/PAIRED_BALANCED80.json"
rollout_root="${eval_root}/fresh-execution-libero-trial33"
c58_results="${C58B_RESULTS:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-final-eval-v1/fresh-libero-trial33/RESULTS.json}"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
h3_model="${workspace}/models/MiniMax-H3"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
train_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
val_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_val.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"

for path in "${project}" "${policy_python}" "${sim_python}" "${h3_checkpoint}" \
  "${h3_model}" "${source_manifest}" "${train_manifest}" "${val_manifest}" \
  "${cache_root}/stats.pt"; do
  [[ -e "${path}" ]] || { echo "missing C56b paired-eval input: ${path}" >&2; exit 2; }
done
mkdir -p "${eval_root}/balanced80" "${rollout_root}"
lock="${eval_root}/.watcher.lock"
mkdir "${lock}" 2>/dev/null || { echo "another C56b paired evaluator owns ${lock}" >&2; exit 75; }
trap 'rmdir "${lock}" 2>/dev/null || true' EXIT

while [[ ! -s "${main_ready}" || ! -s "${c61_ready}" || ! -s "${c58_results}" ]]; do
  sleep 30
done

readarray -t endpoint_values < <("${policy_python}" - "${main_ready}" "${c61_ready}" <<'PY'
import hashlib, json, sys
from pathlib import Path
for expected, value in zip(("C60_MAIN", "C61_MATCHED"), sys.argv[1:]):
    path = Path(value).resolve()
    ready = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(ready.get("checkpoint", "")).resolve()
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    checks = (
        ready.get("status") == "PASS_C56B_ONLINE_LONG10000_STRICT_RESTORE",
        ready.get("permission") == "READY_FOR_PAIRED_HELDOUT",
        ready.get("arm") == expected,
        ready.get("completed_steps") == 10000,
        checkpoint.stat().st_size == ready.get("checkpoint_size_bytes"),
        digest.hexdigest() == ready.get("checkpoint_sha256"),
    )
    if not all(checks):
        raise SystemExit(f"invalid endpoint {expected}")
    print(checkpoint)
PY
)
main_checkpoint="${endpoint_values[0]}"
c61_checkpoint="${endpoint_values[1]}"

gpu_idle() {
  local index
  for index in 0 1 2 3 4 5 6 7; do
    [[ -z "$(nvidia-smi -i "${index}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')" ]] || return 1
  done
}
while ! gpu_idle; do sleep 30; done
sleep 30
while ! gpu_idle; do sleep 30; done

cd "${project}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export TMPDIR="${workspace}/tmp/c56b-paired-final-eval"
mkdir -p "${TMPDIR}"

if [[ -e "${paired_report}" ]]; then
  echo "refusing pre-existing paired heldout report: ${paired_report}" >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES=0 "${policy_python}" \
  scripts/h3wam/evaluate_c56b_fact_online_paired.py \
  --main-ready "${main_ready}" --c61-ready "${c61_ready}" \
  --h3-checkpoint "${h3_checkpoint}" --source-manifest "${source_manifest}" \
  --train-manifest "${train_manifest}" --val-manifest "${val_manifest}" \
  --cache-root "${cache_root}" --device cuda:0 --output "${paired_report}" \
  >"${eval_root}/balanced80/evaluator.log" 2>&1

"${policy_python}" - "${paired_report}" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("permission") != "GO_PAIRED_LIBERO":
    raise SystemExit("paired heldout did not authorize LIBERO")
PY

run_suite() {
  local arm="$1" suite="$2" gpu="$3" checkpoint
  checkpoint="${main_checkpoint}"
  [[ "${arm}" == "c61_matched" ]] && checkpoint="${c61_checkpoint}"
  local output="${rollout_root}/${arm}/${suite}"
  if [[ -e "${output}/results.json" ]]; then
    echo "refusing pre-existing C56b rollout result: ${output}/results.json" >&2
    return 2
  fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHON_BIN="${sim_python}" SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site" \
  bash "${project}/scripts/h3wam/run_cloud_libero.sh" \
    "${sim_python}" "${project}/scripts/h3wam/rollout_libero.py" \
    --policy h3_fact_online_int8 --policy-python "${policy_python}" \
    --checkpoint "${checkpoint}" --c56b-paired-ready "${paired_report}" \
    --cache-root "${cache_root}" --h3-checkpoint "${h3_checkpoint}" \
    --h3-model "${h3_model}" --dreamwam-source-manifest "${source_manifest}" \
    --device cuda:0 --suite "${suite}" --task-ids 0 1 2 3 4 5 6 7 8 9 \
    --trial-indices 33 --max-steps 400 --wait-steps 30 --replan-steps 8 \
    --action-horizon 32 --h3-feature-audio-horizon 32 --target-latent-frames 12 \
    --model-evaluations 10 --seed 42 --normalized-action-pre-clamp \
    --output-dir "${output}" >"${output}/launcher.log" 2>&1
}

pids=()
index=0
for arm in c60_main c61_matched; do
  for suite in libero_spatial libero_object libero_goal libero_10; do
    run_suite "${arm}" "${suite}" "${index}" &
    pids+=("$!")
    index=$((index + 1))
  done
done
for pid in "${pids[@]}"; do wait "${pid}"; done

"${sim_python}" scripts/h3wam/aggregate_c56b_fact_paired_libero.py \
  --root "${rollout_root}" --gate "${paired_report}" \
  --main-ready "${main_ready}" --c61-ready "${c61_ready}" \
  --c58-results "${c58_results}" \
  --output "${rollout_root}/RESULTS.json"
