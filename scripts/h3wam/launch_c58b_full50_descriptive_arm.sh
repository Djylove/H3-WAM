#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root="${C58_FULL50_ROOT:?C58_FULL50_ROOT is required}"
arm="${C58_FULL50_ARM:?C58_FULL50_ARM is required}"
case "${arm}" in candidate_c58b|control_d0) ;; *) echo "invalid arm: ${arm}" >&2; exit 2;; esac
prepared="${root}/PREPARED.json"
manifest="${root}/jobs.jsonl"
policy_python="${workspace}/runtime/h3-int8-native/bin/python"
sim_python="${workspace}/runtime/conda-py311/bin/python"
gate="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-final-eval-v1/balanced80/BALANCED80_READY.14fa645.json"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
h3_model="${workspace}/models/MiniMax-H3"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
dreamwam_root="${project}/third_party/DreamWAM/dreamwam"
resume="${C58_FULL50_RESUME:-0}"
resume_rebalance="${C58_FULL50_RESUME_REBALANCE:-0}"
max_attempts="${C58_FULL50_MAX_ATTEMPTS:-3}"

case "${resume}" in 0|1) ;; *) echo "C58_FULL50_RESUME must be 0 or 1" >&2; exit 2;; esac
case "${resume_rebalance}" in 0|1) ;; *) echo "C58_FULL50_RESUME_REBALANCE must be 0 or 1" >&2; exit 2;; esac
[[ "${resume}" == 1 || "${resume_rebalance}" == 0 ]] || {
  echo "C58_FULL50_RESUME_REBALANCE requires C58_FULL50_RESUME=1" >&2; exit 2;
}
[[ "${max_attempts}" =~ ^[1-9][0-9]*$ ]] || {
  echo "C58_FULL50_MAX_ATTEMPTS must be a positive integer" >&2; exit 2;
}

for path in "${prepared}" "${manifest}" "${policy_python}" "${sim_python}" \
  "${gate}" "${h3_checkpoint}" "${h3_model}" "${source_manifest}" \
  "${cache_root}/stats.pt" "${source_root}/action_dit.py" \
  "${dreamwam_root}/layers.py" "${dreamwam_root}/experts.py" \
  "${dreamwam_root}/mot.py"; do
  [[ -e "${path}" ]] || { echo "missing full50 input: ${path}" >&2; exit 2; }
done
while IFS=: read -r path expected; do
  actual="$(sha256sum "${path}" | cut -d' ' -f1)"
  [[ "${actual}" == "${expected}" ]] || {
    echo "pinned DreamWAM source hash mismatch: ${path}" >&2; exit 2;
  }
done <<EOF
${dreamwam_root}/layers.py:3cd38ad24eff05e748d9353af3f39200e93b16b6d07d22f153ccef0f36becd96
${dreamwam_root}/experts.py:9ba51dbb15b8df8e4ff01c5a08acf443a950c422544e4497d13e0e2658bd489c
${dreamwam_root}/mot.py:5467d135287a6e77074cb653fc3d72218490fcfa40ac486b61d5cc5975ab6c01
EOF
completed="${root}/${arm}.COMPLETED.json"
[[ ! -e "${completed}" ]] || { echo "${arm} already complete"; exit 0; }
lock="${root}/.${arm}.launcher.lock"
mkdir "${lock}" 2>/dev/null || { echo "another ${arm} launcher owns ${lock}" >&2; exit 75; }
work_manifest="${manifest}"
trap '[[ "${work_manifest}" == "${manifest}" ]] || rm -f "${work_manifest}"; rmdir "${lock}" 2>/dev/null || true' EXIT

"${sim_python}" - "${prepared}" "${arm}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); arm=sys.argv[2]
assert p["permission"] == "GO_DESCRIPTIVE_FULL50_SUPPLEMENT_NO_PROMOTION"
assert p["jobs"] == 2640 and p["episodes_per_arm"] == 1320
assert p["process_contract"] == "one fresh simulator and policy process per episode"
assert arm in p["checkpoints"]
PY

cd "${project}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
mkdir -p "${root}/logs/${arm}"

if [[ "${resume_rebalance}" == 1 ]]; then
  work_manifest="${root}/.${arm}.resume-work.$$.jsonl"
  "${sim_python}" - "${manifest}" "${arm}" "${work_manifest}" <<'PY'
import json, os, sys
from pathlib import Path

rows=[]
for line in open(sys.argv[1]):
    row=json.loads(line)
    if row["arm"] != sys.argv[2]:
        continue
    if (Path(row["output"])/"results.json").is_file():
        continue
    rows.append(row)
for index,row in enumerate(rows):
    row=dict(row); row["gpu"]=index % 8
    rows[index]=row
tmp=Path(sys.argv[3]).with_name(f".{Path(sys.argv[3]).name}.{os.getpid()}.partial")
tmp.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in rows))
os.replace(tmp, sys.argv[3])
print(f"resume_rebalanced_jobs={len(rows)}", flush=True)
PY
fi

run_worker() {
  local gpu="$1" row ordinal assigned_gpu suite task trial policy checkpoint output
  local log attempt retry_quarantine
  while IFS=$'\t' read -r ordinal assigned_gpu suite task trial policy checkpoint output; do
    [[ "${assigned_gpu}" == "${gpu}" ]] || continue
    if [[ "${resume}" == 1 && -s "${output}/results.json" ]]; then
      continue
    fi
    [[ ! -e "${output}" ]] || {
      echo "refusing partial/reused full50 output: ${output}" >&2; return 1;
    }
    for ((attempt=1; attempt<=max_attempts; attempt++)); do
      mkdir -p "${output}"
      log="${root}/logs/${arm}/job$(printf '%04d' "${ordinal}")_${suite}_task${task}_trial${trial}_attempt${attempt}.log"
      extra=()
      [[ "${arm}" != candidate_c58b ]] || extra+=(--c58b-balanced80-ready "${gate}")
      if CUDA_VISIBLE_DEVICES="${gpu}" PYTHON_BIN="${sim_python}" \
        SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site" \
        bash "${project}/scripts/h3wam/run_cloud_libero.sh" \
          "${sim_python}" "${project}/scripts/h3wam/rollout_libero.py" \
          --policy "${policy}" --policy-python "${policy_python}" \
          --checkpoint "${checkpoint}" "${extra[@]}" --cache-root "${cache_root}" \
          --h3-checkpoint "${h3_checkpoint}" --h3-model "${h3_model}" \
          --dreamwam-source-manifest "${source_manifest}" --device cuda:0 \
          --suite "${suite}" --task-ids "${task}" --trial-indices "${trial}" \
          --max-steps 400 --wait-steps 30 --replan-steps 8 --action-horizon 32 \
          --h3-feature-audio-horizon 32 --target-latent-frames 12 \
          --model-evaluations 10 --seed 42 --normalized-action-pre-clamp \
          --save-trajectories --output-dir "${output}" >"${log}" 2>&1; then
        break
      fi
      if (( attempt == max_attempts )); then
        echo "full50 job failed after ${max_attempts} attempts: ${output}" >&2
        return 1
      fi
      retry_quarantine="${root}/quarantine/${arm}/job$(printf '%04d' "${ordinal}")_attempt${attempt}_pid$$"
      mkdir -p "$(dirname "${retry_quarantine}")"
      mv "${output}" "${retry_quarantine}"
      sleep 2
    done
  done < <("${sim_python}" - "${work_manifest}" "${arm}" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    row=json.loads(line)
    if row["arm"] != sys.argv[2]: continue
    print("\t".join(map(str, (
        row["arm_ordinal"], row["gpu"], row["suite"], row["task"], row["trial"],
        row["policy"], row["checkpoint"], row["output"],
    ))))
PY
)
}

pids=()
for gpu in 0 1 2 3 4 5 6 7; do run_worker "${gpu}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "one or more ${arm} workers failed" >&2; exit 1; }

"${sim_python}" - "${prepared}" "${manifest}" "${arm}" "${completed}" <<'PY'
import json,os,sys
from pathlib import Path
prepared=json.load(open(sys.argv[1])); arm=sys.argv[3]
rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip() and json.loads(x)["arm"]==arm]
if len(rows)!=1320: raise ValueError("full50 per-arm manifest count mismatch")
for row in rows:
    payload=json.load(open(Path(row["output"])/"results.json"))
    episodes=[e for task in payload.get("tasks",[]) for e in task.get("episodes",[])]
    if (len(episodes)!=1 or payload.get("task_ids")!=[row["task"]]
            or payload.get("trial_indices")!=[row["trial"]]):
        raise ValueError(f"invalid full50 episode: {row['output']}")
out=Path(sys.argv[4]);tmp=out.with_name(f".{out.name}.{os.getpid()}.partial")
tmp.write_text(json.dumps({"format":"h3wam-c58b-full50-arm-complete-v1","status":"COMPLETE","arm":arm,"episodes":1320,"manifest_sha256":prepared["manifest_sha256"]},indent=2)+"\n")
os.replace(tmp,out)
PY
