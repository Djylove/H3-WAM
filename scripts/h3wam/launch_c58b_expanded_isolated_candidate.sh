#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root="${C58B_EXPANDED_ROOT:?C58B_EXPANDED_ROOT is required}"
prepared="${root}/PREPARED.json"
manifest="${root}/jobs.jsonl"
checkpoint="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt"
gate="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-final-eval-v1/balanced80/BALANCED80_READY.14fa645.json"
d0_ready="${workspace}/eval/c58b-expanded-d0-control-v1/READY.json"
policy_python="${workspace}/runtime/h3-int8-native/bin/python"
sim_python="${workspace}/runtime/conda-py311/bin/python"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
h3_model="${workspace}/models/MiniMax-H3"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"

for path in "${prepared}" "${manifest}" "${checkpoint}" "${gate}" "${d0_ready}" \
  "${policy_python}" "${sim_python}" "${h3_checkpoint}" "${h3_model}" \
  "${source_manifest}" "${cache_root}/stats.pt" "${source_root}/action_dit.py"; do
  [[ -e "${path}" ]] || { echo "missing isolated C58b input: ${path}" >&2; exit 2; }
done
[[ ! -e "${root}/COMPLETED.json" ]] || { echo "isolated candidate rollout already complete"; exit 0; }
lock="${root}/.launcher.lock"
mkdir "${lock}" 2>/dev/null || { echo "another isolated launcher owns ${lock}" >&2; exit 75; }
trap 'rmdir "${lock}" 2>/dev/null || true' EXIT

"${sim_python}" - "${prepared}" "${d0_ready}" <<'PY'
import json, sys
p=json.load(open(sys.argv[1])); d=json.load(open(sys.argv[2]))
assert p["permission"] == "GO_8GPU_640_FRESH_PROCESSES_NO_INTERMEDIATE_STOP"
assert p["jobs"] == 640 and p["candidate_episodes"] == 640
assert p["one_episode_per_process"] is True
assert d["permission"] == "GO_C58B_CANDIDATE_ONLY_TRIALS34_49"
PY

cd "${project}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
mkdir -p "${root}/logs"

run_worker() {
  local gpu="$1" row job_id suite task trial assigned_gpu output log
  while IFS=$'\t' read -r job_id suite task trial assigned_gpu output; do
    [[ "${assigned_gpu}" == "${gpu}" ]] || continue
    [[ ! -e "${output}" ]] || {
      echo "refusing partial/reused isolated candidate output: ${output}" >&2
      return 1
    }
    mkdir -p "${output}"
    log="${root}/logs/job$(printf '%03d' "${job_id}")_${suite}_task${task}_trial${trial}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHON_BIN="${sim_python}" \
    SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site" \
    bash "${project}/scripts/h3wam/run_cloud_libero.sh" \
      "${sim_python}" "${project}/scripts/h3wam/rollout_libero.py" \
      --policy h3_fastwam_online_int8 --policy-python "${policy_python}" \
      --checkpoint "${checkpoint}" --c58b-balanced80-ready "${gate}" \
      --cache-root "${cache_root}" --h3-checkpoint "${h3_checkpoint}" \
      --h3-model "${h3_model}" --dreamwam-source-manifest "${source_manifest}" \
      --device cuda:0 --suite "${suite}" --task-ids "${task}" \
      --trial-indices "${trial}" --max-steps 400 --wait-steps 30 \
      --replan-steps 8 --action-horizon 32 --h3-feature-audio-horizon 32 \
      --target-latent-frames 12 --model-evaluations 10 --seed 42 \
      --normalized-action-pre-clamp --save-trajectories --output-dir "${output}" \
      >"${log}" 2>&1
  done < <("${sim_python}" - "${manifest}" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    row=json.loads(line)
    print("\t".join(map(str, (
        row["job_id"], row["suite"], row["tasks"][0], row["trials"][0],
        row["gpu"], row["output"],
    ))))
PY
)
}

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  run_worker "${gpu}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
(( status == 0 )) || { echo "one or more isolated C58b workers failed" >&2; exit 1; }

"${sim_python}" - "${prepared}" "${manifest}" "${root}/COMPLETED.json" <<'PY'
import json, os, sys
from pathlib import Path
prepared=json.load(open(sys.argv[1])); rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
if len(rows) != 640: raise ValueError("isolated job count mismatch")
episodes=0
for row in rows:
    path=Path(row["output"])/"results.json"; data=json.load(open(path))
    found=[e for task in data.get("tasks",[]) for e in task.get("episodes",[])]
    if (len(found) != 1 or data.get("task_ids") != row["tasks"]
            or data.get("trial_indices") != row["trials"]):
        raise ValueError(f"invalid isolated C58b job: {path}")
    episodes += 1
if episodes != 640: raise ValueError("isolated candidate episode count mismatch")
out=Path(sys.argv[3]); tmp=out.with_name(f".{out.name}.{os.getpid()}.partial")
tmp.write_text(json.dumps({"format":"h3wam-c58b-expanded-isolated-candidate-complete-v1","status":"COMPLETE","episodes":episodes,"manifest_sha256":prepared["manifest_sha256"]},indent=2)+"\n")
os.replace(tmp,out)
PY
