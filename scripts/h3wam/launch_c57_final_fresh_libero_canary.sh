#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
sim_python="${SIM_PYTHON:-${workspace}/runtime/conda-py311/bin/python}"
policy_python="${POLICY_PYTHON:-${workspace}/runtime/h3-int8-native/bin/python}"
gpu="${C57_CANARY_GPU:-0}"
decision="${C57_FINAL_DECISION:-${workspace}/outputs/c57-lingbot-persistent-kv/long5000/FINAL.json}"
plan="${C57_CANARY_PLAN:-${project}/experiments/plans/c57_final_fresh_libero_canary_v1.json}"
c57_checkpoint="${C57_FINAL_CHECKPOINT:-${workspace}/outputs/c57-lingbot-persistent-kv/long5000/checkpoints/c57_step05000.pt}"
d0_checkpoint="${D0_CHECKPOINT:-${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt}"
output_root="${C57_CANARY_OUTPUT_ROOT:-${workspace}/outputs/c57-lingbot-persistent-kv/fresh-libero-canary-v1}"
c56_go_long="${C57_C56_GO_LONG:-${workspace}/outputs/c56b-fact-online-v1/optimizer-canary10-v1/GO_LONG.json}"
c56_parent_checkpoint="${C57_C56_PARENT_CHECKPOINT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt}"
c56_parent_ready="${C57_C56_PARENT_READY:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/READY.json}"
c56_final_checkpoint="${C57_C56_FINAL_CHECKPOINT:-${workspace}/outputs/c56b-fact-online-v1/online-long10000-v1/checkpoints/c56b_online_s10000.pt}"
c56_final_restore="${C57_C56_FINAL_RESTORE:-${workspace}/outputs/c56b-fact-online-v1/online-long10000-v1/restore/restore_s10000.json}"

[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "C57_CANARY_GPU must be a non-negative integer" >&2; exit 2; }
for path in "${sim_python}" "${policy_python}" "${decision}" "${plan}" \
  "${c57_checkpoint}" "${d0_checkpoint}" "${workspace}/data/v7_dense_h3_cache/stats.pt" \
  "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  "${workspace}/models/MiniMax-H3" "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"; do
  [[ -e "${path}" ]] || { echo "missing C57 canary input: ${path}" >&2; exit 2; }
done

"${policy_python}" - "${decision}" "${c57_checkpoint}" "${plan}" <<'PY'
import hashlib, json, sys
from pathlib import Path

decision_path, checkpoint, plan_path = map(lambda value: Path(value).resolve(), sys.argv[1:])
decision = json.loads(decision_path.read_text(encoding="utf-8"))
plan = json.loads(plan_path.read_text(encoding="utf-8"))
if decision.get("status") != "PASS_C57_FINAL_OFFLINE_GATE" or decision.get("permission") != "GO_FRESH_LIBERO_CANARY":
    raise SystemExit("C57 final offline gate did not authorize the fresh canary")
if Path(decision.get("checkpoint", "")).resolve() != checkpoint:
    raise SystemExit("C57 final decision/checkpoint identity mismatch")
digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    while chunk := stream.read(16 * 1024 * 1024):
        digest.update(chunk)
if digest.hexdigest() != decision.get("checkpoint_sha256"):
    raise SystemExit("C57 final checkpoint hash changed")
expected_contract = {
    "wait_steps": 0, "replan_steps": 8, "action_horizon": 32,
    "observe_every": 4, "h3_feature_audio_horizon": 32,
    "target_latent_frames": 12, "model_evaluations": 10,
    "max_steps": 400, "normalized_action_pre_clamp": True,
    "use_action_ensembler": False, "save_video": False,
    "save_trajectories": False,
}
if plan.get("schema") != "c57_final_fresh_libero_canary_v1" or plan.get("decision_checkpoint_step") != 5000:
    raise SystemExit("C57 fresh canary plan schema/step mismatch")
if plan.get("contract") != expected_contract:
    raise SystemExit("C57 fresh canary execution contract changed")
pairs = plan.get("pairs", [])
if len(pairs) != 4 or {row.get("suite") for row in pairs} != {"libero_goal", "libero_spatial", "libero_object", "libero_10"}:
    raise SystemExit("C57 canary must contain one pair per LIBERO suite")
for row in pairs:
    if row.get("task_id") != 0 or row.get("trial_index") != 49 or row.get("environment_seed") != 42:
        raise SystemExit("C57 fresh task/trial/environment contract changed")
PY

mkdir -p "${output_root}"
lock="${output_root}/.launcher.lock"
mkdir "${lock}" 2>/dev/null || { echo "another C57 canary launcher owns ${lock}" >&2; exit 75; }
trap 'rmdir "${lock}" 2>/dev/null || true' EXIT

c56_final_complete() {
  [[ -s "${c56_final_checkpoint}" && -s "${c56_final_restore}" ]] || return 1
  "${policy_python}" - "${c56_final_checkpoint}" "${c56_final_restore}" <<'PY'
import json, sys
from pathlib import Path
checkpoint = Path(sys.argv[1]).resolve()
report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
raise SystemExit(0 if report.get("status") == "PASS_C56B_STRICT_RESTORE" and report.get("restore_max_abs") == 0.0 and Path(report.get("checkpoint", "")).resolve() == checkpoint else 1)
PY
}

c56_reserved() {
  pgrep -f '([c]56|[C]56)' >/dev/null 2>&1 || \
    { [[ -s "${c56_go_long}" && -s "${c56_parent_checkpoint}" && -s "${c56_parent_ready}" ]] && ! c56_final_complete; }
}

gpu_compute_pids() {
  nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
}

wait_for_slot() {
  while c56_reserved || [[ -n "$(gpu_compute_pids)" ]]; do sleep 30; done
  sleep 30
  while c56_reserved || [[ -n "$(gpu_compute_pids)" ]]; do sleep 30; done
}

rollout_complete() {
  local result="$1" expected_checkpoint="$2" suite="$3" task_id="$4" trial="$5"
  [[ -s "${result}" ]] || return 1
  "${sim_python}" - "${result}" "${expected_checkpoint}" "${suite}" "${task_id}" "${trial}" <<'PY'
import json, sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = Path(sys.argv[2]).resolve()
valid = (
    Path(result.get("checkpoint", "")).resolve() == expected
    and result.get("suite") == sys.argv[3]
    and result.get("task_ids") == [int(sys.argv[4])]
    and result.get("trial_indices") == [int(sys.argv[5])]
    and result.get("episodes") == 1
    and result.get("successes") in (0, 1)
)
raise SystemExit(0 if valid else 1)
PY
}

export CUDA_VISIBLE_DEVICES="${gpu}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export PYTHON_BIN="${sim_python}"
export SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site"
export TMPDIR="${workspace}/tmp/c57-final-canary"
mkdir -p "${TMPDIR}"

mapfile -t pairs < <("${sim_python}" - "${plan}" <<'PY'
import json, sys
from pathlib import Path
for row in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["pairs"]:
    print("\t".join(str(row[key]) for key in ("suite", "task_id", "trial_index", "environment_seed", "policy_noise_seed_base")))
PY
)

run_one() {
  local arm="$1" suite="$2" task_id="$3" trial="$4" environment_seed="$5" policy_seed="$6"
  local checkpoint runner output
  if [[ "${arm}" == "c57" ]]; then
    checkpoint="${c57_checkpoint}"
    runner="${project}/scripts/h3wam/rollout_c57_lingbot_libero.py"
  else
    checkpoint="${d0_checkpoint}"
    runner="${project}/scripts/h3wam/rollout_libero.py"
  fi
  output="${output_root}/${suite}_task${task_id}_trial${trial}/${arm}"
  rollout_complete "${output}/results.json" "${checkpoint}" "${suite}" "${task_id}" "${trial}" && return
  wait_for_slot
  bash "${project}/scripts/h3wam/run_cloud_libero.sh" \
    "${sim_python}" "${runner}" \
    --policy h3_dreamwam_kv_int8 \
    --policy-python "${policy_python}" \
    --checkpoint "${checkpoint}" \
    --cache-root "${workspace}/data/v7_dense_h3_cache" \
    --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
    --h3-model "${workspace}/models/MiniMax-H3" \
    --dreamwam-source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
    --device cuda:0 --suite "${suite}" --task-ids "${task_id}" --trial-indices "${trial}" \
    --max-steps 400 --wait-steps 0 --replan-steps 8 --action-horizon 32 \
    --h3-feature-audio-horizon 32 --target-latent-frames 12 --model-evaluations 10 \
    --seed 42 --environment-seed "${environment_seed}" --policy-noise-seed-base "${policy_seed}" \
    --normalized-action-pre-clamp --output-dir "${output}"
}

for pair in "${pairs[@]}"; do
  IFS=$'\t' read -r suite task_id trial environment_seed policy_seed <<<"${pair}"
  run_one d0 "${suite}" "${task_id}" "${trial}" "${environment_seed}" "${policy_seed}"
  run_one c57 "${suite}" "${task_id}" "${trial}" "${environment_seed}" "${policy_seed}"
done

"${sim_python}" "${project}/scripts/h3wam/aggregate_c57_final_libero_canary.py" \
  --plan "${plan}" --root "${output_root}" --c57-checkpoint "${c57_checkpoint}" \
  --d0-checkpoint "${d0_checkpoint}" --output "${output_root}/RESULTS.json"
