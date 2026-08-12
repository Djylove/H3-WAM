#!/usr/bin/env bash
set -Eeuo pipefail

H3_ROOT=${H3_ROOT:-/mnt/h3-wam}
PROJECT=${H3_ROOT}/project
PYTHON=${H3_ROOT}/.venv/bin/python
MODEL=${H3_ROOT}/models/MiniMax-H3
CACHE=${H3_ROOT}/data/v2_full_cache
CANDIDATE=${H3_ROOT}/data/v4_multisuite_uniform_candidate
OUTPUTS=${H3_ROOT}/outputs/h3dotwam
LOGS=${H3_ROOT}/logs/pipeline
M0=m0v2_h32_gb128_s150
M2=m2_language_rank_full50_gb128_s5
M0_EVAL=${OUTPUTS}/${M0}_posttrain
M2_EVAL=${OUTPUTS}/${M2}_posttrain
COUNTERFACTUAL=${PROJECT}/experiments/h3dotwam/counterfactual_task0_val.jsonl

mkdir -p "${OUTPUTS}" "${LOGS}" "${M0_EVAL}" "${M2_EVAL}"
export HOME=${H3_ROOT}
export XDG_CACHE_HOME=${H3_ROOT}/cache
export HF_HOME=${H3_ROOT}/cache/huggingface
export TMPDIR=${H3_ROOT}/tmp
export PYTHONPATH=${PROJECT}/src:${PROJECT}

bootstrap_pid=$(<"${LOGS}/bootstrap_cloud_m0.pid")
echo "WAIT bootstrap_m0 pid=${bootstrap_pid}"
while kill -0 "${bootstrap_pid}" 2>/dev/null; do
  sleep 30
done
test -s "${OUTPUTS}/${M0}.json"
test -s "${OUTPUTS}/${M0}.pt"

stage_for_step() {
  local step=$1
  if [[ "${step}" -eq 150 ]]; then
    echo "${OUTPUTS}/${M0}.pt"
  else
    printf '%s/%s_step%06d.pt\n' "${OUTPUTS}" "${M0}" "${step}"
  fi
}

for step in 25 50 75 100 125 150; do
  stage=$(stage_for_step "${step}")
  test -s "${stage}"
  report=$(printf '%s/val40_step%06d.json' "${M0_EVAL}" "${step}")
  if [[ ! -s "${report}" ]]; then
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=8 \
      "${PROJECT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
      --model "${MODEL}" --data-root "${CACHE}" \
      --manifest "${CANDIDATE}/manifest_val_stratified40.jsonl" \
      --output "${report}" --load-stage "${stage}" \
      --eval-only --steps 5 --sample-steps 10 --action-horizon 32 \
      --require-text-only-context --log-every 1 \
      > "${M0_EVAL}/val40_step$(printf '%06d' "${step}").log" 2>&1
  fi
done

best_stage=$("${PYTHON}" - "${M0_EVAL}" "${OUTPUTS}" "${M0}" <<'PY'
import json
import pathlib
import sys

root, outputs, run = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
scores = []
for path in sorted(root.glob("val40_step*.json")):
    step = int(path.stem.rsplit("step", 1)[1])
    score = float(json.loads(path.read_text())["mean_action_loss"])
    scores.append((score, step))
if len(scores) != 6:
    raise SystemExit(f"expected six M0 evaluations, found {len(scores)}")
best_score, best_step = min(scores)
if best_score > 0.25:
    raise SystemExit(f"M0 reproduction gate failed: val40={best_score:.6f} > 0.25")
stage = outputs / (run + ".pt" if best_step == 150 else f"{run}_step{best_step:06d}.pt")
payload = {
    "best_step": best_step,
    "mean_action_mse": best_score,
    "checkpoint": str(stage),
    "all": [{"step": step, "mean_action_mse": score} for score, step in sorted(scores, key=lambda x: x[1])],
}
(root / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
print(stage)
PY
)
echo "M0_GATE_OK stage=${best_stage}"

run_counterfactual() {
  local prefix=$1
  local load_mode=$2
  local load_path=$3
  local override=$4
  local report=${prefix}.json
  local extra=()
  if [[ -n "${override}" ]]; then
    extra=(--context-override-id "${override}")
  fi
  if [[ ! -s "${report}" ]]; then
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=8 \
      "${PROJECT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
      --model "${MODEL}" --data-root "${CACHE}" \
      --manifest "${COUNTERFACTUAL}" --output "${report}" \
      "${load_mode}" "${load_path}" \
      --eval-only --steps 1 --sample-steps 10 --action-horizon 32 \
      --require-text-only-context --record-sampled-actions "${extra[@]}" \
      > "${prefix}.log" 2>&1
  fi
}

run_counterfactual "${M0_EVAL}/counterfactual_correct" --load-stage "${best_stage}" ""
run_counterfactual "${M0_EVAL}/counterfactual_stove" --load-stage "${best_stage}" task_c0d1b2f3264d13ce

M2_JOINT=${OUTPUTS}/${M2}_joint
if [[ ! -s "${OUTPUTS}/${M2}.json" || ! -s "${M2_JOINT}/joint_stage.json" ]]; then
  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "${PROJECT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
    --model "${MODEL}" --data-root "${CACHE}" \
    --manifest "${CANDIDATE}/manifest_train_uniform.jsonl" \
    --output "${OUTPUTS}/${M2}.json" --load-stage "${best_stage}" \
    --save-joint-stage "${M2_JOINT}" \
    --steps 5 --gradient-accumulation-steps 16 \
    --action-horizon 32 --learning-rate 1e-5 --h3-learning-rate 1e-6 \
    --last-h3-blocks 50 --video-loss-weight 0.25 \
    --language-ranking-weight 0.5 --language-ranking-margin 0.05 \
    --language-ranking-every 1 --require-text-only-context --log-every 1 \
    > "${LOGS}/${M2}.log" 2>&1
fi

if [[ ! -s "${M2_EVAL}/val40.json" ]]; then
  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "${PROJECT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
    --model "${MODEL}" --data-root "${CACHE}" \
    --manifest "${CANDIDATE}/manifest_val_stratified40.jsonl" \
    --output "${M2_EVAL}/val40.json" --load-joint-stage "${M2_JOINT}" \
    --eval-only --steps 5 --sample-steps 10 --action-horizon 32 \
    --require-text-only-context --log-every 1 \
    > "${M2_EVAL}/val40.log" 2>&1
fi
run_counterfactual "${M2_EVAL}/counterfactual_correct" --load-joint-stage "${M2_JOINT}" ""
run_counterfactual "${M2_EVAL}/counterfactual_stove" --load-joint-stage "${M2_JOINT}" task_c0d1b2f3264d13ce

"${PYTHON}" - "${M0_EVAL}" "${M2_EVAL}" <<'PY'
import json
import pathlib
import sys

import numpy as np

m0, m2 = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])

def cosine(root: pathlib.Path) -> float:
    correct = np.asarray(json.loads((root / "counterfactual_correct.json").read_text())["history"][0]["sampled_actions"])
    wrong = np.asarray(json.loads((root / "counterfactual_stove.json").read_text())["history"][0]["sampled_actions"])
    return float(np.dot(correct.ravel(), wrong.ravel()) / (np.linalg.norm(correct) * np.linalg.norm(wrong)))

selection = json.loads((m0 / "selection.json").read_text())
m0_cosine = cosine(m0)
m2_cosine = cosine(m2)
m2_mse = float(json.loads((m2 / "val40.json").read_text())["mean_action_loss"])
payload = {
    "m0_best_step": selection["best_step"],
    "m0_val40_mse": selection["mean_action_mse"],
    "m0_counterfactual_cosine": m0_cosine,
    "m2_val40_mse": m2_mse,
    "m2_counterfactual_cosine": m2_cosine,
    "counterfactual_cosine_change": m2_cosine - m0_cosine,
    "m2_val_within_10pct": m2_mse <= selection["mean_action_mse"] * 1.10,
    "language_sensitivity_improved": m2_cosine < m0_cosine,
}
(m2 / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

echo "M2_READY analysis=${M2_EVAL}/analysis.json"
