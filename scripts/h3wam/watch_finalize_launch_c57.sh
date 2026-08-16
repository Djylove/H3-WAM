#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${C57_EVAL_PYTHON:-${workspace}/runtime/h3-int8-native/bin/python}"
root="${C57_LONG_ROOT:-${workspace}/outputs/c57-lingbot-persistent-kv/long5000}"
checkpoint="${root}/checkpoints/c57_step05000.pt"
train_report="${root}/report.json"
heldout_report="${workspace}/outputs/c57-lingbot-persistent-kv/heldout_eval/step05000_paired.json"
heldout_plan="${C57_EVAL_PLAN:-${workspace}/data/c57-lingbot-replan8-v1/heldout_eval_plan.json}"
decision="${root}/FINAL.json"
watch_log="${root}/final_watcher.log"

mkdir -p "${root}"
exec >>"${watch_log}" 2>&1
echo "[$(date --iso-8601=seconds)] C57 final watcher started"
while [[ ! -s "${checkpoint}" || ! -s "${train_report}" || ! -s "${heldout_report}" ]]; do sleep 30; done

if [[ ! -s "${decision}" ]]; then
  "${python_bin}" "${project}/scripts/h3wam/finalize_c57_lingbot_long5000.py" \
    --checkpoint "${checkpoint}" --train-report "${train_report}" \
    --heldout-report "${heldout_report}" --plan "${heldout_plan}" --output "${decision}"
fi

permission=$("${python_bin}" - "${decision}" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("permission", ""))
PY
)
if [[ "${permission}" != "GO_FRESH_LIBERO_CANARY" ]]; then
  echo "[$(date --iso-8601=seconds)] C57 final heldout is NO_GO; no LIBERO canary launched"
  exit 0
fi

echo "[$(date --iso-8601=seconds)] C57 final heldout passed; entering guarded canary launcher"
exec bash "${project}/scripts/h3wam/launch_c57_final_fresh_libero_canary.sh"
