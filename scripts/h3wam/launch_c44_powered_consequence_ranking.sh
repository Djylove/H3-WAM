#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
c34_root="${workspace}/eval/c34-combined-consequence-ranking-v1"
c43_root="${workspace}/eval/c43-powered-causal-ranking-v1"
root="${C44_ROOT:-${workspace}/eval/c44-powered-consequence-ranking-v1}"
output_root="${workspace}/outputs/c44-powered-consequence-ranking-v1"
python="${workspace}/runtime/conda-py311/bin/python"
feature_python="${workspace}/runtime/h3-int8-native/bin/python"

test -s "${c43_root}/COMPLETED"
"${python}" - "${c43_root}/COMPLETED" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["training_permission"] == "GO_C44_POWERED_CONSEQUENCE_VALUE_RANKING", p["training_permission"]
PY
mkdir -p "${root}" "${output_root}"
test ! -e "${output_root}/COMPLETED"

cd "${project}"
if [[ ! -s "${root}/dataset.pt" ]]; then
  PYTHONPATH=src:scripts/h3wam:. "${python}" \
    scripts/h3wam/prepare_c44_powered_consequence_ranking_dataset.py \
    --c34-dataset "${c34_root}/dataset.pt" \
    --c43-root "${c43_root}" \
    --output "${root}/dataset.pt"
fi

if [[ ! -s "${root}/h3_features.pt" ]]; then
  env CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=src:scripts/h3wam:. \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    "${feature_python}" scripts/h3wam/precompute_c26_causal_h3_features.py \
    --dataset "${root}/dataset.pt" \
    --cache-root "${workspace}/data/v7_dense_h3_cache" \
    --source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
    --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
    --h3-model "${workspace}/models/MiniMax-H3" \
    --output "${root}/h3_features.pt" \
    --device cuda:0 \
    --progress-every 32
fi

if [[ -e "${output_root}/report.json" || -e "${output_root}/ranker.pt" ]]; then
  test -s "${output_root}/report.json"
  test -s "${output_root}/ranker.pt"
else
  env CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=src:scripts/h3wam:. \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    "${python}" scripts/h3wam/train_c44_powered_consequence_value_ranker.py \
    --dataset "${root}/dataset.pt" \
    --features "${root}/h3_features.pt" \
    --consequence-checkpoints \
      "${workspace}/outputs/c38-temporal-paired-null-replication-v1/temporal_seed161803/checkpoints/temporal_seed161803_step10000.pt" \
      "${workspace}/outputs/c38-temporal-paired-null-replication-v1/temporal_seed271828/checkpoints/temporal_seed271828_step10000.pt" \
      "${workspace}/outputs/c38-temporal-paired-null-replication-v1/temporal_seed8675309/checkpoints/temporal_seed8675309_step10000.pt" \
      "${workspace}/outputs/c38-temporal-paired-null-replication-v1/temporal_seed20260815/checkpoints/temporal_seed20260815_step10000.pt" \
    --output "${output_root}/report.json" \
    --checkpoint "${output_root}/ranker.pt" \
    --device cuda:0
fi

"${python}" - "${root}" "${output_root}" <<'PY'
import hashlib,json,os,pathlib,sys
root,out=map(pathlib.Path,sys.argv[1:])
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b""): h.update(chunk)
    return h.hexdigest()
report=json.load((out/"report.json").open())
payload={
    "format":"h3wam-c44-powered-consequence-ranking-completed-v1",
    "status":report["status"], "permission":report["permission"],
    "dataset_sha256":sha(root/"dataset.pt"),
    "features_sha256":sha(root/"h3_features.pt"),
    "report_sha256":sha(out/"report.json"),
    "checkpoint_sha256":sha(out/"ranker.pt"),
}
target=out/"COMPLETED"; tmp=out/f".{target.name}.{os.getpid()}.partial"
tmp.write_text(json.dumps(payload,indent=2)+"\n"); os.replace(tmp,target)
print(json.dumps(payload,indent=2))
PY
