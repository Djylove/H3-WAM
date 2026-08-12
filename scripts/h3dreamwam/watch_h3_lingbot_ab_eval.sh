#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
VARIANT="${VARIANT:?set VARIANT=baseline or VARIANT=bidirectional}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
case "${VARIANT}" in
  baseline) NAME="ab_s${TRAIN_STEPS}_output_only" ;;
  bidirectional) NAME="ab_s${TRAIN_STEPS}_output_plus_bidir_tail2" ;;
  *) echo "invalid VARIANT" >&2; exit 2 ;;
esac

STAGE="${H3_WORKSPACE}/outputs/h3-lingbot-port/${NAME}.pt"
until [[ -s "${STAGE}" ]]; do sleep 20; done
until [[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) -eq 0 ]]; do sleep 10; done
exec env VARIANT="${VARIANT}" TRAIN_STEPS="${TRAIN_STEPS}" \
  bash "${H3_WORKSPACE}/project/scripts/h3dreamwam/eval_h3_lingbot_ab_canary.sh"
