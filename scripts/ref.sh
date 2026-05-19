#!/usr/bin/env bash
torchrun \
--nnodes=2 \
--nproc_per_node=8 \
--node_rank=$RANK \
--rdzv_backend=c10d \
--rdzv_endpoint=$MASTER_ADDR:29500 \
/workspace/mnt/data/examples-main/distributed/ddp-tutorial-series/multinode.py \
10 2 --batch_size 32
