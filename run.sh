#!/usr/bin/env bash
python train.py \
--env "MontezumaAnnotated" \
--model-type mlp \
--gpu \
--seed 5 \
--lr 0.0005 \
--batchsize 32 \
--replay-buffer-size 10000000 \
--warmup-period 1000000  \
--episodes 3000 \
--reward-clip 10 \
--model-path ./saved_models
