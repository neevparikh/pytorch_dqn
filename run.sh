#!/usr/bin/env bash
python train.py \
--env "CartPole-v1" \
--model-type mlp \
--gpu \
--seed 5 \
--lr 0.0005 \
--batchsize 32 \
--replay-buffer-size 100000 \
--warmup-period 1000  \
--max-steps 9000 \
--reward-clip 1 \
--model-path ./saved_models
