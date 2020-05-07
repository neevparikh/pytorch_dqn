#!/usr/bin/env bash
python train.py \
--env "PongNoFrameskip-v4" \
--model-type cnn \
--gpu \
--seed 5 \
--lr 0.001 \
--batchsize 32 \
--replay-buffer-size 1e6 \
--warmup-period 5e4  \
--max-steps 5e6 \
--test-policy-episodes 10 \
--reward-clip 1 \
--epsilon-decay 1e6 \
--model-path ./saved_models
