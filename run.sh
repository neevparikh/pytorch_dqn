#!/usr/bin/env bash
python train.py \
--env "PongNoFrameskip-v4" \
--model-type cnn \
--seed 5 \
--lr 0.001 \
--target-moving-average 5e-3 \
--batchsize 32 \
--replay-buffer-size 1e6 \
--warmup-period 5e3  \
--max-steps 5e6 \
--test-policy-episodes 10 \
--reward-clip 1 \
--epsilon-decay 1.5e6 \
--epsilon-decay-end 0.05 \
--model-path ./saved_models
