#!/usr/bin/env bash
python train.py \
--env "PongNoFrameskip-v4" \
--model-type cnn \
--no-atari \
--gpu \
--seed 5 \
--lr 0.00025 \
--batchsize 32 \
--replay-buffer-size 100000 \
--warmup-period 10000  \
--max-steps 2000000 \
--test-policy-episodes 10 \
--reward-clip 1 \
--epsilon-decay 20000 \
--model-path ./saved_models
