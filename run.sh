#!/usr/bin/env bash
python train.py \
--env "PongNoFrameskip-v4" \
--model-type cnn \
# --env "CartPole-v1" \
# --model-type mlp \
--gpu \
--seed 5 \
--lr 0.00025 \
--batchsize 32 \
--replay-buffer-size 1000000 \
--warmup-period 100000  \
--max-steps 2000000 \
--test-policy-episodes 10 \
--reward-clip 1 \
--epsilon-decay 20000 \
--model-path ./saved_models
