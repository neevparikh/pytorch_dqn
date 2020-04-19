#!/usr/bin/env bash
python3 train.py \
--env "SpaceInvaders-v0" \
--model-type cnn \
--gpu \
--seed 5 \
--lr 0.00025 \
--batchsize 32 \
--replay-buffer-size 1000000 \
--warmup-period 50000  \
--episodes 2000 \
--reward-clip 1 \
--model-path ./saved_models \
--render
