#!/usr/bin/env bash
python train.py \
--env "Breakout-ram-v0" \
--model-type mlp \
--gpu \
--seed 5 \
--lr 0.00025 \
--batchsize 32 \
--replay-buffer-size 1000000 \
--episodes 300 \
--reward-clip 20 \
--render \
--render-episodes 10 \
--epsilon-decay 2.75 \
--epsilon-decay-start 5 
