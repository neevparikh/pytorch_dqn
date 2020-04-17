#!/usr/bin/env bash
python train.py \
--env "CartPole-v1" \
--model-type mlp \
--gpu \
--seed 5 \
--lr 0.0005 \
--batchsize 32 \
--replay-buffer-size 100000 \
--warmup-period 500  \
--max-steps 20000 \
--reward-clip 1 \
--epsilon-decay 10000 \
--model-path ./saved_models
