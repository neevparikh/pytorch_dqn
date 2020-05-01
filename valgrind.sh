#!/usr/bin/env bash

PYTHONMALLOC=malloc valgrind --leak-check=full python3 train.py \
--env "BreakoutNoFrameskip-v4" \
--model-type cnn \
--seed 5 \
--lr 0.00025 \
--batchsize 32 \
--replay-buffer-size 100000 \
--warmup-period 10  \
--max-steps 20 \
--test-policy-episodes 10000 \
--reward-clip 1 \
--epsilon-decay 10 \
--model-path ./saved_models

