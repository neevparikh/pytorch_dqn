#!/usr/bin/env bash
python train.py \
--env "PongNoFrameskip-v4" \
--model-type cnn \
--batchsize 32 \
--replay-buffer-size 1e6 \
--warmup-period 5e3  \
--max-steps 5e6 \
--reward-clip 1 \
--model-path "./saved_models" \
--output_path "./reward_log" \
--epsilon-decay 1.5e6 \
--epsilon-decay-end 0.05 \
--test-policy-steps 250000 \
--target-moving-average 5e-3 \
--lr 5e-3 \
--uuid "lr5e-3" \
--seed 5
