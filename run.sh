#!/usr/bin/env bash

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"


uv run python run_parallel_train.py \
    --algos caf_cone ppo ppo_lag \
    --seeds 1 2 3 \
    --max_parallel 9 \
    --timestamp "$(date +%Y%m%d_%H%M%S)" \
    --output_dir runs \
    -- \
    --env_name 13Bus \
    --total_steps 1000000 \
    --steps_per_epoch 4096 \
    --train_pi_iters 10 \
    --train_v_iters 10 \
    --train_f_iters 2 \
    --minibatch_size 256 \
    --num_evals 5 \
    --eval_interval 0 \
    --eval_episodes 4 \
    --device cpu \
    --torch_num_threads 1 \
    --torch_num_interop_threads 1


# uv run python run_parallel_train.py \
#     --algos ppo ppo_lag \
#     --seeds 1 2 3 \
#     --max_parallel 3 \
#     --timestamp "$(date +%Y%m%d_%H%M%S)" \
#     --output_dir runs \
#     -- \
#     --env_name 13Bus \
#     --total_steps 1000000 \
#     --steps_per_epoch 4096 \
#     --train_pi_iters 10 \
#     --train_v_iters 10 \
#     --minibatch_size 256 \
#     --num_evals 5 \
#     --eval_interval 0 \
#     --eval_episodes 4 \
#     --device cpu \
#     --torch_num_threads 1 \
#     --torch_num_interop_threads 1


# uv run python run_parallel_train.py \
#     --algos caf_cone \
#     --seeds 1 2 3 \
#     --max_parallel 3 \
#     --timestamp "$(date +%Y%m%d_%H%M%S)" \
#     --output_dir runs \
#     -- \
#     --env_name 13Bus \
#     --total_steps 1000000 \
#     --steps_per_epoch 4096 \
#     --train_pi_iters 10 \
#     --train_v_iters 10 \
#     --train_f_iters 2 \
#     --minibatch_size 256 \
#     --num_evals 5 \
#     --eval_interval 0 \
#     --eval_episodes 4 \
#     --device cpu \
#     --torch_num_threads 1 \
#     --torch_num_interop_threads 1



