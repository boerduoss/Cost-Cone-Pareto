TIMESTAMP="$(ls -1 runs | sort | tail -n 1)"


uv run python plot_training_results.py \
    --timestamp "$TIMESTAMP" \
    --algo ppo ppo_lag caf_cone \
    --seeds 1 2 3 \
    --value_key train_return \
    --smooth_window 5 \
    --runs_dir runs \
    --viz_dir viz


uv run python plot_training_results.py \
    --timestamp 20260308_112234 \
    --algo ppo ppo_lag \
    --seeds 1 2 3 \
    --value_key train_cost train_return \
    --smooth_window 5 \
    --runs_dir runs \
    --viz_dir viz