ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TIMESTAMP="$(ls -1 runs | sort | tail -n 1)"


uv run python scripts/plot_training_results.py \
    --timestamp "$TIMESTAMP" \
    --algo ppo ppo_lag caf_cone \
    --seeds 1 2 3 \
    --value_key train_return \
    --smooth_window 5 \
    --runs_dir runs \
    --viz_dir viz


uv run python scripts/plot_training_results.py \
    --timestamp 20260308_112234 \
    --algo ppo ppo_lag \
    --seeds 1 2 3 \
    --value_key train_cost train_return \
    --smooth_window 5 \
    --runs_dir runs \
    --viz_dir viz
