uv run python data_aggregation/plot_voltage_trajectory_comparison.py \
    --algos ppo ppo_lag cost_cone_pareto \
    --seed 1

uv run python data_aggregation/plot_algorithm_comparison.py \
    --algos ppo ppo_lag cost_cone_pareto \
    --smooth_window 5
