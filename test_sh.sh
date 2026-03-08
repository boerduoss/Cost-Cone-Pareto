uv run python evaluate_agent.py \
    --checkpoint_path runs/20260308_154152/caf_cone_pareto/seed1/latest.pt \
    --device cpu \
    --plot_node_voltage_graph


uv run python test_timestamp_trajectory.py \
    --timestamp 20260308_201032 \
    --algos ppo ppo_lag caf_cone caf_cone_pareto cost_cone cost_cone_pareto \
    --device cpu

