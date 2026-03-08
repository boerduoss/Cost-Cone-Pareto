import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from data_aggregation.plot_style import ALGO_COLORS, ALGO_LABELS, apply_plot_style
except ModuleNotFoundError:
    from plot_style import ALGO_COLORS, ALGO_LABELS, apply_plot_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare train_return and train_cost across algorithms on a single figure."
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["ppo", "ppo_lag", "caf_cone", "caf_cone_pareto", "cost_cone", "cost_cone_pareto"],
        help="Algorithms to compare. Defaults to all supported algorithms under data_aggregation/.",
    )
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="data_aggregation")
    parser.add_argument("--output_name", type=str, default="algorithm_train_return_cost_comparison.png")
    return parser.parse_args()


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def load_algo_curves(algo_dir: Path, value_key: str, smooth_window: int) -> List[pd.DataFrame]:
    seed_dirs = [path for path in algo_dir.iterdir() if path.is_dir() and path.name.startswith("seed")]
    seed_dirs.sort(key=lambda path: int(path.name.replace("seed", "")))
    curves: List[pd.DataFrame] = []
    for seed_dir in seed_dirs:
        metrics_path = seed_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        if value_key not in df.columns:
            raise KeyError(f"{value_key} not found in {metrics_path}")
        curve = df[["total_env_steps", value_key]].copy()
        curve[value_key] = smooth_series(curve[value_key], smooth_window)
        curves.append(curve)
    return curves


def merge_seed_curves(curves: List[pd.DataFrame], value_key: str) -> pd.DataFrame:
    merged = curves[0].rename(columns={value_key: "seed_0"})
    for idx, curve in enumerate(curves[1:], start=1):
        merged = merged.merge(
            curve.rename(columns={value_key: f"seed_{idx}"}),
            on="total_env_steps",
            how="inner",
        )
    return merged


def aggregate_algo(algo_dir: Path, value_key: str, smooth_window: int) -> pd.DataFrame:
    curves = load_algo_curves(algo_dir, value_key, smooth_window)
    if not curves:
        raise FileNotFoundError(f"No metrics.csv files found under {algo_dir}")
    merged = merge_seed_curves(curves, value_key)
    seed_columns = [col for col in merged.columns if col.startswith("seed_")]
    values = merged[seed_columns].to_numpy(dtype=np.float64)
    result = pd.DataFrame(
        {
            "total_env_steps": merged["total_env_steps"].to_numpy(),
            "mean": values.mean(axis=1),
            "std": values.std(axis=1),
            "ci95": 1.96 * values.std(axis=1) / np.sqrt(values.shape[1]),
            "num_seeds": values.shape[1],
        }
    )
    return result


def main() -> None:
    args = parse_args()
    apply_plot_style()
    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = (script_dir.parent / data_dir).resolve()

    output_dir = data_dir / "comparison_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ["train_return", "train_cost"]
    aggregated: Dict[str, Dict[str, pd.DataFrame]] = {}

    for algo in args.algos:
        algo_dir = data_dir / algo
        if not algo_dir.exists():
            continue
        aggregated[algo] = {}
        for metric in metrics_to_plot:
            agg = aggregate_algo(algo_dir, metric, args.smooth_window)
            aggregated[algo][metric] = agg
            agg.to_csv(output_dir / f"{algo}_{metric}_aggregated.csv", index=False)

    if not aggregated:
        raise FileNotFoundError(f"No algorithm data found under {data_dir}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 10.5), sharex=True)
    legend_handles = {}
    for axis, metric in zip(axes, metrics_to_plot):
        for algo, algo_metrics in aggregated.items():
            agg = algo_metrics[metric]
            color = ALGO_COLORS.get(algo, None)
            x = agg["total_env_steps"].to_numpy()
            mean = agg["mean"].to_numpy()
            ci95 = agg["ci95"].to_numpy()
            line, = axis.plot(x, mean, linewidth=2.5, color=color, label=ALGO_LABELS.get(algo, algo.upper()))
            axis.fill_between(x, mean - ci95, mean + ci95, alpha=0.18, color=color)
            legend_handles[algo] = line
        axis.set_ylabel(metric.replace("_", " "), fontsize=22)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Environment Steps", fontsize=22)
    fig.suptitle("Algorithm Comparison Across Training", y=0.98)
    fig.legend(
        [legend_handles[algo] for algo in aggregated.keys()],
        [ALGO_LABELS.get(algo, algo.upper()) for algo in aggregated.keys()],
        loc="lower center",
        ncol=min(len(aggregated), 4),
        frameon=True,
        fancybox=False,
        edgecolor="0.2",
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    figure_path = output_dir / args.output_name
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    print(figure_path)
    for algo in aggregated:
        for metric in metrics_to_plot:
            print(output_dir / f"{algo}_{metric}_aggregated.csv")


if __name__ == "__main__":
    main()
