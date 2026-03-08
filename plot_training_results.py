import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot smoothed reward mean/std over multiple seeds.")
    parser.add_argument("--timestamp", type=str, required=True, help="Training timestamp under runs/<timestamp>/...")
    parser.add_argument("--algo", nargs="+", choices=["ppo", "ppo_lag", "caf_cone", "caf_cone_pareto", "cost_cone", "cost_cone_pareto"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Seed directories to aggregate, e.g. 1 2 3")
    parser.add_argument("--value_key", nargs="+", default=["train_return"], help="Column(s) in metrics.csv to plot")
    parser.add_argument("--smooth_window", type=int, default=5, help="Rolling window size for smoothing")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--viz_dir", type=str, default="viz")
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def load_seed_curve(metrics_path: Path, value_key: str, smooth_window: int) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    if value_key not in df.columns:
        raise KeyError(f"{value_key} not found in {metrics_path}")
    result = df[["total_env_steps", value_key]].copy()
    result[value_key] = smooth_series(result[value_key], smooth_window)
    return result


def merge_seed_curves(curves: List[pd.DataFrame], value_key: str) -> pd.DataFrame:
    merged = curves[0].rename(columns={value_key: "seed_0"})
    for idx, curve in enumerate(curves[1:], start=1):
        merged = merged.merge(
            curve.rename(columns={value_key: f"seed_{idx}"}),
            on="total_env_steps",
            how="inner",
        )
    return merged


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    runs_dir = Path(args.runs_dir).expanduser()
    if not runs_dir.is_absolute():
        runs_dir = (project_root / runs_dir).resolve()

    viz_dir = Path(args.viz_dir).expanduser()
    if not viz_dir.is_absolute():
        viz_dir = (project_root / viz_dir).resolve()

    output_paths = []
    for algo in args.algo:
        for value_key in args.value_key:
            curves = []
            for seed in args.seeds:
                metrics_path = runs_dir / args.timestamp / algo / f"seed{seed}" / "metrics.csv"
                curves.append(load_seed_curve(metrics_path, value_key, args.smooth_window))

            merged = merge_seed_curves(curves, value_key)
            seed_columns = [col for col in merged.columns if col.startswith("seed_")]
            values = merged[seed_columns].to_numpy(dtype=np.float64)
            mean = values.mean(axis=1)
            std = values.std(axis=1)
            ci95 = 1.96 * std / np.sqrt(values.shape[1])

            if args.output_path is None:
                output_dir = viz_dir / args.timestamp
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{algo}_{value_key}_mean_std.png"
            else:
                if len(args.algo) > 1 or len(args.value_key) > 1:
                    raise ValueError("--output_path only supports a single algo and a single value_key.")
                output_path = Path(args.output_path).expanduser()
                if not output_path.is_absolute():
                    output_path = (project_root / output_path).resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            x = merged["total_env_steps"].to_numpy()
            ax.plot(x, mean, linewidth=2.5, label=f"{value_key} mean")
            ax.fill_between(x, mean - ci95, mean + ci95, alpha=0.25, label="95% CI")
            ax.set_title(f"Smoothed Reward Across Seeds | {algo} | {args.timestamp}")
            ax.set_xlabel("Environment Steps")
            ax.set_ylabel(value_key)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
            output_paths.append(output_path)

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
