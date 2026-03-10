import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

try:
    from data_aggregation.plot_style import ALGO_COLORS, ALGO_LABELS, apply_plot_style
except ModuleNotFoundError:
    from plot_style import ALGO_COLORS, ALGO_LABELS, apply_plot_style


SUPPORTED_ALGOS = ["ppo", "ppo_lag", "caf_cone", "caf_cone_pareto", "cost_cone", "cost_cone_pareto"]

NODE_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot test-stage bus voltage trajectories for algorithms in a 1x3 subplot layout."
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=SUPPORTED_ALGOS,
        help="Algorithms to plot. At most three are displayed in a 1x3 layout.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed index to use for each algorithm.")
    parser.add_argument("--viz_dir", type=str, default="viz")
    parser.add_argument("--output_dir", type=str, default="data_aggregation/comparison_outputs")
    parser.add_argument("--output_name", type=str, default="algorithm_voltage_trajectory_comparison.png")
    parser.add_argument("--ylim_low", type=float, default=0.9)
    parser.add_argument("--ylim_high", type=float, default=1.1)
    return parser.parse_args()


def find_latest_trajectory_csv(viz_dir: Path, algo: str, seed: int) -> Tuple[Path, str]:
    pattern = f"*/{algo}/seed{seed}/trajectory_test/bus_voltage_timeseries.csv"
    candidates = sorted(viz_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No trajectory voltage CSV found for {algo}/seed{seed} under {viz_dir}")
    latest = candidates[-1]
    timestamp = latest.parts[-5]
    return latest, timestamp


def build_node_color_map(loaded: Dict[str, Tuple[pd.DataFrame, str]]) -> Dict[str, str]:
    node_names = sorted(
        {
            column
            for df, _ in loaded.values()
            for column in df.columns
            if column != "step"
        }
    )
    return {
        node: NODE_COLORS[idx % len(NODE_COLORS)]
        for idx, node in enumerate(node_names)
    }


def main() -> None:
    args = parse_args()
    apply_plot_style()

    project_root = Path(__file__).resolve().parent.parent
    viz_dir = Path(args.viz_dir).expanduser()
    if not viz_dir.is_absolute():
        viz_dir = (project_root / viz_dir).resolve()

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    algos = args.algos
    loaded: Dict[str, Tuple[pd.DataFrame, str]] = {}
    for algo in algos:
        csv_path, timestamp = find_latest_trajectory_csv(viz_dir, algo, args.seed)
        loaded[algo] = (pd.read_csv(csv_path), timestamp)

    if not loaded:
        raise FileNotFoundError("No trajectory voltage CSV files were found.")

    node_color_map = build_node_color_map(loaded)

    algo_groups = [algos[idx : idx + 3] for idx in range(0, len(algos), 3)]
    output_paths = []
    for page_idx, algo_group in enumerate(algo_groups, start=1):
        fig, axes = plt.subplots(1, 3, figsize=(21, 6.8), sharey=True)
        if not isinstance(axes, (list, tuple)):
            try:
                axes_flat = axes.flatten()
            except AttributeError:
                axes_flat = [axes]
        else:
            axes_flat = list(axes)
        if len(algo_group) < 3:
            for axis in axes_flat[len(algo_group):]:
                axis.axis("off")

        upper_handle = None
        lower_handle = None

        for axis, algo in zip(axes_flat, algo_group):
            df, _ = loaded[algo]
            steps = df["step"]
            node_columns = [col for col in df.columns if col != "step"]

            for node in node_columns:
                axis.plot(
                    steps,
                    df[node],
                    color=node_color_map[node],
                    linewidth=1.15,
                    alpha=0.7,
                )

            lower_handle = axis.axhline(0.95, color="tab:red", linestyle="--", linewidth=2.0)
            upper_handle = axis.axhline(1.05, color="tab:orange", linestyle="--", linewidth=2.0)
            axis.set_title(ALGO_LABELS.get(algo, algo.upper()))
            axis.set_xlabel("Episode Step", fontsize=26)
            axis.set_ylim(args.ylim_low, args.ylim_high)
            axis.margins(x=0)
            axis.grid(True, alpha=0.25)

        if len(axes_flat) > 0:
            axes_flat[0].set_ylabel("Voltage (p.u.)", fontsize=26)
        fig.suptitle("Test-Phase Bus Voltage Trajectories", y=0.98)
        if lower_handle is not None and upper_handle is not None:
            fig.legend(
                [lower_handle, upper_handle],
                ["Lower Bound", "Upper Bound"],
                loc="lower center",
                ncol=2,
                frameon=True,
                fancybox=False,
                edgecolor="0.2",
                bbox_to_anchor=(0.5, 0.005),
            )
        fig.tight_layout(rect=[0, 0.12, 1, 0.93])

        if len(algo_groups) == 1:
            output_name = args.output_name
        else:
            stem = Path(args.output_name).stem
            suffix = Path(args.output_name).suffix or ".png"
            output_name = f"{stem}_page{page_idx}{suffix}"
        output_path = output_dir / output_name
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        output_paths.append(output_path)

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
