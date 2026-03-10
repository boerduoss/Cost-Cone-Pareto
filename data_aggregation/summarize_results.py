import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from data_aggregation.plot_style import ALGO_LABELS
except ModuleNotFoundError:
    from plot_style import ALGO_LABELS


DEFAULT_ALGOS = ["cost_cone_pareto", "ppo_lag", "ppo"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize train/test return, cost, and voltage violation rate for selected algorithms."
    )
    parser.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS)
    parser.add_argument("--data_dir", type=str, default="data_aggregation")
    parser.add_argument("--viz_dir", type=str, default="viz")
    parser.add_argument("--seed", type=int, default=1, help="Test seed used for trajectory_test files.")
    parser.add_argument("--output_dir", type=str, default="data_aggregation/comparison_outputs")
    parser.add_argument("--lower_bound", type=float, default=0.95)
    parser.add_argument("--upper_bound", type=float, default=1.05)
    return parser.parse_args()


def load_training_summary(algo_dir: Path) -> Tuple[float, float, float, float]:
    seed_dirs = [path for path in algo_dir.iterdir() if path.is_dir() and path.name.startswith("seed")]
    seed_dirs.sort(key=lambda path: int(path.name.replace("seed", "")))
    train_returns: List[float] = []
    train_costs: List[float] = []
    for seed_dir in seed_dirs:
        metrics_path = seed_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        last = df.iloc[-1]
        train_returns.append(float(last["train_return"]))
        train_costs.append(float(last["train_cost"]))
    if not train_returns:
        raise FileNotFoundError(f"No metrics found under {algo_dir}")
    return (
        float(pd.Series(train_returns).mean()),
        float(pd.Series(train_returns).std(ddof=0)),
        float(pd.Series(train_costs).mean()),
        float(pd.Series(train_costs).std(ddof=0)),
    )


def find_latest_test_paths(viz_dir: Path, algo: str, seed: int) -> Tuple[Path, Path]:
    step_candidates = sorted(viz_dir.glob(f"*/{algo}/seed{seed}/trajectory_test/step_metrics.csv"))
    voltage_candidates = sorted(viz_dir.glob(f"*/{algo}/seed{seed}/trajectory_test/bus_voltage_timeseries.csv"))
    if not step_candidates or not voltage_candidates:
        raise FileNotFoundError(f"Missing trajectory_test files for {algo}/seed{seed} under {viz_dir}")
    return step_candidates[-1], voltage_candidates[-1]


def load_test_summary(step_metrics_path: Path, voltage_path: Path, lower_bound: float, upper_bound: float) -> Tuple[float, float, float]:
    step_df = pd.read_csv(step_metrics_path)
    voltage_df = pd.read_csv(voltage_path)

    test_return = float(step_df["reward"].sum())
    test_cost = float(step_df["cost"].sum())

    voltage_df = voltage_df[voltage_df["step"] > 0].copy()
    node_columns = [col for col in voltage_df.columns if col != "step"]
    values = voltage_df[node_columns].to_numpy(dtype=float)
    violations = ((values < lower_bound) | (values > upper_bound)).sum()
    total = values.size
    violation_rate = float(violations / total) if total > 0 else 0.0
    return test_return, test_cost, violation_rate


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def to_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, separator, *rows])


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = (project_root / data_dir).resolve()

    viz_dir = Path(args.viz_dir).expanduser()
    if not viz_dir.is_absolute():
        viz_dir = (project_root / viz_dir).resolve()

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for algo in args.algos:
        algo_dir = data_dir / algo
        if not algo_dir.exists():
            continue

        train_return_mean, train_return_std, train_cost_mean, train_cost_std = load_training_summary(algo_dir)
        step_metrics_path, voltage_path = find_latest_test_paths(viz_dir, algo, args.seed)
        test_return, test_cost, violation_rate = load_test_summary(
            step_metrics_path,
            voltage_path,
            args.lower_bound,
            args.upper_bound,
        )

        rows.append(
            {
                "algorithm": ALGO_LABELS.get(algo, algo.upper()),
                "train_return": format_mean_std(train_return_mean, train_return_std),
                "train_cost": format_mean_std(train_cost_mean, train_cost_std),
                "test_return": f"{test_return:.3f}",
                "test_cost": f"{test_cost:.3f}",
                "voltage_violation_rate": f"{100.0 * violation_rate:.2f}%",
            }
        )

    if not rows:
        raise FileNotFoundError("No matching algorithms with both training metrics and test trajectory files were found.")

    df = pd.DataFrame(rows)
    csv_path = output_dir / "algorithm_result_summary.csv"
    md_path = output_dir / "algorithm_result_summary.md"
    df.to_csv(csv_path, index=False)
    markdown_table = to_markdown_table(df)
    with md_path.open("w", encoding="utf-8") as fout:
        fout.write(markdown_table)
        fout.write("\n")

    print(csv_path)
    print(md_path)
    print(markdown_table)


if __name__ == "__main__":
    main()
