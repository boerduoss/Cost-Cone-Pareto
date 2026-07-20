import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SUPPORTED_ALGOS = ["ppo", "ppo_lag", "caf_cone", "caf_cone_pareto", "cost_cone", "cost_cone_pareto"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run trajectory tests for all algorithms/seeds under a training timestamp."
    )
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--algos", nargs="+", choices=SUPPORTED_ALGOS, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--viz_dir", type=str, default="viz")
    parser.add_argument("--checkpoint_name", type=str, default="best_reward.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation RNG seed passed to test_agent_trajectory.py")
    parser.add_argument("--load_profile_idx", type=int, default=None)
    parser.add_argument("--bus_stat", choices=["min", "mean", "max"], default="min")
    parser.add_argument("--include_aux_buses", action="store_true")
    parser.add_argument("--plot_node_voltage_graph", action="store_true")
    parser.add_argument("--max_parallel", type=int, default=1)
    return parser.parse_args()


def discover_jobs(args: argparse.Namespace, project_root: Path) -> List[Tuple[str, int, Path]]:
    runs_dir = Path(args.runs_dir).expanduser()
    if not runs_dir.is_absolute():
        runs_dir = (project_root / runs_dir).resolve()

    timestamp_dir = runs_dir / args.timestamp
    if not timestamp_dir.exists():
        raise FileNotFoundError(f"Timestamp directory not found: {timestamp_dir}")

    algos = args.algos
    if algos is None:
        algos = [path.name for path in timestamp_dir.iterdir() if path.is_dir() and path.name in SUPPORTED_ALGOS]
        algos.sort()

    jobs: List[Tuple[str, int, Path]] = []
    for algo in algos:
        algo_dir = timestamp_dir / algo
        if not algo_dir.exists():
            continue
        seed_dirs = [path for path in algo_dir.iterdir() if path.is_dir() and path.name.startswith("seed")]
        seed_dirs.sort(key=lambda path: int(path.name.replace("seed", "")))
        if args.seeds is None and seed_dirs:
            seed_dirs = seed_dirs[:1]
        for seed_dir in seed_dirs:
            seed = int(seed_dir.name.replace("seed", ""))
            if args.seeds is not None and seed not in args.seeds:
                continue
            checkpoint_path = seed_dir / args.checkpoint_name
            if checkpoint_path.exists():
                jobs.append((algo, seed, checkpoint_path))
    return jobs


def build_command(args: argparse.Namespace, project_root: Path, checkpoint_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "test_agent_trajectory.py"),
        "--checkpoint_path",
        str(checkpoint_path),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--bus_stat",
        args.bus_stat,
    ]
    if args.load_profile_idx is not None:
        cmd.extend(["--load_profile_idx", str(args.load_profile_idx)])
    if args.include_aux_buses:
        cmd.append("--include_aux_buses")
    if args.plot_node_voltage_graph:
        cmd.append("--plot_node_voltage_graph")
    return cmd


def run_one(args: argparse.Namespace, project_root: Path, job: Tuple[str, int, Path]) -> Tuple[str, int, int]:
    algo, seed, checkpoint_path = job
    cmd = build_command(args, project_root, checkpoint_path)
    result = subprocess.run(cmd, cwd=str(project_root))
    return algo, seed, result.returncode


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    jobs = discover_jobs(args, project_root)

    if not jobs:
        raise FileNotFoundError("No matching checkpoints found for the given timestamp/filters.")

    print(f"timestamp={args.timestamp}")
    for algo, seed, checkpoint_path in jobs:
        print(f"queued algo={algo} seed={seed} checkpoint={checkpoint_path}")

    failures: List[Tuple[str, int, int]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as executor:
        futures = {executor.submit(run_one, args, project_root, job): job for job in jobs}
        for future in as_completed(futures):
            algo, seed, _ = futures[future]
            result_algo, result_seed, returncode = future.result()
            status = "finished" if returncode == 0 else "failed"
            print(f"{status} algo={result_algo} seed={result_seed} returncode={returncode}")
            if returncode != 0:
                failures.append((algo, seed, returncode))

    if failures:
        details = ", ".join(f"{algo}/seed{seed}:{code}" for algo, seed, code in failures)
        raise SystemExit(f"Some trajectory tests failed: {details}")


if __name__ == "__main__":
    main()
