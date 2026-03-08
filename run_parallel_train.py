import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from rl_lib.common import generate_timestamp


@dataclass
class Job:
    algo: str
    seed: int
    slot: int
    command: List[str]
    stdout_path: Path
    stderr_path: Path
    process: Optional[subprocess.Popen] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch multiple PowerGym training jobs in parallel.")
    parser.add_argument("--algos", nargs="+", required=True, choices=["ppo", "ppo_lag", "caf_cone", "caf_cone_pareto", "cost_cone", "cost_cone_pareto"])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--max_parallel", type=int, default=1)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--viz_dir", type=str, default="viz")
    parser.add_argument("--plot_keys", nargs="+", default=["train_return", "eval_return", "train_cost", "eval_cost"])
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--no_auto_plot", action="store_true")
    parser.add_argument("--poll_interval", type=float, default=2.0)
    parser.add_argument("--dry_run", action="store_true")
    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    args.extra_args = extra_args
    return args


def build_jobs(args: argparse.Namespace, project_root: Path) -> List[Job]:
    timestamp = args.timestamp or generate_timestamp()
    args.timestamp = timestamp

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    log_dir = output_dir / timestamp / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Job] = []
    existing_run_dirs: List[str] = []
    slot = 0
    for algo in args.algos:
        for seed in args.seeds:
            run_dir = output_dir / timestamp / algo / f"seed{seed}"
            if run_dir.exists():
                existing_run_dirs.append(str(run_dir))
            stdout_path = log_dir / f"{algo}_seed{seed}.out"
            stderr_path = log_dir / f"{algo}_seed{seed}.err"
            command = [
                sys.executable,
                str(project_root / "train_agent.py"),
                "--algo",
                algo,
                "--seed",
                str(seed),
                "--timestamp",
                timestamp,
                "--output_dir",
                str(output_dir),
                "--worker_idx",
                str(slot),
                *args.extra_args,
            ]
            jobs.append(
                Job(
                    algo=algo,
                    seed=seed,
                    slot=slot,
                    command=command,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
            )
            slot = (slot + 1) % max(1, args.max_parallel)

    if existing_run_dirs:
        conflicts = "\n".join(existing_run_dirs)
        raise FileExistsError(
            "Refusing to start because the following run directories already exist:\n"
            f"{conflicts}\n"
            "Use a fresh timestamp, or delete the old run directories first."
        )

    manifest = {
        "timestamp": timestamp,
        "algos": args.algos,
        "seeds": args.seeds,
        "max_parallel": args.max_parallel,
        "extra_args": args.extra_args,
        "jobs": [
            {
                "algo": job.algo,
                "seed": job.seed,
                "seed_dir": f"seed{job.seed}",
                "slot": job.slot,
                "command": job.command,
                "stdout": str(job.stdout_path),
                "stderr": str(job.stderr_path),
            }
            for job in jobs
        ],
    }
    with (output_dir / timestamp / "launch_manifest.json").open("w", encoding="utf-8") as fout:
        json.dump(manifest, fout, indent=2, ensure_ascii=True)
    return jobs


def launch_job(job: Job) -> None:
    stdout_file = job.stdout_path.open("w", encoding="utf-8")
    stderr_file = job.stderr_path.open("w", encoding="utf-8")
    job.process = subprocess.Popen(
        job.command,
        stdout=stdout_file,
        stderr=stderr_file,
        cwd=job.command[1] and str(Path(job.command[1]).resolve().parent),
    )
    job._stdout_file = stdout_file  # type: ignore[attr-defined]
    job._stderr_file = stderr_file  # type: ignore[attr-defined]


def close_job_files(job: Job) -> None:
    job._stdout_file.close()  # type: ignore[attr-defined]
    job._stderr_file.close()  # type: ignore[attr-defined]


def generate_plots(
    args: argparse.Namespace,
    project_root: Path,
    completed: List[Dict[str, int]],
) -> List[Dict[str, str]]:
    if args.no_auto_plot:
        return []

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    successful_seeds_by_algo: Dict[str, List[int]] = {}
    for item in completed:
        if item["returncode"] != 0:
            continue
        successful_seeds_by_algo.setdefault(item["algo"], []).append(item["seed"])

    generated: List[Dict[str, str]] = []
    for algo, seeds in successful_seeds_by_algo.items():
        seeds = sorted(set(seeds))
        if not seeds:
            continue
        for value_key in args.plot_keys:
            cmd = [
                sys.executable,
                str(project_root / "plot_training_results.py"),
                "--timestamp",
                args.timestamp,
                "--algo",
                algo,
                "--seeds",
                *[str(seed) for seed in seeds],
                "--value_key",
                value_key,
                "--smooth_window",
                str(args.smooth_window),
                "--runs_dir",
                str(output_dir),
                "--viz_dir",
                args.viz_dir,
            ]
            subprocess.run(cmd, check=True, cwd=str(project_root))
            generated.append({"algo": algo, "value_key": value_key})
    return generated


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    jobs = build_jobs(args, project_root)

    print(f"timestamp={args.timestamp}")
    for job in jobs:
        print(f"queued algo={job.algo} seed={job.seed} slot={job.slot}")
        print("  " + " ".join(job.command))

    if args.dry_run:
        return

    pending = jobs.copy()
    running: List[Job] = []
    completed: List[Dict[str, int]] = []
    free_slots = list(range(max(1, args.max_parallel)))

    while pending or running:
        while pending and free_slots:
            slot = free_slots.pop(0)
            next_index = next(i for i, job in enumerate(pending) if job.slot == slot)
            job = pending.pop(next_index)
            launch_job(job)
            running.append(job)
            print(f"started algo={job.algo} seed={job.seed} slot={slot}")

        time.sleep(args.poll_interval)
        still_running: List[Job] = []
        for job in running:
            assert job.process is not None
            returncode = job.process.poll()
            if returncode is None:
                still_running.append(job)
                continue
            close_job_files(job)
            free_slots.append(job.slot)
            free_slots.sort()
            completed.append({"algo": job.algo, "seed": job.seed, "returncode": returncode})
            status = "finished" if returncode == 0 else "failed"
            print(f"{status} algo={job.algo} seed={job.seed} slot={job.slot} returncode={returncode}")
        running = still_running

    failures = [item for item in completed if item["returncode"] != 0]
    summary_path = project_root / args.output_dir / args.timestamp / "launch_summary.json"
    if not summary_path.is_absolute():
        summary_path = summary_path.resolve()
    generated_plots = generate_plots(args, project_root, completed)
    with summary_path.open("w", encoding="utf-8") as fout:
        json.dump({"completed": completed, "failures": failures, "plots": generated_plots}, fout, indent=2, ensure_ascii=True)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
