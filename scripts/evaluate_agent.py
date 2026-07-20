import argparse
from pathlib import Path
from typing import Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from powergym.env_register import remove_runtime_artifacts
from rl_lib.adaptor import action_tensor_to_env, compute_cost, make_powergym_train_env, reset_with_random_profile
from rl_lib.common.checkpoint import build_model_from_checkpoint, load_checkpoint
from rl_lib.common import infer_run_metadata_from_checkpoint
from rl_lib.common.spaces import parse_action_space


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PowerGym agent and plot voltage magnitude / rewards.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--load_profile_idx",
        type=int,
        default=None,
        help="Optional fixed load profile index. Defaults to a seeded random choice.",
    )
    parser.add_argument(
        "--plot_node_voltage_graph",
        action="store_true",
        help="Also save the final network voltage layout using env.plot_graph().",
    )
    parser.add_argument("--keep_runtime_artifacts", action="store_true")
    return parser.parse_args()


def collect_voltage_stats(bus_voltages: Dict[str, List[float]]) -> Dict[str, float]:
    flat = np.hstack([np.asarray(v, dtype=np.float32) for v in bus_voltages.values()])
    return {
        "min": float(flat.min()),
        "mean": float(flat.mean()),
        "max": float(flat.max()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    project_root = PROJECT_ROOT
    checkpoint_path = Path(args.checkpoint_path).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (project_root / checkpoint_path).resolve()

    output_dir_arg = None
    if args.output_dir is not None:
        output_dir_arg = Path(args.output_dir).expanduser()
        if not output_dir_arg.is_absolute():
            output_dir_arg = (project_root / output_dir_arg).resolve()

    checkpoint = load_checkpoint(str(checkpoint_path), device)
    run_token = "eval"
    if checkpoint_path.stem:
        run_token = f"eval_{checkpoint_path.stem}"
    model, train_args = build_model_from_checkpoint(checkpoint, device, run_token=run_token)

    env = make_powergym_train_env(train_args["env_name"], train_args.get("observe_load", False), run_token=run_token)
    action_spec = parse_action_space(env.action_space)

    if output_dir_arg is None:
        timestamp, algo_name, seed_name = infer_run_metadata_from_checkpoint(checkpoint_path)
        if timestamp is None:
            output_dir = project_root / "viz" / "adhoc_evaluation"
        else:
            output_dir = project_root / "viz" / timestamp / algo_name / seed_name
    else:
        output_dir = output_dir_arg
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    if args.load_profile_idx is None:
        obs, profile_idx = reset_with_random_profile(env, rng)
    else:
        profile_idx = args.load_profile_idx
        obs = env.reset(load_profile_idx=profile_idx)

    rewards: List[float] = []
    costs: List[float] = []
    min_voltages: List[float] = []
    mean_voltages: List[float] = []
    max_voltages: List[float] = []

    done = False
    while not done:
        stats = collect_voltage_stats(env.obs["bus_voltages"])
        min_voltages.append(stats["min"])
        mean_voltages.append(stats["mean"])
        max_voltages.append(stats["max"])

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            step_info = model.step(obs_tensor, deterministic=True)
        env_action = action_tensor_to_env(step_info["action"].squeeze(0), action_spec)
        obs, reward, done, info = env.step(env_action)
        rewards.append(float(reward))
        costs.append(float(compute_cost(info, train_args.get("cost_mode", "voltage"))))

    stats = collect_voltage_stats(env.obs["bus_voltages"])
    min_voltages.append(stats["min"])
    mean_voltages.append(stats["mean"])
    max_voltages.append(stats["max"])

    steps_voltage = np.arange(len(min_voltages))
    steps_reward = np.arange(1, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps_voltage, min_voltages, label="Min Voltage", linewidth=2.0)
    ax.plot(steps_voltage, mean_voltages, label="Mean Voltage", linewidth=2.0)
    ax.plot(steps_voltage, max_voltages, label="Max Voltage", linewidth=2.0)
    ax.axhline(0.95, color="tab:red", linestyle="--", linewidth=1.5, label="Lower Limit")
    ax.axhline(1.05, color="tab:orange", linestyle="--", linewidth=1.5, label="Upper Limit")
    ax.set_title(f"Voltage Magnitude (p.u.) | Profile {profile_idx}")
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("Voltage (p.u.)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    voltage_plot_path = output_dir / "voltage_magnitude.png"
    fig.savefig(voltage_plot_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps_reward, rewards, label="Reward", linewidth=2.0)
    ax.plot(steps_reward, costs, label="Cost", linewidth=2.0)
    ax.set_title(f"Episode Reward / Cost | Profile {profile_idx}")
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    reward_plot_path = output_dir / "episode_reward.png"
    fig.savefig(reward_plot_path, dpi=200)
    plt.close(fig)

    if args.plot_node_voltage_graph:
        fig, _ = env.plot_graph()
        fig.tight_layout()
        graph_plot_path = output_dir / "final_voltage_layout.png"
        fig.savefig(graph_plot_path, dpi=200)
        plt.close(fig)

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as fout:
        fout.write(f"checkpoint={checkpoint_path}\n")
        fout.write(f"env_name={train_args['env_name']}\n")
        fout.write(f"load_profile_idx={profile_idx}\n")
        fout.write(f"episode_return={sum(rewards):.6f}\n")
        fout.write(f"episode_cost={sum(costs):.6f}\n")
        fout.write(f"min_voltage={min(min_voltages):.6f}\n")
        fout.write(f"max_voltage={max(max_voltages):.6f}\n")

    print(voltage_plot_path)
    print(reward_plot_path)
    print(summary_path)
    if not args.keep_runtime_artifacts:
        remove_runtime_artifacts(train_args["env_name"], run_token)


if __name__ == "__main__":
    main()
