import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from powergym.env_register import remove_runtime_artifacts
from rl_lib.adaptor import action_tensor_to_env, compute_cost, make_powergym_train_env, reset_with_random_profile
from rl_lib.common import infer_run_metadata_from_checkpoint
from rl_lib.common.checkpoint import build_model_from_checkpoint, load_checkpoint
from rl_lib.common.spaces import parse_action_space


DEFAULT_AUX_BUSES = {
    "13Bus": {"sourcebus", "rg60", "650"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a post-training trajectory test and visualize bus voltages, battery states, rewards, and costs."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--load_profile_idx", type=int, default=None)
    parser.add_argument("--bus_stat", choices=["min", "mean", "max"], default="min")
    parser.add_argument(
        "--include_aux_buses",
        action="store_true",
        help="For 13Bus, include sourcebus/rg60/650 instead of only the 13 feeder nodes.",
    )
    parser.add_argument(
        "--plot_node_voltage_graph",
        action="store_true",
        help="Also save the final network voltage layout using env.plot_graph().",
    )
    parser.add_argument("--keep_runtime_artifacts", action="store_true")
    return parser.parse_args()


def reduce_bus_voltage(values: List[float], mode: str) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if mode == "min":
        return float(arr.min())
    if mode == "mean":
        return float(arr.mean())
    if mode == "max":
        return float(arr.max())
    raise ValueError(f"Unknown bus_stat: {mode}")


def collect_voltage_stats(bus_voltages: Dict[str, List[float]]) -> Dict[str, float]:
    flat = np.hstack([np.asarray(v, dtype=np.float32) for v in bus_voltages.values()])
    return {
        "min": float(flat.min()),
        "mean": float(flat.mean()),
        "max": float(flat.max()),
    }


def select_voltage_buses(env_name: str, bus_names: List[str], include_aux_buses: bool) -> List[str]:
    selected = sorted(bus_names)
    if include_aux_buses:
        return selected
    aux = DEFAULT_AUX_BUSES.get(env_name, set())
    filtered = [bus for bus in selected if bus not in aux]
    return filtered if filtered else selected


def record_state(
    env,
    selected_buses: List[str],
    bus_stat: str,
    bus_series: Dict[str, List[float]],
    battery_soc_series: Dict[str, List[float]],
    battery_power_series: Dict[str, List[float]],
    voltage_stats_series: Dict[str, List[float]],
) -> None:
    for bus in selected_buses:
        bus_series[bus].append(reduce_bus_voltage(env.obs["bus_voltages"][bus], bus_stat))

    for battery_name, values in env.obs["bat_statuses"].items():
        soc, power_norm = values
        battery_soc_series.setdefault(battery_name, []).append(float(soc))
        battery_power_series.setdefault(battery_name, []).append(float(power_norm))

    voltage_stats = collect_voltage_stats(env.obs["bus_voltages"])
    for key, value in voltage_stats.items():
        voltage_stats_series[key].append(value)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    project_root = Path(__file__).resolve().parent
    checkpoint_path = Path(args.checkpoint_path).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (project_root / checkpoint_path).resolve()

    checkpoint = load_checkpoint(str(checkpoint_path), device)
    timestamp, algo_name, seed_name = infer_run_metadata_from_checkpoint(checkpoint_path)
    run_token_parts = ["test"]
    if timestamp is not None:
        run_token_parts.append(timestamp)
    if algo_name is not None:
        run_token_parts.append(algo_name)
    if seed_name is not None:
        run_token_parts.append(seed_name)
    run_token_parts.append(f"seed{args.seed}")
    run_token = "_".join(run_token_parts)

    model, train_args = build_model_from_checkpoint(checkpoint, device, run_token=run_token)
    env = make_powergym_train_env(
        train_args["env_name"],
        train_args.get("observe_load", False),
        run_token=run_token,
    )
    action_spec = parse_action_space(env.action_space)

    if args.output_dir is None:
        if timestamp is None or algo_name is None or seed_name is None:
            output_dir = project_root / "viz" / "trajectory_test"
        else:
            output_dir = project_root / "viz" / timestamp / algo_name / seed_name / "trajectory_test"
    else:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    if args.load_profile_idx is None:
        obs, profile_idx = reset_with_random_profile(env, rng)
    else:
        profile_idx = args.load_profile_idx
        obs = env.reset(load_profile_idx=profile_idx)

    selected_buses = select_voltage_buses(train_args["env_name"], list(env.obs["bus_voltages"].keys()), args.include_aux_buses)
    bus_series = {
        bus: [reduce_bus_voltage(env.obs["bus_voltages"][bus], args.bus_stat)]
        for bus in selected_buses
    }
    battery_soc_series: Dict[str, List[float]] = {}
    battery_power_series: Dict[str, List[float]] = {}
    voltage_stats_series = {"min": [], "mean": [], "max": []}
    rewards: List[float] = []
    costs: List[float] = []

    done = False
    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            step_info = model.step(obs_tensor, deterministic=True)
        env_action = action_tensor_to_env(step_info["action"].squeeze(0), action_spec)
        obs, reward, done, info = env.step(env_action)
        rewards.append(float(reward))
        costs.append(float(compute_cost(info, train_args.get("cost_mode", "voltage"))))

        # Record the post-action state so voltages, battery state, reward, and cost
        # all align to the same environment step.
        record_state(
            env,
            selected_buses,
            args.bus_stat,
            bus_series,
            battery_soc_series,
            battery_power_series,
            voltage_stats_series,
        )

    transition_steps = np.arange(1, len(rewards) + 1, dtype=np.int32)

    voltage_plot_steps = np.arange(len(next(iter(bus_series.values()))), dtype=np.int32)

    voltage_df = pd.DataFrame({"step": voltage_plot_steps})
    for bus in selected_buses:
        voltage_df[bus] = bus_series[bus]
    voltage_df_path = output_dir / "bus_voltage_timeseries.csv"
    voltage_df.to_csv(voltage_df_path, index=False)

    battery_df = pd.DataFrame({"step": transition_steps})
    for battery_name in sorted(battery_soc_series):
        battery_df[f"{battery_name}_soc"] = battery_soc_series[battery_name]
        battery_df[f"{battery_name}_power_norm"] = battery_power_series[battery_name]
    battery_df_path = output_dir / "battery_timeseries.csv"
    battery_df.to_csv(battery_df_path, index=False)

    transition_df = pd.DataFrame(
        {
            "step": transition_steps,
            "reward": rewards,
            "cost": costs,
            "voltage_min": voltage_stats_series["min"],
            "voltage_mean": voltage_stats_series["mean"],
            "voltage_max": voltage_stats_series["max"],
        }
    )
    transition_df_path = output_dir / "step_metrics.csv"
    transition_df.to_csv(transition_df_path, index=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    for bus in selected_buses:
        values = bus_series[bus]
        if len(values) >= 2:
            dashed_line, = ax.plot(
                voltage_plot_steps[:2],
                values[:2],
                linewidth=1.8,
                linestyle="--",
                label=bus,
            )
            color = dashed_line.get_color()
            if len(values) > 2:
                ax.plot(
                    voltage_plot_steps[1:],
                    values[1:],
                    linewidth=1.8,
                    linestyle="-",
                    color=color,
                )
        else:
            ax.plot(voltage_plot_steps, values, linewidth=1.8, linestyle="--", label=bus)
    ax.axhline(0.95, color="tab:red", linestyle="--", linewidth=1.2, label="Lower Limit")
    ax.axhline(1.05, color="tab:orange", linestyle="--", linewidth=1.2, label="Upper Limit")
    ax.set_title(f"Bus Voltage ({args.bus_stat}) | Profile {profile_idx}")
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("Voltage (p.u.)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
    fig.tight_layout()
    all_bus_voltage_path = output_dir / "all_bus_voltage_timeseries.png"
    fig.savefig(all_bus_voltage_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if battery_soc_series:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for battery_name in sorted(battery_soc_series):
            axes[0].plot(transition_steps, battery_soc_series[battery_name], linewidth=2.0, label=battery_name)
            axes[1].plot(transition_steps, battery_power_series[battery_name], linewidth=2.0, label=battery_name)
        axes[0].set_title("Battery State of Charge")
        axes[0].set_ylabel("SOC")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")
        axes[1].set_title("Battery Power")
        axes[1].set_xlabel("Episode Step")
        axes[1].set_ylabel("Normalized Power")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")
        fig.tight_layout()
        battery_plot_path = output_dir / "battery_timeseries.png"
        fig.savefig(battery_plot_path, dpi=200)
        plt.close(fig)
    else:
        battery_plot_path = None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(transition_steps, rewards, label="Reward", linewidth=2.0)
    ax.plot(transition_steps, costs, label="Cost", linewidth=2.0)
    ax.set_title(f"Episode Reward / Cost | Profile {profile_idx}")
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    reward_cost_path = output_dir / "episode_reward_cost.png"
    fig.savefig(reward_cost_path, dpi=200)
    plt.close(fig)

    if args.plot_node_voltage_graph:
        fig, _ = env.plot_graph()
        fig.tight_layout()
        final_graph_path = output_dir / "final_voltage_layout.png"
        fig.savefig(final_graph_path, dpi=200)
        plt.close(fig)
    else:
        final_graph_path = None

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as fout:
        fout.write(f"checkpoint={checkpoint_path}\n")
        fout.write(f"env_name={train_args['env_name']}\n")
        fout.write(f"load_profile_idx={profile_idx}\n")
        fout.write(f"bus_stat={args.bus_stat}\n")
        fout.write(f"selected_buses={','.join(selected_buses)}\n")
        fout.write(f"episode_return={sum(rewards):.6f}\n")
        fout.write(f"episode_cost={sum(costs):.6f}\n")
        fout.write(f"min_voltage={min(voltage_stats_series['min']):.6f}\n")
        fout.write(f"max_voltage={max(voltage_stats_series['max']):.6f}\n")

    print(all_bus_voltage_path)
    if battery_plot_path is not None:
        print(battery_plot_path)
    print(reward_cost_path)
    print(voltage_df_path)
    print(battery_df_path)
    print(transition_df_path)
    print(summary_path)
    if final_graph_path is not None:
        print(final_graph_path)
    if not args.keep_runtime_artifacts:
        remove_runtime_artifacts(train_args["env_name"], run_token)


if __name__ == "__main__":
    main()
