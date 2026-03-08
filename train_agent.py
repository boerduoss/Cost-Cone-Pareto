import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from powergym.env_register import remove_runtime_artifacts
from rl_lib import CAFConEConfig, CAFConEParetoUpdater, CAFConEUpdater, CostConEConfig, CostConEUpdater, CostConEParetoUpdater, PPOConfig, PPOLagConfig, PPOUpdater, PPOLagUpdater
from rl_lib.adaptor import action_tensor_to_env, compute_cost, evaluate_policy, make_powergym_train_env, reset_with_random_profile
from rl_lib.common import CSVLogger, RolloutBuffer, build_run_dir, generate_timestamp, infer_obs_dim, parse_action_space, safe_mean, set_seed
from rl_lib.networks import MultiHeadActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO or PPO-Lag on PowerGym.")
    parser.add_argument("--algo", choices=["ppo", "ppo_lag", "caf_cone", "caf_cone_pareto", "cost_cone", "cost_cone_pareto"], default="ppo_lag")
    parser.add_argument("--env_name", type=str, default="13Bus")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total_steps", type=int, default=8192)
    parser.add_argument("--steps_per_epoch", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--cost_gamma", type=float, default=0.99)
    parser.add_argument("--cost_gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--target_kl", type=float, default=0.02)
    parser.add_argument("--pi_lr", type=float, default=3e-4)
    parser.add_argument("--vf_lr", type=float, default=1e-3)
    parser.add_argument("--lagrangian_lr", type=float, default=0.05)
    parser.add_argument("--train_pi_iters", type=int, default=40)
    parser.add_argument("--train_v_iters", type=int, default=40)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--cost_limit", type=float, default=0.0)
    parser.add_argument("--initial_lagrange_multiplier", type=float, default=1e-3)
    parser.add_argument("--caf_beta", type=float, default=0.5)
    parser.add_argument("--f_lr", type=float, default=3e-4)
    parser.add_argument("--f_target_tau", type=float, default=0.05)
    parser.add_argument("--train_f_iters", type=int, default=2)
    parser.add_argument("--safety_gamma", type=float, default=0.99)
    parser.add_argument("--safety_gae_lambda", type=float, default=0.95)
    parser.add_argument("--safety_adv_scale", type=float, default=1.0)
    parser.add_argument("--safety_cost_threshold", type=float, default=0.0)
    parser.add_argument("--cost_mode", choices=["voltage", "control", "voltage_control"], default="voltage")
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--observe_load", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=0, help="Evaluate every N epochs. Use 0 to disable interval-based eval.")
    parser.add_argument("--num_evals", type=int, default=5, help="Total number of evaluations spread across the full run.")
    parser.add_argument("--eval_episodes", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--worker_idx", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torch_num_threads", type=int, default=1)
    parser.add_argument("--torch_num_interop_threads", type=int, default=1)
    parser.add_argument("--keep_runtime_artifacts", action="store_true")
    return parser.parse_args()

def save_checkpoint(path: Path, model: MultiHeadActorCritic, args: argparse.Namespace, metrics: Dict[str, float]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def build_eval_schedule(total_epochs: int, eval_interval: int, num_evals: int) -> set[int]:
    schedule: set[int] = set()
    if eval_interval > 0:
        schedule.update(range(eval_interval, total_epochs + 1, eval_interval))
    if num_evals > 0:
        eval_count = min(total_epochs, num_evals)
        eval_points = np.linspace(1, total_epochs, num=eval_count, dtype=int)
        schedule.update(int(point) for point in eval_points.tolist())
    schedule.add(total_epochs)
    return schedule


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_num_interop_threads > 0:
        torch.set_num_interop_threads(args.torch_num_interop_threads)
    device = torch.device(args.device)
    set_seed(args.seed)
    project_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    if args.timestamp is None:
        args.timestamp = generate_timestamp()
    run_token = f"{args.timestamp}_{args.algo}_seed{args.seed}"

    train_worker_idx = None if args.worker_idx is None else args.worker_idx * 2
    eval_worker_idx = None if args.worker_idx is None else args.worker_idx * 2 + 1

    env = make_powergym_train_env(args.env_name, args.observe_load, worker_idx=train_worker_idx, run_token=run_token)
    action_spec = parse_action_space(env.action_space)
    obs_dim = infer_obs_dim(env.observation_space)
    action_dim = len(action_spec.discrete_nvec) + action_spec.continuous_dim
    model = MultiHeadActorCritic(obs_dim=obs_dim, action_spec=action_spec, hidden_sizes=args.hidden_sizes).to(device)

    if args.algo == "ppo":
        updater = PPOUpdater(
            model,
            PPOConfig(
                clip_ratio=args.clip_ratio,
                pi_lr=args.pi_lr,
                vf_lr=args.vf_lr,
                train_pi_iters=args.train_pi_iters,
                train_v_iters=args.train_v_iters,
                target_kl=args.target_kl,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                minibatch_size=args.minibatch_size,
            ),
        )
    elif args.algo == "ppo_lag":
        updater = PPOLagUpdater(
            model,
            PPOLagConfig(
                clip_ratio=args.clip_ratio,
                pi_lr=args.pi_lr,
                vf_lr=args.vf_lr,
                lagrangian_lr=args.lagrangian_lr,
                train_pi_iters=args.train_pi_iters,
                train_v_iters=args.train_v_iters,
                target_kl=args.target_kl,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                minibatch_size=args.minibatch_size,
                cost_limit=args.cost_limit,
                initial_lagrange_multiplier=args.initial_lagrange_multiplier,
            ),
        )
    elif args.algo == "caf_cone":
        updater = CAFConEUpdater(
            model,
            CAFConEConfig(
                clip_ratio=args.clip_ratio,
                pi_lr=args.pi_lr,
                vf_lr=args.vf_lr,
                f_lr=args.f_lr,
                f_target_tau=args.f_target_tau,
                train_pi_iters=args.train_pi_iters,
                train_v_iters=args.train_v_iters,
                train_f_iters=args.train_f_iters,
                target_kl=args.target_kl,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                minibatch_size=args.minibatch_size,
                gamma=args.gamma,
                safety_gamma=args.safety_gamma,
                safety_gae_lambda=args.safety_gae_lambda,
                safety_adv_scale=args.safety_adv_scale,
                beta=args.caf_beta,
                safety_cost_threshold=args.safety_cost_threshold,
                f_hidden_sizes=args.hidden_sizes,
            ),
        )
    elif args.algo == "caf_cone_pareto":
        updater = CAFConEParetoUpdater(
            model,
            CAFConEConfig(
                clip_ratio=args.clip_ratio,
                pi_lr=args.pi_lr,
                vf_lr=args.vf_lr,
                f_lr=args.f_lr,
                f_target_tau=args.f_target_tau,
                train_pi_iters=args.train_pi_iters,
                train_v_iters=args.train_v_iters,
                train_f_iters=args.train_f_iters,
                target_kl=args.target_kl,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                minibatch_size=args.minibatch_size,
                gamma=args.gamma,
                safety_gamma=args.safety_gamma,
                safety_gae_lambda=args.safety_gae_lambda,
                safety_adv_scale=args.safety_adv_scale,
                beta=args.caf_beta,
                safety_cost_threshold=args.safety_cost_threshold,
                f_hidden_sizes=args.hidden_sizes,
            ),
        )
    elif args.algo == "cost_cone_pareto":
        updater = CostConEParetoUpdater(
            model,
            CostConEConfig(
                clip_ratio=args.clip_ratio,
                pi_lr=args.pi_lr,
                vf_lr=args.vf_lr,
                train_pi_iters=args.train_pi_iters,
                train_v_iters=args.train_v_iters,
                target_kl=args.target_kl,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                minibatch_size=args.minibatch_size,
            ),
        )
    elif args.algo == "cost_cone":
        updater = CostConEUpdater(
            model,
            CostConEConfig(
                clip_ratio=args.clip_ratio,
                pi_lr=args.pi_lr,
                vf_lr=args.vf_lr,
                train_pi_iters=args.train_pi_iters,
                train_v_iters=args.train_v_iters,
                target_kl=args.target_kl,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                minibatch_size=args.minibatch_size,
                beta=args.caf_beta,
            ),
        )
    else:
        raise NotImplementedError(f"Unsupported algorithm: {args.algo}")

    run_dir = build_run_dir(str(output_dir), args.timestamp, args.algo, args.seed)
    with (run_dir / "config.json").open("w", encoding="utf-8") as fout:
        json.dump(vars(args), fout, indent=2, ensure_ascii=True)

    logger = CSVLogger(
        run_dir / "metrics.csv",
        [
            "epoch",
            "total_env_steps",
            "train_return",
            "train_cost",
            "train_ep_len",
            "eval_return",
            "eval_cost",
            "eval_ep_len",
            "loss_pi",
            "loss_v",
            "loss_cost_v",
            "entropy",
            "approx_kl",
            "clip_frac",
            "value_ev",
            "cost_value_ev",
            "lagrange_multiplier",
            "loss_f",
            "safety_prob_mean",
        ],
    )

    total_epochs = max(1, math.ceil(args.total_steps / args.steps_per_epoch))
    eval_schedule = build_eval_schedule(total_epochs, args.eval_interval, args.num_evals)
    rng = np.random.default_rng(args.seed)
    obs, _ = reset_with_random_profile(env, rng)
    ep_return = 0.0
    ep_cost = 0.0
    ep_len = 0
    best_eval_return = float("-inf")

    try:
        for epoch in range(1, total_epochs + 1):
            model.train()
            buffer = RolloutBuffer(obs_dim=obs_dim, action_dim=action_dim, size=args.steps_per_epoch, device=device)
            episode_returns: List[float] = []
            episode_costs: List[float] = []
            episode_lengths: List[int] = []

            for step in range(args.steps_per_epoch):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    step_info = model.step(obs_tensor)
                env_action = action_tensor_to_env(step_info["action"].squeeze(0), action_spec)
                next_obs, reward, done, info = env.step(env_action)
                cost = compute_cost(info, args.cost_mode)

                buffer.store(
                    obs=obs,
                    action=step_info["action"],
                    next_obs=next_obs,
                    done=done,
                    reward=reward,
                    cost=cost,
                    value=step_info["value"],
                    cost_value=step_info["cost_value"],
                    log_prob=step_info["log_prob"],
                )

                ep_return += reward
                ep_cost += cost
                ep_len += 1
                obs = next_obs

                if done:
                    zero = torch.zeros(1, dtype=torch.float32, device=device)
                    buffer.finish_path(
                        last_value=zero,
                        last_cost_value=zero,
                        gamma=args.gamma,
                        lam=args.gae_lambda,
                        cost_gamma=args.cost_gamma,
                        cost_lam=args.cost_gae_lambda,
                    )
                    episode_returns.append(ep_return)
                    episode_costs.append(ep_cost)
                    episode_lengths.append(ep_len)
                    obs, _ = reset_with_random_profile(env, rng)
                    ep_return = 0.0
                    ep_cost = 0.0
                    ep_len = 0

            if buffer.path_start_idx != buffer.ptr:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    bootstrap = model.step(obs_tensor, deterministic=True)
                buffer.finish_path(
                    last_value=bootstrap["value"],
                    last_cost_value=bootstrap["cost_value"],
                    gamma=args.gamma,
                    lam=args.gae_lambda,
                    cost_gamma=args.cost_gamma,
                    cost_lam=args.cost_gae_lambda,
                )

            batch = buffer.get()
            if args.algo == "ppo":
                update_metrics = updater.update(batch)
            elif args.algo == "ppo_lag":
                update_metrics = updater.update(batch, mean_episode_cost=safe_mean(episode_costs))
            elif args.algo == "caf_cone":
                update_metrics = updater.update(batch)
            elif args.algo == "caf_cone_pareto":
                update_metrics = updater.update(batch)
            elif args.algo == "cost_cone":
                update_metrics = updater.update(batch)
            elif args.algo == "cost_cone_pareto":
                update_metrics = updater.update(batch)
            else:
                raise NotImplementedError(f"Unsupported algorithm: {args.algo}")

            evaluated = epoch in eval_schedule
            if evaluated:
                eval_metrics = evaluate_policy(
                    model=model,
                    env_name=args.env_name,
                    observe_load=args.observe_load,
                    action_spec=action_spec,
                    device=device,
                    episodes=args.eval_episodes,
                    seed=args.seed + epoch,
                    cost_mode=args.cost_mode,
                    worker_idx=eval_worker_idx,
                    run_token=run_token,
                )
            else:
                eval_metrics = {"eval_return": 0.0, "eval_cost": 0.0, "eval_ep_len": 0.0}

            metrics = {
                "epoch": epoch,
                "total_env_steps": epoch * args.steps_per_epoch,
                "train_return": safe_mean(episode_returns),
                "train_cost": safe_mean(episode_costs),
                "train_ep_len": safe_mean(episode_lengths),
                "eval_return": eval_metrics["eval_return"],
                "eval_cost": eval_metrics["eval_cost"],
                "eval_ep_len": eval_metrics["eval_ep_len"],
                "loss_pi": update_metrics.get("loss_pi", 0.0),
                "loss_v": update_metrics.get("loss_v", 0.0),
                "loss_cost_v": update_metrics.get("loss_cost_v", 0.0),
                "entropy": update_metrics.get("entropy", 0.0),
                "approx_kl": update_metrics.get("approx_kl", 0.0),
                "clip_frac": update_metrics.get("clip_frac", 0.0),
                "value_ev": update_metrics.get("value_ev", 0.0),
                "cost_value_ev": update_metrics.get("cost_value_ev", 0.0),
                "lagrange_multiplier": update_metrics.get("lagrange_multiplier", 0.0),
                "loss_f": update_metrics.get("loss_f", 0.0),
                "safety_prob_mean": update_metrics.get("safety_prob_mean", 0.0),
            }
            logger.log(metrics)
            save_checkpoint(run_dir / "latest.pt", model, args, metrics)
            if evaluated and metrics["eval_return"] >= best_eval_return:
                best_eval_return = metrics["eval_return"]
                save_checkpoint(run_dir / "best_reward.pt", model, args, metrics)

            print(
                "epoch={epoch} steps={total_env_steps} train_return={train_return:.3f} train_cost={train_cost:.3f} "
                "eval_return={eval_return:.3f} eval_cost={eval_cost:.3f} loss_pi={loss_pi:.4f} "
                "loss_v={loss_v:.4f} lambda={lagrange_multiplier:.4f}".format(**metrics)
            )
    finally:
        logger.close()
        if not args.keep_runtime_artifacts:
            remove_runtime_artifacts(args.env_name, run_token)


if __name__ == "__main__":
    main()
