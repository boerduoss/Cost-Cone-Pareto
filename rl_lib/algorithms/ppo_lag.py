from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn

from rl_lib.common.utils import explained_variance
from rl_lib.networks.actor_critic import MultiHeadActorCritic


@dataclass
class PPOLagConfig:
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    lagrangian_lr: float = 0.05
    train_pi_iters: int = 80
    train_v_iters: int = 80
    target_kl: float = 0.02
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    minibatch_size: int = 256
    cost_limit: float = 0.0
    initial_lagrange_multiplier: float = 1e-3


class PPOLagUpdater:
    def __init__(self, model: MultiHeadActorCritic, config: PPOLagConfig):
        self.model = model
        self.config = config
        self.pi_optimizer = torch.optim.Adam(self.model.parameters(), lr=config.pi_lr)
        self.v_optimizer = torch.optim.Adam(
            list(self.model.feature_extractor.parameters())
            + list(self.model.value_head.parameters())
            + list(self.model.cost_value_head.parameters()),
            lr=config.vf_lr,
        )
        initial_lambda = max(config.initial_lagrange_multiplier, 1e-6)
        self.log_lagrange = nn.Parameter(torch.log(torch.tensor(initial_lambda, dtype=torch.float32)))
        self.lambda_optimizer = torch.optim.Adam([self.log_lagrange], lr=config.lagrangian_lr)

    @property
    def lagrange_multiplier(self) -> torch.Tensor:
        return torch.clamp(self.log_lagrange.exp(), min=0.0, max=1e6)

    def update(self, data: Dict[str, torch.Tensor], mean_episode_cost: float) -> Dict[str, float]:
        obs = data["obs"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        reward_advantages = data["advantages"]
        cost_advantages = data["cost_advantages"]
        reward_returns = data["returns"]
        cost_returns = data["cost_returns"]
        old_values = data["values"]
        old_cost_values = data["cost_values"]

        batch_size = obs.shape[0]
        minibatch_size = min(self.config.minibatch_size, batch_size)
        last_pi_loss = 0.0
        last_reward_v_loss = 0.0
        last_cost_v_loss = 0.0
        last_entropy = 0.0
        approx_kl = 0.0
        clip_frac = 0.0

        for _ in range(self.config.train_pi_iters):
            permutation = torch.randperm(batch_size, device=obs.device)
            epoch_kls: List[float] = []
            epoch_clipfracs: List[float] = []
            for start in range(0, batch_size, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                dist_info = self.model.evaluate_actions(obs[idx], actions[idx])
                ratio = torch.exp(dist_info["log_prob"] - old_log_probs[idx])
                clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                lagrange = self.lagrange_multiplier.detach()
                combined_adv = (reward_advantages[idx] - lagrange * cost_advantages[idx]) / (1.0 + lagrange)
                surrogate = torch.min(ratio * combined_adv, clipped_ratio * combined_adv)
                entropy = dist_info["entropy"].mean()
                loss_pi = -(surrogate.mean() + self.config.entropy_coef * entropy)

                self.pi_optimizer.zero_grad()
                loss_pi.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.pi_optimizer.step()

                epoch_kls.append((old_log_probs[idx] - dist_info["log_prob"]).mean().item())
                epoch_clipfracs.append(((ratio > 1 + self.config.clip_ratio) | (ratio < 1 - self.config.clip_ratio)).float().mean().item())
                last_pi_loss = float(loss_pi.item())
                last_entropy = float(entropy.item())

            approx_kl = sum(epoch_kls) / len(epoch_kls)
            clip_frac = sum(epoch_clipfracs) / len(epoch_clipfracs)
            if approx_kl > 1.5 * self.config.target_kl:
                break

        for _ in range(self.config.train_v_iters):
            permutation = torch.randperm(batch_size, device=obs.device)
            for start in range(0, batch_size, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                eval_info = self.model.evaluate_actions(obs[idx], actions[idx])
                reward_values = eval_info["value"]
                cost_values = eval_info["cost_value"]

                reward_value_pred_clipped = old_values[idx] + torch.clamp(
                    reward_values - old_values[idx],
                    -self.config.clip_ratio,
                    self.config.clip_ratio,
                )
                reward_losses = (reward_values - reward_returns[idx]).pow(2)
                reward_losses_clipped = (reward_value_pred_clipped - reward_returns[idx]).pow(2)
                reward_value_loss = 0.5 * torch.max(reward_losses, reward_losses_clipped).mean()

                cost_value_pred_clipped = old_cost_values[idx] + torch.clamp(
                    cost_values - old_cost_values[idx],
                    -self.config.clip_ratio,
                    self.config.clip_ratio,
                )
                cost_losses = (cost_values - cost_returns[idx]).pow(2)
                cost_losses_clipped = (cost_value_pred_clipped - cost_returns[idx]).pow(2)
                cost_value_loss = 0.5 * torch.max(cost_losses, cost_losses_clipped).mean()
                total_value_loss = reward_value_loss + cost_value_loss

                self.v_optimizer.zero_grad()
                total_value_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.v_optimizer.step()

                last_reward_v_loss = float(reward_value_loss.item())
                last_cost_v_loss = float(cost_value_loss.item())

        lagrange = self.lagrange_multiplier
        cost_gap = torch.tensor(mean_episode_cost - self.config.cost_limit, dtype=torch.float32, device=lagrange.device)
        lambda_loss = -(lagrange * cost_gap.detach())
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()

        with torch.no_grad():
            final_reward_values = self.model.value(obs)
            final_cost_values = self.model.cost_value(obs)

        return {
            "loss_pi": last_pi_loss,
            "loss_v": last_reward_v_loss,
            "loss_cost_v": last_cost_v_loss,
            "entropy": last_entropy,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "lagrange_multiplier": float(self.lagrange_multiplier.item()),
            "value_ev": explained_variance(final_reward_values, reward_returns),
            "cost_value_ev": explained_variance(final_cost_values, cost_returns),
            "mean_episode_cost": mean_episode_cost,
        }
