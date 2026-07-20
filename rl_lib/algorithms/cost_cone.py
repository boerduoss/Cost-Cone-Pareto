from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn

from rl_lib.algorithms.caf_cone import _clipped_objective, _flat_from_grads, _set_flat_grad, _zero_fill_grads, grad_crcpo, grad_pareto
from rl_lib.common.utils import explained_variance
from rl_lib.networks.actor_critic import MultiHeadActorCritic


@dataclass
class CostConEConfig:
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_pi_iters: int = 80
    train_v_iters: int = 80
    target_kl: float = 0.02
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    minibatch_size: int = 256
    beta: float = 0.5
    policy_gradient_mode: str = "crcpo"


class CostConEUpdater:
    def __init__(self, model: MultiHeadActorCritic, config: CostConEConfig):
        self.model = model
        self.config = config
        self.actor_parameters = list(self.model.feature_extractor.parameters())
        self.actor_parameters.extend(self.model.discrete_heads.parameters())
        if self.model.continuous_head is not None:
            self.actor_parameters.extend(self.model.continuous_head.parameters())

        self.pi_optimizer = torch.optim.Adam(self.actor_parameters, lr=config.pi_lr)
        self.v_optimizer = torch.optim.Adam(
            list(self.model.feature_extractor.parameters())
            + list(self.model.value_head.parameters())
            + list(self.model.cost_value_head.parameters()),
            lr=config.vf_lr,
        )

    def update(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
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
                current_log_probs = dist_info["log_prob"]
                entropy = dist_info["entropy"].mean()

                reward_objective = _clipped_objective(
                    new_log_probs=current_log_probs,
                    old_log_probs=old_log_probs[idx],
                    advantages=reward_advantages[idx],
                    clip_ratio=self.config.clip_ratio,
                )
                reward_objective = reward_objective + self.config.entropy_coef * entropy

                # Negate the cost surrogate so the second objective becomes
                # "maximize safety / minimize cost".
                safety_objective = -_clipped_objective(
                    new_log_probs=current_log_probs,
                    old_log_probs=old_log_probs[idx],
                    advantages=cost_advantages[idx],
                    clip_ratio=self.config.clip_ratio,
                )

                reward_grads = torch.autograd.grad(
                    reward_objective,
                    self.actor_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                safety_grads = torch.autograd.grad(
                    safety_objective,
                    self.actor_parameters,
                    retain_graph=False,
                    allow_unused=True,
                )
                reward_grads = _zero_fill_grads(reward_grads, self.actor_parameters)
                safety_grads = _zero_fill_grads(safety_grads, self.actor_parameters)
                reward_direction = _flat_from_grads(reward_grads).detach()
                safety_direction = _flat_from_grads(safety_grads).detach()

                if self.config.policy_gradient_mode == "crcpo":
                    ascent_direction = grad_crcpo(
                        reward_direction,
                        safety_direction,
                        beta=self.config.beta,
                    )
                elif self.config.policy_gradient_mode == "pareto":
                    ascent_direction = grad_pareto(reward_direction, safety_direction)
                else:
                    raise ValueError(f"Unknown policy_gradient_mode: {self.config.policy_gradient_mode}")

                self.pi_optimizer.zero_grad()
                _set_flat_grad(self.actor_parameters, -ascent_direction)
                nn.utils.clip_grad_norm_(self.actor_parameters, self.config.max_grad_norm)
                self.pi_optimizer.step()

                ratio = torch.exp(current_log_probs - old_log_probs[idx])
                epoch_kls.append((old_log_probs[idx] - current_log_probs).mean().item())
                epoch_clipfracs.append(
                    ((ratio > 1 + self.config.clip_ratio) | (ratio < 1 - self.config.clip_ratio))
                    .float()
                    .mean()
                    .item()
                )
                last_pi_loss = float(-reward_objective.item())
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

                self.v_optimizer.zero_grad()
                (reward_value_loss + cost_value_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.v_optimizer.step()

                last_reward_v_loss = float(reward_value_loss.item())
                last_cost_v_loss = float(cost_value_loss.item())

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
            "value_ev": explained_variance(final_reward_values, reward_returns),
            "cost_value_ev": explained_variance(final_cost_values, cost_returns),
            "safety_prob_mean": 0.0,
        }


class CostConEParetoUpdater(CostConEUpdater):
    def __init__(self, model: MultiHeadActorCritic, config: CostConEConfig):
        pareto_config = CostConEConfig(**{**config.__dict__, "policy_gradient_mode": "pareto"})
        super().__init__(model=model, config=pareto_config)
