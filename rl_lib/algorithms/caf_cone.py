from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import nn

from rl_lib.common.utils import explained_variance
from rl_lib.networks.actor_critic import MultiHeadActorCritic


def _build_mlp(sizes: Iterable[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    sizes = list(sizes)
    for idx in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[idx], sizes[idx + 1]))
        if idx < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class SafetyCAFNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: List[int]):
        super().__init__()
        self.net = _build_mlp([obs_dim, *hidden_sizes, 1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


@dataclass
class CAFConEConfig:
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    f_lr: float = 3e-4
    f_target_tau: float = 0.05
    train_pi_iters: int = 80
    train_v_iters: int = 80
    train_f_iters: int = 2
    target_kl: float = 0.02
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    minibatch_size: int = 256
    gamma: float = 0.99
    safety_gamma: float = 0.99
    safety_gae_lambda: float = 0.95
    safety_adv_scale: float = 1.0
    beta: float = 0.5
    policy_gradient_mode: str = "crcpo"
    safety_cost_threshold: float = 0.0
    f_hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])


def _flat_from_grads(grads: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([grad.reshape(-1) for grad in grads], dim=0)


def _set_flat_grad(parameters: List[nn.Parameter], flat_grad: torch.Tensor) -> None:
    offset = 0
    for param in parameters:
        numel = param.numel()
        grad = flat_grad[offset : offset + numel].view_as(param).clone()
        if param.grad is None:
            param.grad = grad
        else:
            param.grad.copy_(grad)
        offset += numel
    if offset != flat_grad.numel():
        raise ValueError("Flat gradient length mismatch.")


def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.mul_(1.0 - tau)
            tgt_param.data.add_(tau * src_param.data)


def _zero_fill_grads(grads, parameters: List[nn.Parameter]) -> List[torch.Tensor]:
    filled: List[torch.Tensor] = []
    for grad, param in zip(grads, parameters):
        if grad is None:
            filled.append(torch.zeros_like(param))
        else:
            filled.append(grad)
    return filled


def _clipped_objective(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float,
) -> torch.Tensor:
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    return torch.min(ratio * advantages, clipped_ratio * advantages).mean()


def _compute_safety_advantages(
    violation_costs: torch.Tensor,
    f_values: torch.Tensor,
    f_next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    advantages = torch.zeros_like(violation_costs)
    gae = torch.zeros((), dtype=violation_costs.dtype, device=violation_costs.device)
    for idx in reversed(range(len(violation_costs))):
        if dones[idx] > 0.5:
            target = violation_costs[idx]
            mask = 0.0
        else:
            target = violation_costs[idx] + (1.0 - violation_costs[idx]) * gamma * f_next_values[idx]
            mask = 1.0
        delta = target - f_values[idx]
        gae = delta + gamma * lam * mask * gae
        advantages[idx] = -gae
    return advantages


def grad_crcpo(gr: torch.Tensor, gs: torch.Tensor, beta: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    dot = torch.dot(gr, gs)
    if dot < 0:
        gs_norm2 = torch.dot(gs, gs) + eps
        gr_norm2 = torch.dot(gr, gr) + eps
        gr_plus = gr - dot / gs_norm2 * gs
        gs_plus = gs - dot / gr_norm2 * gr
        return beta * gr_plus + (1.0 - beta) * gs_plus
    return beta * gr


def grad_pareto(gr: torch.Tensor, gs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    gs_norm2 = torch.dot(gs, gs) + eps
    gr_norm2 = torch.dot(gr, gr) + eps
    gr_perp = gr - torch.dot(gr, gs) / gs_norm2 * gs
    gs_perp = gs - torch.dot(gs, gr) / gr_norm2 * gr
    direction = gr_perp - gs_perp
    denom = torch.dot(direction, direction) + eps
    beta_reward = torch.dot(gs_perp, gs_perp - gr_perp) / denom
    beta_reward = torch.clamp(beta_reward, 0.0, 1.0)
    return beta_reward * gr_perp + (1.0 - beta_reward) * gs_perp


class CAFConEUpdater:
    def __init__(self, model: MultiHeadActorCritic, config: CAFConEConfig):
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
        feature_layer = next(iter(self.model.feature_extractor))
        if not isinstance(feature_layer, nn.Linear):
            raise TypeError("Expected first feature extractor layer to be nn.Linear.")
        obs_dim = feature_layer.in_features
        self.f_net = SafetyCAFNet(obs_dim=obs_dim, hidden_sizes=config.f_hidden_sizes).to(next(self.model.parameters()).device)
        self.f_target = SafetyCAFNet(obs_dim=obs_dim, hidden_sizes=config.f_hidden_sizes).to(next(self.model.parameters()).device)
        self.f_target.load_state_dict(self.f_net.state_dict())
        self.f_optimizer = torch.optim.Adam(self.f_net.parameters(), lr=config.f_lr)
        self.f_loss = nn.BCEWithLogitsLoss()

    def update(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = data["obs"]
        actions = data["actions"]
        next_obs = data["next_obs"]
        dones = data["dones"]
        old_log_probs = data["log_probs"]
        reward_advantages = data["advantages"]
        reward_returns = data["returns"]
        cost_returns = data["cost_returns"]
        costs = data["costs"] if "costs" in data else None
        old_values = data["values"]
        old_cost_values = data["cost_values"]

        if costs is None:
            raise KeyError("CAFConEUpdater requires raw costs in rollout data.")

        batch_size = obs.shape[0]
        minibatch_size = min(self.config.minibatch_size, batch_size)
        violation_costs = (costs > self.config.safety_cost_threshold).to(dtype=torch.float32)

        last_f_loss = 0.0
        for _ in range(self.config.train_f_iters):
            permutation = torch.randperm(batch_size, device=obs.device)
            for start in range(0, batch_size, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                current_obs = obs[idx]
                current_next_obs = next_obs[idx]
                current_violation = violation_costs[idx]
                current_done = dones[idx]

                with torch.no_grad():
                    next_probs = torch.sigmoid(self.f_target(current_next_obs))
                    targets = torch.where(
                        current_done > 0.5,
                        current_violation,
                        current_violation + (1.0 - current_violation) * self.config.safety_gamma * next_probs,
                    )
                    targets = torch.clamp(targets, 0.0, 1.0)

                logits = self.f_net(current_obs)
                loss_f = self.f_loss(logits, targets)
                self.f_optimizer.zero_grad()
                loss_f.backward()
                nn.utils.clip_grad_norm_(self.f_net.parameters(), self.config.max_grad_norm)
                self.f_optimizer.step()
                _soft_update(self.f_net, self.f_target, self.config.f_target_tau)
                last_f_loss = float(loss_f.item())

        with torch.no_grad():
            f_values = torch.sigmoid(self.f_net(obs))
            f_next_values = torch.sigmoid(self.f_net(next_obs))
            safety_advantages = _compute_safety_advantages(
                violation_costs=violation_costs,
                f_values=f_values,
                f_next_values=f_next_values,
                dones=dones,
                gamma=self.config.safety_gamma,
                lam=self.config.safety_gae_lambda,
            )
            safety_advantages = (safety_advantages - safety_advantages.mean()) / (safety_advantages.std(unbiased=False) + 1e-8)
            safety_advantages = safety_advantages * self.config.safety_adv_scale

        last_pi_loss = 0.0
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
                safety_objective = _clipped_objective(
                    new_log_probs=current_log_probs,
                    old_log_probs=old_log_probs[idx],
                    advantages=safety_advantages[idx],
                    clip_ratio=self.config.clip_ratio,
                )
                reward_objective = reward_objective + self.config.entropy_coef * entropy

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
                    ascent_direction = grad_pareto(
                        reward_direction,
                        safety_direction,
                    )
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
                last_pi_loss = float(-(reward_objective.item()))
                last_entropy = float(entropy.item())

            approx_kl = sum(epoch_kls) / len(epoch_kls)
            clip_frac = sum(epoch_clipfracs) / len(epoch_clipfracs)
            if approx_kl > 1.5 * self.config.target_kl:
                break

        last_reward_v_loss = 0.0
        last_cost_v_loss = 0.0
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
            "loss_f": last_f_loss,
            "entropy": last_entropy,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "value_ev": explained_variance(final_reward_values, reward_returns),
            "cost_value_ev": explained_variance(final_cost_values, cost_returns),
            "safety_prob_mean": float(f_values.mean().item()),
        }


class CAFConEParetoUpdater(CAFConEUpdater):
    def __init__(self, model: MultiHeadActorCritic, config: CAFConEConfig):
        pareto_config = CAFConEConfig(**{**config.__dict__, "policy_gradient_mode": "pareto"})
        super().__init__(model=model, config=pareto_config)
