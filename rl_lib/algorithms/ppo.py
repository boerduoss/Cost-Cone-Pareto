from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn

from rl_lib.common.utils import explained_variance
from rl_lib.networks.actor_critic import MultiHeadActorCritic


@dataclass
class PPOConfig:
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_pi_iters: int = 80
    train_v_iters: int = 80
    target_kl: float = 0.02
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    minibatch_size: int = 256


class PPOUpdater:
    def __init__(self, model: MultiHeadActorCritic, config: PPOConfig):
        self.model = model
        self.config = config
        self.pi_optimizer = torch.optim.Adam(self.model.parameters(), lr=config.pi_lr)
        self.v_optimizer = torch.optim.Adam(
            list(self.model.feature_extractor.parameters()) + list(self.model.value_head.parameters()),
            lr=config.vf_lr,
        )

    def update(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = data["obs"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        advantages = data["advantages"]
        returns = data["returns"]
        old_values = data["values"]

        batch_size = obs.shape[0]
        minibatch_size = min(self.config.minibatch_size, batch_size)
        last_pi_loss = 0.0
        last_v_loss = 0.0
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
                log_ratio = dist_info["log_prob"] - old_log_probs[idx]
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                surrogate = torch.min(ratio * advantages[idx], clipped_ratio * advantages[idx])
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
                values = self.model.value(obs[idx])
                value_pred_clipped = old_values[idx] + torch.clamp(
                    values - old_values[idx],
                    -self.config.clip_ratio,
                    self.config.clip_ratio,
                )
                value_losses = (values - returns[idx]).pow(2)
                value_losses_clipped = (value_pred_clipped - returns[idx]).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                self.v_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.v_optimizer.step()
                last_v_loss = float(value_loss.item())

        with torch.no_grad():
            final_values = self.model.value(obs)

        return {
            "loss_pi": last_pi_loss,
            "loss_v": last_v_loss,
            "entropy": last_entropy,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "value_ev": explained_variance(final_values, returns),
        }
