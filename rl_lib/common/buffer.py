from typing import Dict

import numpy as np
import torch


def discounted_cumsum(x: torch.Tensor, discount: float) -> torch.Tensor:
    result = torch.zeros_like(x)
    running = torch.zeros((), dtype=x.dtype, device=x.device)
    for i in reversed(range(len(x))):
        running = x[i] + discount * running
        result[i] = running
    return result


class RolloutBuffer:
    def __init__(self, obs_dim: int, action_dim: int, size: int, device: torch.device):
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.costs = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.cost_values = torch.zeros(size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.cost_advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(size, dtype=torch.float32, device=device)
        self.cost_returns = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size
        self.device = device

    def store(
        self,
        obs: np.ndarray,
        action: torch.Tensor,
        next_obs: np.ndarray,
        done: bool,
        reward: float,
        cost: float,
        value: torch.Tensor,
        cost_value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        if self.ptr >= self.max_size:
            raise RuntimeError("RolloutBuffer overflow.")
        self.obs[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = action.squeeze(0)
        self.next_obs[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = float(done)
        self.rewards[self.ptr] = reward
        self.costs[self.ptr] = cost
        self.values[self.ptr] = value.squeeze(0)
        self.cost_values[self.ptr] = cost_value.squeeze(0)
        self.log_probs[self.ptr] = log_prob.squeeze(0)
        self.ptr += 1

    def finish_path(
        self,
        last_value: torch.Tensor,
        last_cost_value: torch.Tensor,
        gamma: float,
        lam: float,
        cost_gamma: float,
        cost_lam: float,
    ) -> None:
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = torch.cat((self.rewards[path_slice], last_value.reshape(1)))
        values = torch.cat((self.values[path_slice], last_value.reshape(1)))
        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = discounted_cumsum(deltas, gamma * lam)
        self.returns[path_slice] = discounted_cumsum(rewards, gamma)[:-1]

        costs = torch.cat((self.costs[path_slice], last_cost_value.reshape(1)))
        cost_values = torch.cat((self.cost_values[path_slice], last_cost_value.reshape(1)))
        cost_deltas = costs[:-1] + cost_gamma * cost_values[1:] - cost_values[:-1]
        self.cost_advantages[path_slice] = discounted_cumsum(cost_deltas, cost_gamma * cost_lam)
        self.cost_returns[path_slice] = discounted_cumsum(costs, cost_gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self) -> Dict[str, torch.Tensor]:
        if self.ptr != self.max_size:
            raise RuntimeError("RolloutBuffer must be full before sampling.")
        adv_mean, adv_std = self.advantages.mean(), self.advantages.std(unbiased=False)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        cost_adv_mean, cost_adv_std = self.cost_advantages.mean(), self.cost_advantages.std(unbiased=False)
        self.cost_advantages = (self.cost_advantages - cost_adv_mean) / (cost_adv_std + 1e-8)
        data = {
            "obs": self.obs,
            "actions": self.actions,
            "next_obs": self.next_obs,
            "dones": self.dones,
            "costs": self.costs,
            "log_probs": self.log_probs,
            "advantages": self.advantages,
            "cost_advantages": self.cost_advantages,
            "returns": self.returns,
            "cost_returns": self.cost_returns,
            "values": self.values,
            "cost_values": self.cost_values,
        }
        self.ptr = 0
        self.path_start_idx = 0
        return data
