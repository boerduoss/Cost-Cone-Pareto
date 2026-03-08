from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from rl_lib.common.spaces import ActionSpec


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def flatten_action(discrete_action: Optional[torch.Tensor], continuous_action: Optional[torch.Tensor]) -> torch.Tensor:
    pieces = []
    if discrete_action is not None:
        pieces.append(discrete_action.to(torch.float32))
    if continuous_action is not None:
        pieces.append(continuous_action.to(torch.float32))
    if not pieces:
        raise ValueError("Action must have at least one component.")
    return torch.cat(pieces, dim=-1)


class DiagGaussianHead(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Linear(input_dim, action_dim)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std


class MultiHeadActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_spec: ActionSpec, hidden_sizes: List[int]):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer width.")
        self.action_spec = action_spec

        layers: List[nn.Module] = []
        prev_dim = obs_dim
        for hidden_dim in hidden_sizes:
            layers.extend((nn.Linear(prev_dim, hidden_dim), nn.Tanh()))
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        feature_dim = prev_dim

        self.discrete_heads = nn.ModuleList([nn.Linear(feature_dim, n) for n in action_spec.discrete_nvec])
        self.continuous_head = DiagGaussianHead(feature_dim, action_spec.continuous_dim) if action_spec.has_continuous else None
        self.value_head = nn.Linear(feature_dim, 1)
        self.cost_value_head = nn.Linear(feature_dim, 1)

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(obs)

    def _discrete_dist(self, features: torch.Tensor):
        return [Categorical(logits=head(features)) for head in self.discrete_heads]

    def _continuous_dist(self, features: torch.Tensor) -> Optional[Normal]:
        if self.continuous_head is None:
            return None
        mean, log_std = self.continuous_head(features)
        return Normal(mean, log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.extract_features(obs)).squeeze(-1)

    def cost_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.cost_value_head(self.extract_features(obs)).squeeze(-1)

    def step(self, obs: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        features = self.extract_features(obs)
        discrete_action = None
        continuous_action = None
        log_prob = torch.zeros(obs.shape[0], device=obs.device)
        entropy_terms: List[torch.Tensor] = []

        if self.action_spec.has_discrete:
            samples = []
            for dist in self._discrete_dist(features):
                sample = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
                samples.append(sample)
                log_prob = log_prob + dist.log_prob(sample)
                entropy_terms.append(dist.entropy())
            discrete_action = torch.stack(samples, dim=-1)

        if self.action_spec.has_continuous:
            dist = self._continuous_dist(features)
            assert dist is not None
            raw_action = dist.mean if deterministic else dist.rsample()
            continuous_action = torch.tanh(raw_action)
            cont_log_prob = dist.log_prob(raw_action).sum(dim=-1)
            correction = torch.log(torch.clamp(1 - continuous_action.pow(2), min=1e-6)).sum(dim=-1)
            log_prob = log_prob + cont_log_prob - correction
            entropy_terms.append(dist.entropy().sum(dim=-1))

        entropy = torch.stack(entropy_terms, dim=0).sum(dim=0) if entropy_terms else torch.zeros_like(log_prob)
        return {
            "action": flatten_action(discrete_action, continuous_action),
            "log_prob": log_prob,
            "entropy": entropy,
            "value": self.value_head(features).squeeze(-1),
            "cost_value": self.cost_value_head(features).squeeze(-1),
        }

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.extract_features(obs)
        log_prob = torch.zeros(obs.shape[0], device=obs.device)
        entropy_terms: List[torch.Tensor] = []
        offset = 0

        if self.action_spec.has_discrete:
            discrete_actions = actions[:, : len(self.action_spec.discrete_nvec)].long()
            offset = len(self.action_spec.discrete_nvec)
            for dim, dist in enumerate(self._discrete_dist(features)):
                current_action = discrete_actions[:, dim]
                log_prob = log_prob + dist.log_prob(current_action)
                entropy_terms.append(dist.entropy())

        if self.action_spec.has_continuous:
            continuous_actions = torch.clamp(actions[:, offset:], -0.999999, 0.999999)
            raw_actions = torch.atanh(continuous_actions)
            dist = self._continuous_dist(features)
            assert dist is not None
            cont_log_prob = dist.log_prob(raw_actions).sum(dim=-1)
            correction = torch.log(torch.clamp(1 - continuous_actions.pow(2), min=1e-6)).sum(dim=-1)
            log_prob = log_prob + cont_log_prob - correction
            entropy_terms.append(dist.entropy().sum(dim=-1))

        entropy = torch.stack(entropy_terms, dim=0).sum(dim=0) if entropy_terms else torch.zeros_like(log_prob)
        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": self.value_head(features).squeeze(-1),
            "cost_value": self.cost_value_head(features).squeeze(-1),
        }
