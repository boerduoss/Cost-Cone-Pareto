from dataclasses import dataclass
from typing import List

import gym


@dataclass
class ActionSpec:
    discrete_nvec: List[int]
    continuous_dim: int

    @property
    def has_discrete(self) -> bool:
        return len(self.discrete_nvec) > 0

    @property
    def has_continuous(self) -> bool:
        return self.continuous_dim > 0


def parse_action_space(action_space: gym.Space) -> ActionSpec:
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return ActionSpec(discrete_nvec=action_space.nvec.tolist(), continuous_dim=0)
    if isinstance(action_space, gym.spaces.Tuple):
        if len(action_space.spaces) != 2:
            raise NotImplementedError("Only Tuple(MultiDiscrete, Box) action spaces are supported.")
        discrete_space, continuous_space = action_space.spaces
        if not isinstance(discrete_space, gym.spaces.MultiDiscrete):
            raise NotImplementedError("Tuple action space must start with MultiDiscrete.")
        if not isinstance(continuous_space, gym.spaces.Box):
            raise NotImplementedError("Tuple action space must end with Box.")
        if continuous_space.shape is None or len(continuous_space.shape) != 1:
            raise NotImplementedError("Only 1D continuous Box action spaces are supported.")
        return ActionSpec(
            discrete_nvec=discrete_space.nvec.tolist(),
            continuous_dim=int(continuous_space.shape[0]),
        )
    raise NotImplementedError(f"Unsupported action space: {action_space!r}")
