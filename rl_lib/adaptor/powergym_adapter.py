from typing import Dict, List, Tuple

import numpy as np
import torch

from powergym.env_register import make_env
from rl_lib.common.spaces import ActionSpec
from rl_lib.common.utils import safe_mean


def make_powergym_train_env(env_name: str, observe_load: bool, worker_idx=None, run_token=None):
    env = make_env(env_name, worker_idx=worker_idx, run_token=run_token)
    env.reset_obs_space(wrap_observation=True, observe_load=observe_load)
    return env


def reset_with_random_profile(env, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    profile_idx = int(rng.integers(env.num_profiles))
    obs = env.reset(load_profile_idx=profile_idx)
    return obs, profile_idx


def action_tensor_to_env(action: torch.Tensor, action_spec: ActionSpec) -> np.ndarray:
    action_np = action.detach().cpu().numpy().reshape(-1)
    if action_spec.has_discrete and not action_spec.has_continuous:
        return action_np.astype(np.int64)
    if action_spec.has_discrete:
        discrete_dim = len(action_spec.discrete_nvec)
        discrete_part = np.rint(action_np[:discrete_dim]).astype(np.int64)
        continuous_part = action_np[discrete_dim:].astype(np.float32)
        return np.concatenate((discrete_part.astype(np.float32), continuous_part), axis=0)
    return action_np.astype(np.float32)


def compute_cost(info: Dict[str, float], mode: str) -> float:
    if mode == "voltage":
        return float(max(0.0, -info.get("vol_reward", 0.0)))
    if mode == "control":
        return float(max(0.0, -info.get("ctrl_reward", 0.0)))
    if mode == "voltage_control":
        return float(max(0.0, -info.get("vol_reward", 0.0)) + max(0.0, -info.get("ctrl_reward", 0.0)))
    raise ValueError(f"Unsupported cost mode: {mode}")


def evaluate_policy(
    model,
    env_name: str,
    observe_load: bool,
    action_spec: ActionSpec,
    device: torch.device,
    episodes: int,
    seed: int,
    cost_mode: str,
    worker_idx=None,
    run_token=None,
) -> Dict[str, float]:
    eval_env = make_powergym_train_env(env_name, observe_load, worker_idx=worker_idx, run_token=run_token)
    rng = np.random.default_rng(seed)
    returns: List[float] = []
    costs: List[float] = []
    lengths: List[int] = []

    model.eval()
    for _ in range(episodes):
        obs, _ = reset_with_random_profile(eval_env, rng)
        ep_return = 0.0
        ep_cost = 0.0
        ep_len = 0
        done = False
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                step_info = model.step(obs_tensor, deterministic=True)
            env_action = action_tensor_to_env(step_info["action"].squeeze(0), action_spec)
            obs, reward, done, info = eval_env.step(env_action)
            ep_return += reward
            ep_cost += compute_cost(info, cost_mode)
            ep_len += 1
        returns.append(ep_return)
        costs.append(ep_cost)
        lengths.append(ep_len)

    return {
        "eval_return": safe_mean(returns),
        "eval_cost": safe_mean(costs),
        "eval_ep_len": safe_mean(lengths),
    }
