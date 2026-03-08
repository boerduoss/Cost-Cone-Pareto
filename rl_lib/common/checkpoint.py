from pathlib import Path
from typing import Dict, Tuple

import torch

from rl_lib.common.spaces import parse_action_space
from rl_lib.common.utils import infer_obs_dim
from rl_lib.networks import MultiHeadActorCritic
from rl_lib.adaptor import make_powergym_train_env


def load_checkpoint(path: str, device: torch.device) -> Dict:
    checkpoint_path = Path(path).expanduser().resolve()
    return torch.load(checkpoint_path, map_location=device)


def build_model_from_checkpoint(checkpoint: Dict, device: torch.device, run_token: str | None = None) -> Tuple[MultiHeadActorCritic, Dict]:
    args = checkpoint["args"]
    env = make_powergym_train_env(args["env_name"], args.get("observe_load", False), run_token=run_token)
    action_spec = parse_action_space(env.action_space)
    obs_dim = infer_obs_dim(env.observation_space)
    model = MultiHeadActorCritic(
        obs_dim=obs_dim,
        action_spec=action_spec,
        hidden_sizes=args["hidden_sizes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, args
