import math
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import gym
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def generate_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_run_dir(base_dir: str, timestamp: str, algo_name: str, seed: int) -> Path:
    run_dir = Path(base_dir) / timestamp / algo_name / f"seed{seed}"
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def infer_timestamp_from_path(path: Path, marker: str) -> Optional[str]:
    parts = list(path.resolve().parts)
    if marker not in parts:
        return None
    idx = parts.index(marker)
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


def infer_run_metadata_from_checkpoint(checkpoint_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parts = list(checkpoint_path.resolve().parts)
    if "runs" not in parts:
        return None, None, None
    idx = parts.index("runs")
    if idx + 3 >= len(parts):
        return None, None, None
    return parts[idx + 1], parts[idx + 2], parts[idx + 3]


def explained_variance(preds: torch.Tensor, targets: torch.Tensor) -> float:
    var_y = torch.var(targets)
    if torch.isclose(var_y, torch.tensor(0.0, device=targets.device)):
        return 0.0
    return float(1.0 - torch.var(targets - preds) / var_y)


def infer_obs_dim(observation_space: gym.Space) -> int:
    if not isinstance(observation_space, gym.spaces.Box):
        raise NotImplementedError("Only flattened Box observation spaces are supported.")
    return int(math.prod(observation_space.shape))
