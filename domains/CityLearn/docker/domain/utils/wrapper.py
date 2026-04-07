############# USE LOCAL FRAMEWORK #############
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CityLearn"))
##############################################

import numpy as np
import pickle as pk

from gymnasium import Env
from gymnasium.spaces import Space

from pathlib import Path
from citylearn.citylearn import CityLearnEnv
from typing import List, Optional, Dict, Tuple, Any


class EnvWrapper(Env):
    def __init__(self, env_name: str, env: CityLearnEnv):
        self.env = env
        cwd_path = Path(__file__).parent
        obs_space_fn = "phase_1" if "phase_1" in env_name else "phase_2"
        obs_space_fn += "_obs_space.pkl"
        with open(cwd_path / obs_space_fn, "rb") as f:
            self.obs_space = pk.load(f)

    @property
    def action_space(self) -> Space:
        return self.env.action_space[0]

    @property
    def action_names(self) -> List[str]:
        return self.env.action_names[0]

    @property
    def observation_space(self) -> Space:
        return self.obs_space

    @property
    def observation_names(self) -> List[str]:
        return self.env.observation_names[0]

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._clip_obs(obs[0])
        return obs, info

    def step(
        self, action: np.ndarray | list
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step([action])
        obs = self._clip_obs(obs[0])
        return obs, reward[0], terminated, truncated, info

    def _clip_obs(self, obs: np.ndarray) -> np.ndarray:
        obs_space = self.observation_space
        return np.clip(obs, obs_space.low, obs_space.high, dtype=np.float32)

    def __getattr__(self, name):
        return getattr(self.env, name)
