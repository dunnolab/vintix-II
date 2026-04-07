from typing import Optional, List
from gymnasium import Env

import gymnasium
import numpy as np

import controlgym
from controlgym.envs import LinearControlEnv

AIRPLANE_ENVS = [f"aircraft_{env_id}" for env_id in range(1, 11)]
CABLE_ENVS = [f"cable_{env_id}" for env_id in range(1, 6)]
PDE_ENVS = [
    "pde_convection_diffusion_reaction",
    "pde_wave",
    "pde_schrodinger",
]


class ActionPlusDisturbanceWrapper(gymnasium.Wrapper):
    def __init__(self, env: LinearControlEnv):
        assert isinstance(
            env, LinearControlEnv), "ActionPlusDisturbanceWrapper only works with LinearControlEnv"
        assert env.action_space.shape is not None and len(
            env.action_space.shape) == 1, "Action space must be 1-dimensional"
        super().__init__(env)

        self._disturbance_size = env.n_disturbance
        self._action_size = env.action_space.shape[0]

        self.action_space = gymnasium.spaces.Box(
            low=np.hstack([env.action_space.low, np.array(
                [env.action_space.low[0]] * self._disturbance_size)]),
            high=np.hstack([env.action_space.high, np.array(
                [env.action_space.high[0]] * self._disturbance_size)]),
            shape=(self._action_size + self._disturbance_size,),
            dtype=env.action_space.dtype,
            seed=env.action_space._np_random,
        )

    def __getattr__(self, name: str):
        if name in ['A', 'B1', 'B2', 'C', 'C1', 'D11', 'D12', 'D21', 'Q', 'R', 'S', 'n_steps', 'sensor_noise_cov', 'category', 'state', 'target_state', 'noise_cov', 'process_noise_cov', 'n_state', 'n_observation', 'n_action']:
            return getattr(self.env, name)

        return super().__getattr__(name)

    def step(self, action):
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
        if action.shape[0] == self._action_size:
            return self.env.step(action=action)
        elif action.shape[0] == self._action_size + self._disturbance_size:
            return self.env.step(action=action[:self._action_size], disturbance=action[self._action_size:])
        raise ValueError(f"Invalid action shape: {action.shape}")


def get_env_names() -> List[str]:
    """Get list of environments names

    Args:
        None

    Returns:
        List[str]: list of environments names (keys) for 
            this domain
    """
    return (AIRPLANE_ENVS + CABLE_ENVS + PDE_ENVS)


def get_domain_name() -> str:
    """Get the name of the domain

    Args:
        None

    Returns:
        str: Name of the domain
    """
    return "controlgym"


def is_single_process() -> bool:
    """Verify if only one environment instance may be 
    sampled per process

    Args:
        None

    Returns:
        bool: True if only one environment instance may be 
            sampled per process, False othervise
    """
    return False


def get_env(env_name: str, seed: Optional[int] = None) -> Env:
    """Factory method for domain environment

    Factory method for domain environment

    Args:
        env_name: environment name string from `get_env_names`
            function
        seed: seed for the environment

    Returns:
        Env: instance of gymnasium-like environment
    """
    pde_params = {}
    if env_name in PDE_ENVS:
        # Here we can modify the parameters for PDE environments (there are default values for PDEs: {n_observation: 10, n_action: 8, n_state: 200 | 256})
        pde_params = {"n_action": 1}

    if env_name.startswith("aircraft_"):
        env_name = env_name.replace("aircraft_", "ac")
    elif env_name.startswith("cable_"):
        env_name = env_name.replace("cable_", "cm")
    elif env_name.startswith("pde_"):
        env_name = env_name.replace("pde_", "")

    physics_env = controlgym.make(
        env_name, observation_limit=1000, action_limit=10, **pde_params)

    assert isinstance(physics_env, Env), "Invalid environment name"

    if isinstance(physics_env, LinearControlEnv):
        physics_env = ActionPlusDisturbanceWrapper(physics_env)

    physics_env.reset(seed=seed)

    return physics_env
