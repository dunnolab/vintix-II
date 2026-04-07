from typing import Optional, List
from gymnasium import Env
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
import gymnasium as gym

expert_obs_cfg = dict(
    lidar=dict(num_lasers=240, distance=50, num_others=4,
               gaussian_noise=0.0, dropout_prob=0.0)
)


class ObsCorrectionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        obs[15] = 1 - obs[15]
        obs[10] = 1 - obs[10]
        return obs

    def reset(self, **kwargs):
        kwargs.pop('options', None)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    @property
    def agent(self):
        return self.env.agent


def create_env(env_config):
    def _init():
        env = MetaDriveEnv(env_config)
        env = ObsCorrectionWrapper(env)
        return env
    return _init()


def get_env_names() -> List[str]:
    """Get list of environments names

    Args:
        None

    Returns:
        List[str]: list of environments names (keys) for
            this domain
    """
    return [
        "def_S",
        'def_C_69',
        'def_C_859',
        'def_C_574',
        'def_C_85',
        'def_C_785',
        'def_C_520',
        'def_O_785',
        'def_O_291',
        'def_O_647',
        'def_O_971',
        'def_T_971',
        'def_T_855',
        'def_T_98',
        "cones_S",
        'cones_C_69',
        'cones_C_859',
        'cones_C_574',
        'cones_C_85',
        'cones_C_785',
        'cones_C_520',
    ]


def get_domain_name() -> str:
    """Get the name of the domain

    Args:
        None

    Returns:
        str: Name of the domain
    """
    return "metadrive"


def is_single_process() -> bool:
    """Verify if only one environment instance may be
    sampled per process

    Args:
        None

    Returns:
        bool: True if only one environment instance may be
            sampled per process, False othervise
    """
    return True


def get_config(env_name: str):
    map_config = {BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                  BaseMap.GENERATE_CONFIG: "OOO",
                  BaseMap.LANE_NUM: 3}
    config = dict(
        map_config=map_config,
        num_scenarios=100,
        horizon=1000,
        start_seed=0,
        traffic_density=0,
        accident_prob=0,
        log_level=50,
        vehicle_config=expert_obs_cfg,
        # random_lane_width=True,
        object_manager_seed=47
    )

    if env_name == "def_S":
        map_config['config'] = 'SSSSSS'
    elif env_name == "def_C_69":
        map_config['config'] = 'CCCCC'
        map_config['map_seed'] = 69
    elif env_name == "def_C_859":
        map_config['config'] = 'CCCCC'
        map_config['map_seed'] = 859
    elif env_name == "def_C_574":
        map_config['config'] = 'CCCCC'
        map_config['map_seed'] = 574
    elif env_name == "def_C_85":
        map_config['config'] = 'CCCCC'
        map_config['map_seed'] = 85
    elif env_name == "def_C_785":
        map_config['config'] = 'CCCCC'
        map_config['map_seed'] = 785
    elif env_name == "def_C_520":
        map_config['config'] = 'CCCCC'
        map_config['map_seed'] = 520
    elif env_name == "def_O_785":
        map_config['config'] = 'OOOOO'
        map_config['map_seed'] = 785
    elif env_name == "def_O_291":
        map_config['config'] = 'OOOOO'
        map_config['map_seed'] = 291
    elif env_name == "def_O_647":
        map_config['config'] = 'OOOOO'
        map_config['map_seed'] = 647
    elif env_name == "def_O_971":
        map_config['config'] = 'OOOOO'
        map_config['map_seed'] = 971
    elif env_name == "def_T_971":
        map_config['config'] = 'TTTTTT'
        map_config['map_seed'] = 971
    elif env_name == "def_T_855":
        map_config['config'] = 'TTTTTT'
        map_config['map_seed'] = 855
    elif env_name == "def_T_98":
        map_config['config'] = 'TTTTTT'
        map_config['map_seed'] = 98
    # CONES
    elif env_name == "cones_S":
        map_config['config'] = 'SSSSSS'
        config['accident_prob'] = 1
    elif env_name == "cones_C_69":
        map_config['config'] = 'CCCCC'
        map_config['map_seed'] = 69
        config['accident_prob'] = 1
    elif env_name == "cones_C_859":
        map_config['config'] = 'CCCCC'
        map_config['map_seed'] = 859
        config['accident_prob'] = 1
    elif env_name == "cones_C_574":
        map_config['config'] = 'CCCCC'
        map_config['map_seed'] = 574
        config['accident_prob'] = 1
    elif env_name == "cones_C_85":
        map_config['config'] = 'CCCCC'
        map_config['map_seed'] = 85
        config['accident_prob'] = 1
    elif env_name == "cones_C_785":
        map_config['config'] = 'CCCCC'
        map_config['map_seed'] = 785
        config['accident_prob'] = 1
    elif env_name == "cones_C_520":
        map_config['config'] = 'CCCC'
        map_config['map_seed'] = 520
        config['accident_prob'] = 1
    return config


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
    config = get_config(env_name)
    if seed is not None:
        config['start_seed'] = seed
    env = create_env(config)
    return env
