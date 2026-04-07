from typing import Optional, List
from gymnasium import Env
import gymnasium as gym

from domain.sinergym.utils.wrappers import NormalizeObservation, NormalizeAction, ClipActionWrapper

WEATHER_TIMES = [
    "winter",
    "spring",
    "summer",
    "autumn"
]

BUILDINGS = [
    "5zone",
    "warehouse"
]

ENV_NAMES = [
    '5zone-hot_dec_jan_feb',
    '5zone-hot_mar_apr_may',
    '5zone-hot_jun_jul_aug',
    '5zone-hot_sep',
    '5zone-hot_oct_nov',
    '5zone-mixed_dec_jan_feb',
    '5zone-mixed_mar_apr_may',
    '5zone-mixed_jun_jul_aug',
    '5zone-mixed_sep',
    '5zone-mixed_oct_nov',
    '5zone-cool_dec_jan_feb',
    '5zone-cool_mar_apr_may',
    '5zone-cool_jun_jul_aug',
    '5zone-cool_sep',
    '5zone-cool_oct_nov',
    'warehouse-hot_dec_jan_feb',
    'warehouse-hot_mar',
    'warehouse-hot_apr_may',
    'warehouse-hot_jun_jul_aug',
    'warehouse-hot_sep',
    'warehouse-hot_oct',
    'warehouse-hot_nov',
    'warehouse-mixed_dec_jan_feb',
    'warehouse-mixed_mar',
    'warehouse-mixed_apr_may',
    'warehouse-mixed_jun',
    'warehouse-mixed_jul_aug',
    'warehouse-mixed_sep',
    # 'warehouse-mixed_oct_nov',
    'warehouse-cool_dec_jan_feb',
    'warehouse-cool_mar',
    'warehouse-cool_apr_may',
    # 'warehouse-cool_jun',
    # 'warehouse-cool_jul',
    # 'warehouse-cool_aug',
    # 'warehouse-cool_sep',
    'warehouse-cool_oct',
    'warehouse-cool_nov',
]


def get_env_names() -> List[str]:
    """Get list of environments names

    Args:
        None

    Returns:
        List[str]: list of environments names (keys) for 
            this domain
    """
    return ENV_NAMES
    # envs = [n[6:-25] for n in domain.sinergym.__ids__ if "continuous-stochastic-v1" in n]
    # selected_buildings = [e for e in envs if e.split("-")[0] in BUILDINGS]
    # return [e + '_' + wt for e in selected_buildings for wt in WEATHER_TIMES]


def get_domain_name() -> str:
    """Get the name of the domain

    Args:
        None

    Returns:
        str: Name of the domain
    """
    return "sinergym"


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


def get_env(env_name: str, seed: Optional[int] = None, **kwargs) -> Env:
    """Factory method for domain environment

    Factory method for domain environment

    Args:
        env_name: environment name string from `get_env_names`
            function
        seed: seed for the environment

    Returns:
        Env: instance of gymnasium-like environment
    """
    args = env_name.split("_")
    building_weather = args[0]
    months = args[1:]

    name = "Eplus-" + building_weather + "-continuous-stochastic-v1"

    env = gym.make(name, seed=seed, **
                   dict({"config_params": {"months": months}}, **kwargs))
    env = NormalizeObservation(env)
    env = NormalizeAction(env)
    env = ClipActionWrapper(env)

    return env
