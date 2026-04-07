from typing import Optional, List
from gymnasium import Env

from domain.utils import DATA_PATH, ALL_TASKS, EnvWrapper, CustomReward, CityLearnEnv


def get_env_names() -> List[str]:
    """Get list of environments names

    Args:
        None

    Returns:
        List[str]: list of environments names (keys) for
            this domain
    """
    return list(ALL_TASKS.keys())


def get_domain_name() -> str:
    """Get the name of the domain

    Args:
        None

    Returns:
        str: Name of the domain
    """
    return "citylearn"


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
    task = ALL_TASKS[env_name]

    name = task.env_name
    root_path = DATA_PATH / name
    schema_path = root_path / "schema.json"

    citylearn_env = CityLearnEnv(
        schema=schema_path,
        root_directory=root_path,
        central_agent=True,
        reward_function=CustomReward,
        simulation_start_time_step=task.start_time_step,
        simulation_end_time_step=task.end_time_step,
        random_seed=seed,
    )

    return EnvWrapper(env_name=env_name, env=citylearn_env)
