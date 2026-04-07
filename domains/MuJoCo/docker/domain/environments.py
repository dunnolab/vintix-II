from typing import Optional, List
from gymnasium import Env
import gymnasium as gym
from copy import copy


ENV_NAMES = [
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "Humanoid-v4",
    "HumanoidStandup-v4",
    "InvertedDoublePendulum-v4",
    "InvertedPendulum-v4",
    "Pusher-v4",
    "Reacher-v4",
    "Swimmer-v4",
    "Walker2d-v4"
]


def get_env_names() -> List[str]:
    """Get list of environments names
    
    Args:
        None
        
    Returns:
        List[str]: list of environments names (keys) for 
            this domain
    """
    return copy(ENV_NAMES)

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
    assert env_name in ENV_NAMES
    return gym.make(env_name)