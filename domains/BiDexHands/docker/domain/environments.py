import os
from typing import Any, Dict, Optional, Tuple, List

from gymnasium import Env
from gymnasium.spaces import Box
import numpy as np
from argparse import Namespace
from isaacgym.gymapi import SIM_PHYSX
from bidexhands.utils.config import parse_sim_params, load_cfg
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex
from bidexhands.tasks.hand_base.base_task import BaseTask
import bidexhands
import torch

TASKS = {
    'shadowhandcatchunderarm': 'ShadowHandCatchUnderarm',
    'shadowhandtwocatchunderarm': 'ShadowHandTwoCatchUnderarm',
    'shadowhandcatchabreast': 'ShadowHandCatchAbreast',
    'shadowhandliftunderarm': 'ShadowHandLiftUnderarm',
    'shadowhandcatchover2underarm': 'ShadowHandCatchOver2Underarm',
    'shadowhanddoorcloseinward': 'ShadowHandDoorCloseInward',
    'shadowhanddoorcloseoutward': 'ShadowHandDoorCloseOutward',
    'shadowhanddooropeninward': 'ShadowHandDoorOpenInward',
    'shadowhanddooropenoutward': 'ShadowHandDoorOpenOutward',
    'shadowhandbottlecap': 'ShadowHandBottleCap',
    'shadowhandpushblock': 'ShadowHandPushBlock',
    'shadowhandswingcup': 'ShadowHandSwingCup',
    'shadowhandgraspandplace': 'ShadowHandGraspAndPlace',
    'shadowhandscissors': 'ShadowHandScissors',
    'shadowhandswitch': 'ShadowHandSwitch',
    'shadowhandpen': 'ShadowHandPen',
    'shadowhandkettle': 'ShadowHandKettle',
    'shadowhandblockstack': 'ShadowHandBlockStack'
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def get_args(env_name: str) -> Namespace:
    cfg = 'cfg/ppo/config.yaml'
    if 'Lift' in env_name:
        cfg = 'cfg/ppo/lift_config.yaml'
    # elif 'ReOrientation' in env_name:
    #     cfg = 'cfg/ppo/re_orientation_config.yaml'
    elif 'BlockStack' in env_name:
        cfg = 'cfg/ppo/stack_block_config.yaml'
    return Namespace(
        algo='ppo', cfg_env=f'cfg/{env_name}.yaml', cfg_train=cfg, 
        checkpoint='Base', compute_device_id=0, datatype='random', device='cpu', 
        device_id=0, episode_length=0, experiment='Base', 
        f='', flex=False, graphics_device_id=0, headless=True, horovod=False, 
        logdir=f'logs/{env_name}/ppo/ppo', max_iterations=-1, metadata=False, 
        minibatch_size=-1, model_dir='', num_envs=1, num_threads=0, 
        physics_engine=SIM_PHYSX, physx=False, pipeline='CPU', play=False, 
        randomize=False, resume=0, rl_device=get_device(), seed=None, 
        sim_device='cpu', sim_device_type='cpu', slices=0, steps_num=-1, 
        subscenes=0, task=env_name, task_type='Python', 
        test=False, torch_deterministic=False, train=True, use_gpu=False, 
        use_gpu_pipeline=False
    )


class BiDexGymEnv(Env):
    """Gym wrapper for Bi-DexHands

    Args:
        env: bidexhands task instance
    """

    def __init__(self, env: BaseTask):
        self.env = env
        self.closed = False
        assert self.env.num_agents == 1
        assert self.env.num_envs == 1
        self.action_space = Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            shape=self.env.action_space.shape,
            dtype=self.env.action_space.dtype
        )
        self.observation_space = Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset Environment

        Args:
            seed: optional seed for reproducibility
            options: placeholder to follow gym API

        Returns:
            Tuple[np.ndarray, Any]: observation and info tuple
        """
        super().reset(seed=seed)
        obs = self.env.reset()
        obs = obs.squeeze().detach().cpu().numpy()
        return obs, {}

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Environment Step

        Environment Step

        Args:
            action: numpy array with selected action

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: MDP
                timestamp
        """
        actions = torch.from_numpy(action.astype(np.float32))
        actions = actions.unsqueeze(0)
        observations, rewards, terminations, info = self.env.step(
            actions=actions)
        obs = observations.squeeze().detach().cpu().numpy()
        rews = rewards.squeeze().detach().cpu().numpy().item()
        terminals = bool(terminations.squeeze().detach().cpu().numpy().item())
        return obs, rews, terminals, False, info

    def close(self) -> None:
        """Close Environment

        Destroy IsaacGym simulation as it handles only one
        simulation per Python process
        """
        if not self.closed:
            self.env.task.gym.destroy_sim(self.env.task.sim)
            self.closed = True


def get_env_names() -> List[str]:
    """Get list of environments names
    
    Args:
        None
        
    Returns:
        List[str]: list of environments names (keys) for 
            this domain
    """
    return list(TASKS.keys())


def get_domain_name() -> str:
    """Get the name of the domain
    
    Args:
        None
            
    Returns:
        str: Name of the domain
    """
    return "bidexhands"


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
    # Locate bidexhands installation
    bidexhands_path = os.path.dirname(bidexhands.__file__)
    # Get arguments for CPU installation
    args = get_args(TASKS[env_name])
    if seed is not None:
        args.seed = seed
    # Retrieve environment
    prev_cwd = os.getcwd()
    os.chdir(bidexhands_path)
    cfg, cfg_train, _ = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    agent_index = get_AgentIndex(cfg)
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
    os.chdir(prev_cwd)
    return BiDexGymEnv(env)