from typing import Optional, List
from gymnasium import Env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv
from humenv import make_humenv
from typing import List
from gymnasium.wrappers import FlattenObservation
from humenv.misc.motionlib import MotionBuffer
import gymnasium as gym
from pathlib import Path
import json
import numpy as np
import os


def shorten_motion_name(s: str, x: int):
    return s.split('_')[0]+'_'+str(x)


# base_path = os.path.join(os.getcwd(), "domain")
base_path = os.environ["BASE_HUMENV_PATH"]
pose_file = f"{base_path}/goals.json"
with open(pose_file, "r") as json_file:
    goals = json.load(json_file)
for pose_name, payload in goals.items():
    goals[pose_name] = np.array(payload["observation"])

motions_base_path = Path(f"{base_path}/humenv_amass")


DEFAULT_MAX_EPISODE_STEPS = 300


def get_env_names() -> List[str]:
    """Get list of environments names

    Args:
        None

    Returns:
        List[str]: list of environments names (keys) for 
            this domain
    """
    return ['REWARD~move-ego-0-0', 'REWARD~move-ego-0-2', 'REWARD~raisearms-h-h', 'REWARD~crawl-0_5-0-d', 'REWARD~crouch-0',
            'REWARD~split-0_5', 'REWARD~rotate-x--5-0_8', 'REWARD~rotate-x-5-0_8', 'REWARD~headstand', 'REWARD~move-ego-90-4', 'GOAL~t_pose',
            'GOAL~crouch_medium', 'GOAL~zombie', 'GOAL~sit_hand_behind', 'GOAL~lie_front']


def get_domain_name() -> str:
    """Get the name of the domain

    Args:
        None

    Returns:
        str: Name of the domain
    """
    return "humenv"


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


class RewardEnv(gym.Env):
    def __init__(self, vec_env):
        assert vec_env.num_envs == 1, "Only supports num_envs=1"
        self.vec_env = vec_env
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space

    def reset(self, **kwargs):
        obs = self.vec_env.reset()
        return obs[0], {}  # shape: (obs_dim,)

    def step(self, action):
        obs, reward, done, info = self.vec_env.step([action])
        return obs[0], reward[0], done[0], False, info[0]

    def render(self):
        return self.vec_env.render()

    def close(self):
        self.vec_env.close()


class FixedGoalEnv(gym.Wrapper):
    def __init__(self, env, goal=None):
        super().__init__(env)
        self.env = env
        self.goal = goal

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        info['achieved_goal'] = obs.copy()
        info['desired_goal'] = self.goal
        return obs, self.compute_reward(obs, self.goal, info), terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = - \
            np.linalg.norm(achieved_goal[..., :214] -
                           desired_goal[..., :214], axis=-1)
        return reward


MOTION_BUF = None


def get_env(env_name: str, seed: Optional[int] = None, path=f'{base_path}/demonstrators_artifacts') -> Env:
    """Factory method for domain environment

    Factory method for domain environment

    Args:
        env_name: environment name string from `get_env_names`
            function
        seed: seed for the environment

    Returns:
        Env: instance of gymnasium-like environment
    """
    global MOTION_BUF

    task_type, task_name = env_name.split('~')
    if MOTION_BUF is None:
        if motions_base_path.exists():
            with open(os.path.join(base_path, 'collected_hdf5_files.txt'), 'r') as f:
                motions = sorted([s.strip() for s in f.readlines()])
                print("MOTIONS: ", motions)
                del motions[530]
        MOTION_BUF = MotionBuffer(motions, motions_base_path)

    kwargs = {'wrappers': [
        FlattenObservation,
    ],
        'state_init': "MoCapAndFall",
        'fall_prob': 0.2,
        # 'motion_base_path':motions_base_path,
        # 'motions':motions,
        'motions': MOTION_BUF,
        'seed': seed,
        'vectorization_mode': 'sync'}
    if task_type == "REWARD":
        env, _ = make_humenv(task=task_name.replace('_', '.'), **kwargs)
        env = DummyVecEnv([lambda: env])
        path = os.path.join(path, env_name)
        norm_path = os.path.join(path, "td3_vecnormalize.pkl")
        env = VecNormalize.load(norm_path, env)
        env.norm_reward = False
        env.training = False
        env = RewardEnv(env)
    else:
        env, _ = make_humenv(**kwargs)
        env = FixedGoalEnv(env, goals[task_name])
        env = DummyVecEnv([lambda: env])
        path = os.path.join(path, env_name)
        norm_path = os.path.join(path, "td3_vecnormalize.pkl")
        env = VecNormalize.load(norm_path, env)
        env.norm_reward = False
        env.training = False
        env = RewardEnv(env)
    return env
