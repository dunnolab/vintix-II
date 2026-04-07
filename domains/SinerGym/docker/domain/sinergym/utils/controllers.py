"""Implementation of basic controllers."""
from datetime import datetime
from logging import Logger
from typing import Any, List, Sequence, Optional, Union

import numpy as np
import torch
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

from domain.sinergym.utils.constants import YEAR


class RandomController(object):

    def __init__(self, env: Env):
        """Random agent. It selects available actions randomly.

        Args:
            env (Env): Simulation environment.
        """
        self.env = env

    def act(self) -> Sequence[Any]:
        """Selects a random action from the environment's action_space.

        Returns:
            Sequence[Any]: Action chosen.
        """
        action = self.env.action_space.sample()
        return action


class RBC5Zone(object):

    def __init__(self, env: Env) -> None:
        """Agent based on static rules for controlling 5ZoneAutoDXVAV setpoints.
        Based on ASHRAE Standard 55-2004: Thermal Environmental Conditions for Human Occupancy.

        Args:
            env (Env): Simulation environment
        """

        self.env = env

        self.observation_variables = env.get_wrapper_attr(
            'observation_variables')
        self.action_variables = env.get_wrapper_attr('action_variables')

        self.setpoints_summer = np.array((23.0, 26.0), dtype=np.float32)
        self.setpoints_winter = np.array((20.0, 23.5), dtype=np.float32)

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on indoor temperature.

        Args:
            observation (List[Any]): Perceived observation.

        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = dict(zip(self.observation_variables, observation))
        year = int(obs_dict['year']) if obs_dict.get('year', False) else YEAR
        month = int(obs_dict['month'])
        day = int(obs_dict['day_of_month'])

        summer_start_date = datetime(year, 6, 1, 0, 0, 0)
        summer_final_date = datetime(year, 9, 30, 0, 0, 0)

        current_dt = datetime(year, month + 1, day + 1, 0, 0, 0)

        # Get season comfort range
        if current_dt >= summer_start_date and current_dt <= summer_final_date:  # pragma: no cover
            season_range = self.setpoints_summer
        else:  # pragma: no cover
            season_range = self.setpoints_winter

        return season_range

class RBC5ZonePolicy(BasePolicy):
# class RBC5ZonePolicy(OnPolicyAlgorithm):

    def __init__(self, env: Env, **kwargs):
        observation_space = env.observation_space
        action_space = env.action_space
        # super().__init__(observation_space, action_space, **kwargs)
        super().__init__(
            policy=None,
            env=env,
            learning_rate=0.0,
            verbose=0,
        )

        self.env = env
        self.observation_variables = env.get_wrapper_attr('observation_variables')
        self.action_variables = env.get_wrapper_attr('action_variables')

        self.setpoints_summer = np.array([23.0, 26.0], dtype=np.float32)
        self.setpoints_winter = np.array([20.0, 23.5], dtype=np.float32)

    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        # Convert tensor to list for easier handling
        observation = obs.cpu().numpy().flatten().tolist()
        action = self._rule_based_action(observation)
        return torch.tensor(action, dtype=torch.float32)

    def _rule_based_action(self, observation: List[Any]) -> Sequence[float]:
        obs_dict = dict(zip(self.observation_variables, observation))

        year = int(obs_dict.get('year', 2000))  # fallback to default
        month = int(obs_dict['month'])
        day = int(obs_dict['day_of_month'])

        summer_start = datetime(year, 6, 1)
        summer_end = datetime(year, 9, 30)
        current_dt = datetime(year, month + 1, day + 1)

        if summer_start <= current_dt <= summer_end:
            return self.setpoints_summer
        else:
            return self.setpoints_winter

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.forward(observation, deterministic)

    def set_logger(self, logger: Logger) -> None:
        """
        Setter for for logger object.

        .. warning::

          When passing a custom logger object,
          this will overwrite ``tensorboard_log`` and ``verbose`` settings
          passed to the constructor.
        """
        self._logger = logger
        # User defined logger
        self._custom_logger = True

    def learn(self, total_timesteps: int, callback=None, log_interval: int = 100, eval_env=None, eval_freq: int = -1, n_eval_episodes: int = 5, eval_log_path: Optional[str] = None, reset_num_timesteps: bool = True):
        # Rule-based agent does not learn, but we can log and simulate interaction
        iteration = 0

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                self._dump_logs(iteration)

        callback.on_training_end()

        return self

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)


class RBC5ZoneAgent(BaseAlgorithm):
    def __init__(
        self,
        env: Union[Env, VecEnv],
        verbose: int = 0,
    ):
        # Dummy policy and learning rate; we override behavior
        super().__init__(
            policy=None,
            env=env,
            learning_rate=0.0,
            verbose=verbose,
        )

        # self.observation_variables = self.env.get_wrapper_attr('observation_variables')
        # self.action_variables = self.env.get_wrapper_attr('action_variables')

        self.setpoints_summer = np.array([23.0, 26.0], dtype=np.float32)
        self.setpoints_winter = np.array([20.0, 23.5], dtype=np.float32)

    def set_logger(self, logger: Logger) -> None:
        self._logger = logger

    def _setup_model(self):
        # No model setup needed for rule-based agent
        pass

    def _get_season_range(self, year: int, month: int, day: int) -> np.ndarray:
        summer_start = datetime(year, 6, 1)
        summer_end = datetime(year, 9, 30)
        current_dt = datetime(year, month + 1, day + 1)
        return self.setpoints_summer if summer_start <= current_dt <= summer_end else self.setpoints_winter

    def predict(self, observation: np.ndarray, state: Optional[np.ndarray] = None, episode_start: Optional[np.ndarray] = None, deterministic: bool = True):
        # obs_dict = dict(zip(self.observation_variables, observation))

        obs_dict = dict(observation)
        year = int(obs_dict.get('year', 2000))
        month = int(obs_dict['month'])
        day = int(obs_dict['day_of_month'])

        action = self._get_season_range(year, month, day)
        return action, None

    def learn(self, total_timesteps: int, callback=None, log_interval: int = 100, eval_env=None, eval_freq: int = -1, n_eval_episodes: int = 5, eval_log_path: Optional[str] = None, reset_num_timesteps: bool = True):
        # Rule-based agent does not learn, but we can log and simulate interaction
        self._logger.log("Learning skipped: RBC5ZoneAgent is rule-based.")
        iteration = 0

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            # continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            # if not continue_training:
            #     break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            # if log_interval is not None and iteration % log_interval == 0:
            #     self._dump_logs(iteration)

        callback.on_training_end()

        return self


class RBCDatacenter(object):

    def __init__(self, env: Env) -> None:
        """Agent based on static rules for controlling 2ZoneDataCenterHVAC setpoints.
        Follows the ASHRAE recommended temperature ranges for data centers described in ASHRAE TC9.9 (2016).

        Args:
            env (Env): Simulation environment
        """

        self.env = env
        self.observation_variables = env.get_wrapper_attr(
            'observation_variables')
        self.action_variables = env.get_wrapper_attr('action_variables')

        # ASHRAE recommended temperature range = [18, 27] Celsius
        self.range_datacenter = np.array((18, 27), dtype=np.float32)

    def act(self) -> Sequence[Any]:
        """Select same action always, corresponding with comfort range.

        Returns:
            Sequence[Any]: Action chosen.
        """
        return self.range_datacenter


class RBCIncrementalDatacenter(object):

    def __init__(self, env: Env) -> None:
        """Agent based on rules for controlling 2ZoneDataCenterHVAC setpoints in a incremental way.
        Follows the ASHRAE recommended temperature ranges for data centers described in ASHRAE TC9.9 (2016).
        Args:
            env (Env): Simulation environment
        """

        self.env = env
        self.observation_variables = env.get_wrapper_attr(
            'observation_variables')
        self.action_variables = env.get_wrapper_attr('action_variables')

        # ASHRAE recommended temperature range = [18, 27] Celsius
        self.range_datacenter = (18, 27)

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on indoor temperature.
        Args:
            observation (List[Any]): Perceived observation.
        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = dict(zip(self.observation_variables, observation))

        # Mean temp in datacenter zones
        mean_temp = np.mean([obs_dict['west_zone_air_temperature'],
                             obs_dict['east_zone_air_temperature']])

        current_heat_setpoint = obs_dict[
            'west_zone_htg_setpoint']
        current_cool_setpoint = obs_dict[
            'west_zone_clg_setpoint']

        new_heat_setpoint = current_heat_setpoint
        new_cool_setpoint = current_cool_setpoint

        if mean_temp < self.range_datacenter[0]:  # pragma: no cover
            new_heat_setpoint = current_heat_setpoint + 1
            new_cool_setpoint = current_cool_setpoint + 1
        elif mean_temp > self.range_datacenter[1]:  # pragma: no cover
            new_cool_setpoint = current_cool_setpoint - 1
            new_heat_setpoint = current_heat_setpoint - 1

        return np.array(
            (new_heat_setpoint,
             new_cool_setpoint),
            dtype=np.float32)
