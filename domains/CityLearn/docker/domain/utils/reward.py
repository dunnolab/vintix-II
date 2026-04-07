from typing import Mapping, Any, List
from citylearn.reward_function import RewardFunction


class CustomReward(RewardFunction):
    """Calculates custom user-defined multi-agent reward.

    Reward is the :py:attr:`net_electricity_consumption_emission`
    for entire district if central agent setup otherwise it is the
    :py:attr:`net_electricity_consumption_emission` each building.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    """

    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, int | float]]) -> List[float]:
        electricity_costs = [min(-o["net_electricity_consumption"], 0) * o["electricity_pricing"] for o in observations]
        carbon_emissions = [min(-o["net_electricity_consumption"], 0) * o["carbon_intensity"] for o in observations]
        if self.central_agent:
            total_cost = sum(electricity_costs)
            total_emission = sum(carbon_emissions)
            reward = [total_cost + total_emission]
        else:
            reward = [c + e for c, e in zip(electricity_costs, carbon_emissions)]

        return reward
