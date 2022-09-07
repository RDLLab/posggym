from typing import Tuple, Dict

import posggym.model as M

from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.highway_env import HighwayEnv


class HWAbstractMultiAgentEnv(AbstractEnv):
    """Adds MultiAgent functionality to the HighwayEnv AbstractEnv.

    https://github.com/eleurent/highway-env/blob/v1.6/highway_env/envs/common/abstract.py
    """

    def __init__(self, config: dict = None):
        # Used to control which of the controlled vehicles (i.e. the vehicle
        # for each agent) is being updated/accessed
        self._current_vehicle_idx = 0
        super().__init__(config)

    @property
    def vehicle(self) -> Vehicle:
        if self.controlled_vehicles:
            return self.controlled_vehicles[self._current_vehicle_idx]
        return None

    @vehicle.setter
    def vehicle(self, vehicle: Vehicle) -> None:
        if len(self.controlled_vehicles) == 1:
            self.controlled_vehicles = [vehicle]
        raise AssertionError

    def step(self,
             action: M.JointAction
             ) -> Tuple[M.JointObservation, M.JointReward, Tuple[bool], Dict]:
        if self.road is None or self.vehicle is None:
            raise NotImplementedError(
                "The road and vehicle must be initialized in the environment "
                "implementation"
            )

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()

        # bit of a hack
        rewards = []
        dones = []
        infos = {}
        for i in range(len(self.controlled_vehicles)):
            self._current_vehicle_idx = i
            reward = self._reward(action[i])
            terminal = self._is_terminal()
            info = self._info(obs[i], action[i])

            rewards.append(reward)
            dones.append(terminal)
            infos[i] = info

        return obs, tuple(rewards), tuple(dones), infos


class HWMultiAgentHighwayEnv(HighwayEnv):
    """Adds MultiAgent functionality to the highway-env HighwayEnv environment.

    https://github.com/eleurent/highway-env/blob/master/highway_env/envs/highway_env.py
    """

    def __init__(self, config: dict = None):
        # Used to control which of the controlled vehicles (i.e. the vehicle
        # for each agent) is being updated/accessed
        self._current_vehicle_idx = 0
        super().__init__(config)

    @property
    def vehicle(self) -> Vehicle:
        if self.controlled_vehicles:
            return self.controlled_vehicles[self._current_vehicle_idx]
        return None

    @vehicle.setter
    def vehicle(self, vehicle: Vehicle) -> None:
        if len(self.controlled_vehicles) == 1:
            self.controlled_vehicles = [vehicle]
        raise AssertionError

    def step(self,
             action: M.JointAction
             ) -> Tuple[M.JointObservation, M.JointReward, Tuple[bool], Dict]:
        if self.road is None or self.vehicle is None:
            raise NotImplementedError(
                "The road and vehicle must be initialized in the environment "
                "implementation"
            )

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()

        # bit of a hack
        rewards = []
        dones = []
        infos = {}
        for i in range(len(self.controlled_vehicles)):
            self._current_vehicle_idx = i
            reward = self._reward(action[i])
            terminal = self._is_terminal()
            info = self._info(obs[i], action[i])

            rewards.append(reward)
            dones.append(terminal)
            infos[i] = info

        return obs, tuple(rewards), tuple(dones), infos
