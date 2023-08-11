"""Shortest path policies for pursuit evasion continuous environment."""
from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING, List, Tuple, cast

import numpy as np

from posggym.agents.policy import Policy, PolicyID, PolicyState
from posggym.agents.utils import action_distributions
from posggym.envs.continuous.core import PMBodyState
from posggym.envs.continuous.pursuit_evasion_continuous import (
    PEAction,
    PEObs,
    PursuitEvasionContinuousModel,
)
from posggym.utils import seeding


if TYPE_CHECKING:
    from posggym.posggym.model import POSGModel
    from posggym.utils.history import AgentHistory


class PECShortestPathPolicy(Policy[PEAction, PEObs]):
    """Shortest path policy for pursuit evasion continuous environment."""

    def __init__(self, model: POSGModel, agent_id: str, policy_id: PolicyID):
        super().__init__(model, agent_id, policy_id)
        self.model = cast(PursuitEvasionContinuousModel, model)
        self._rng, _ = seeding.np_random()

        self.n_sensors = self.model.n_sensors
        self.sensor_obs_dim = self.model.sensor_obs_dim
        self.is_evader = self.agent_id == str(self.model.EVADER_IDX)
        self.action_space = self.model.action_spaces[self.agent_id]
        self.action_dtype = self.action_space.dtype

        # world without other agent for simulating next agent position
        self._world = self.model.world.copy()
        if self.is_evader:
            self._world.remove_entity("pursuer")
        else:
            self._world.remove_entity("evader")

        # all evader start and goal coords
        evader_coords = list(
            set(self._world.evader_start_coords + self._world.all_goal_coords)
        )
        self._dists = self._world.get_all_shortest_paths(evader_coords)

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng, _ = seeding.np_random(seed=seed)

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        state.update(
            {
                "action": None,
                "pi": None,
                "last_obs": None,
                "update_num": 0,
                # x, y, angle, vel_x, vel_y, vel_angle
                "body_state": PMBodyState(0, 0, 0, 0, 0, 0),
                "prev_body_state": PMBodyState(0, 0, 0, 0, 0, 0),
                "target_coord": np.zeros(2),
            }
        )
        return state

    def get_next_state(
        self,
        obs: PEObs,
        state: PolicyState,
    ) -> PolicyState:
        if state["update_num"] == 0:
            next_update_num = 1
            if self.is_evader:
                next_coord = obs[self.sensor_obs_dim + 1 : self.sensor_obs_dim + 3]
                target_coord = obs[self.sensor_obs_dim + 5 : self.sensor_obs_dim + 7]
            else:
                next_coord = obs[self.sensor_obs_dim + 3 : self.sensor_obs_dim + 5]
                target_coord = obs[self.sensor_obs_dim + 1 : self.sensor_obs_dim + 3]
            next_body_state = PMBodyState(next_coord[0], next_coord[1], 0, 0, 0, 0)
        else:
            next_update_num = state["update_num"] + 1
            next_body_state = self._get_next_body_state(
                state["body_state"], state["action"]
            )
            target_coord = state["target_coord"]

        _, pi = self._get_shortest_path_action(
            state["body_state"], next_body_state, target_coord
        )

        next_state = {
            "action": pi.sample(),
            "pi": pi,
            "last_obs": obs,
            "update_num": next_update_num,
            "body_state": next_body_state,
            "prev_body_state": state["body_state"],
            "target_coord": target_coord,
        }
        return next_state

    def get_state_from_history(self, history: AgentHistory) -> PolicyState:
        _, last_obs = history.get_last_step()
        if last_obs is None:
            return self.get_initial_state()

        state = self.get_initial_state()
        obs = history[0][1]
        if self.is_evader:
            next_coord = obs[self.sensor_obs_dim + 1 : self.sensor_obs_dim + 3]
            target_coord = obs[self.sensor_obs_dim + 5 : self.sensor_obs_dim + 7]
        else:
            next_coord = obs[self.sensor_obs_dim + 3 : self.sensor_obs_dim + 5]
            target_coord = obs[self.sensor_obs_dim + 1 : self.sensor_obs_dim + 3]
        state["last_obs"] = obs
        state["update_num"] = 1
        state["body_state"] = PMBodyState(next_coord[0], next_coord[1], 0, 0, 0, 0)
        state["target_coord"] = target_coord

        for a, o in history[1:]:
            state["last_obs"] = o
            state["update_num"] += 1
            state["body_state"] = self._get_next_body_state(state["body_state"], a)

        # need to get next action given final observation
        _, pi = self._get_shortest_path_action(state["body_state"], target_coord)

        state["action"] = np.array(pi.sample(), dtype=self.action_dtype)
        state["pi"] = pi
        return state

    def sample_action(self, state: PolicyState) -> PEAction:
        if state["pi"] is None:
            raise ValueError(
                "Policy state does not contain a valid action distribution. Make sure"
                "to call `step` or `get_next_state` before calling `sample_action`"
            )
        return np.array(state["pi"].sample(), dtype=self.action_dtype)

    def get_pi(self, state: PolicyState) -> action_distributions.ActionDistribution:
        return state["pi"]

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` no implemented by {self.__class__.__name__} policy"
        )

    def _get_next_body_state(
        self, body_state: PMBodyState, action: PEAction
    ) -> PMBodyState:
        agent_entity_id = "evader" if self.is_evader else "pursuer"

        self._world.set_entity_state(agent_entity_id, body_state)
        angle = body_state.angle + action[0]
        vel = self._world.linear_to_xy_velocity(action[1], angle)
        self._world.update_entity_state(agent_entity_id, angle=angle, vel=vel)

        self._world.simulate(1.0 / 10, 10)

        return self._world.get_entity_state(agent_entity_id)

    def _get_shortest_path_action(
        self,
        prev_body_state: PMBodyState,
        body_state: PMBodyState,
        target_coord: np.ndarray,
    ) -> Tuple[List[PEAction], action_distributions.ActionDistribution]:
        angle_vels = [
            -self.model.dyaw_limit,
            -self.model.dyaw_limit / 2.0,
            0.0,
            self.model.dyaw_limit / 2.0,
            self.model.dyaw_limit,
        ]
        linear_vels = [0.0, 0.5, 1.0]

        target_int_coord = self._world.convert_to_coord(target_coord)

        # find closest unblocked int coord to agent's float coord
        agent_coord = (body_state.x, body_state.y)
        agent_int_coord = self._world.convert_to_coord(agent_coord)
        dist = self._world.euclidean_dist(agent_coord, agent_int_coord)
        for x, y in product(
            [math.floor(body_state.x), math.ceil(body_state.x)],
            [math.floor(body_state.y), math.ceil(body_state.y)],
        ):
            if (x, y) in self._dists[target_int_coord]:
                d = self._world.euclidean_dist(agent_coord, (x, y))
                if d < dist:
                    agent_int_coord = (x, y)
                    dist = d

        # find adjacent int coords that are on shortest path to target
        target_next_agent_int_coords = [agent_int_coord]
        sp_dist = self._dists[target_int_coord][agent_int_coord]
        for dxy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_agent_int_coord = (
                agent_int_coord[0] + dxy[0],
                agent_int_coord[1] + dxy[1],
            )
            if next_agent_int_coord not in self._dists[target_int_coord]:
                continue

            dist = self._dists[target_int_coord][next_agent_int_coord]
            if dist == sp_dist:
                target_next_agent_int_coords.append(next_agent_int_coord)
            elif dist < sp_dist:
                target_next_agent_int_coords = [next_agent_int_coord]
                sp_dist = dist
        target_next_agent_int_coord = self._rng.choice(target_next_agent_int_coords)

        # find actions that move agent towards target adjacent coord
        sp_actions = [(-self.model.dyaw_limit, 0.0)]
        sp_dist = float("inf")
        for a in product(angle_vels, linear_vels):
            next_body_state = self._get_next_body_state(body_state, a)
            dist = self._world.euclidean_dist(
                (next_body_state.x, next_body_state.y), target_next_agent_int_coord
            )
            if np.isclose(dist, sp_dist):
                sp_actions.append(a)
            elif dist < sp_dist:
                sp_dist = dist
                sp_actions = [a]

        # Need to handle case where agent gets stuck
        if np.isclose(
            (prev_body_state.x, prev_body_state.y),
            (body_state.x, body_state.y),
            rtol=0.0,
            atol=1e-1,
        ).all():
            if all(a[1] == 0.0 for a in sp_actions):
                # try to move forward
                sp_actions = [(a[0], a[1] + 0.5) for a in sp_actions]
            elif all(a[0] == 0.0 for a in sp_actions):
                # try to turn
                old_sp_actions = sp_actions
                sp_actions = []
                for a in old_sp_actions:
                    sp_actions += [
                        (a[0] + self.model.dyaw_limit / 2.0, a[1]),
                        (a[0] - self.model.dyaw_limit / 2.0, a[1]),
                    ]

        pi = action_distributions.DiscreteActionDistribution(
            {a: 1.0 / len(sp_actions) for a in sp_actions},
            self._rng,
        )
        return sp_actions, pi
