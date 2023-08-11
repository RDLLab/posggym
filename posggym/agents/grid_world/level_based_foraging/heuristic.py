"""Heuristic policies for Level-Based Foraging env.

Reference:
https://github.com/uoe-agents/lb-foraging/blob/master/lbforaging/agents/heuristic_agent.py

"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, cast

from posggym.agents.policy import Policy, PolicyID, PolicyState
from posggym.agents.utils import action_distributions
from posggym.envs.grid_world.level_based_foraging import (
    LBFAction,
    LBFObs,
    LevelBasedForagingModel,
)

if TYPE_CHECKING:
    from posggym.envs.grid_world.core import Coord


class LBFHeuristicPolicy(Policy[LBFAction, LBFObs]):
    """Heuristic agent for the Level-Based Foraging env.

    This is the abstract Level-Based Foraging env heuristic policy class.
    Concrete implementations must implement the get_action_from_obs method.
    """

    def __init__(
        self, model: LevelBasedForagingModel, agent_id: str, policy_id: PolicyID
    ):
        super().__init__(model, agent_id, policy_id)
        assert model.observation_mode in ("vector", "tuple")
        self._rng = random.Random()
        self.field_width, self.field_height = model.field_size

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        state["last_obs"] = None
        return state

    def get_next_state(
        self,
        obs: LBFObs,
        state: PolicyState,
    ) -> PolicyState:
        return {"last_obs": obs}

    def sample_action(self, state: PolicyState) -> LBFAction:
        return self._get_action_from_obs(state["last_obs"])

    def get_pi(self, state: PolicyState) -> action_distributions.ActionDistribution:
        pi = self._get_pi_from_obs(state["last_obs"])
        return action_distributions.DiscreteActionDistribution(pi, self._rng)

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` no implemented by {self.__class__.__name__} policy"
        )

    def _get_action_from_obs(self, obs: LBFObs) -> LBFAction:
        """Get action from observation.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _get_pi_from_obs(self, obs: LBFObs) -> Dict[LBFAction, float]:
        """Get action distribution from observation.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _closest_food(
        self,
        agent_pos: Coord,
        food_obs: List[Tuple[int, int, int]],
        max_food_level: Optional[int] = None,
    ) -> Optional[Coord]:
        food_distances: Dict[int, List[Coord]] = {}
        for y, x, level in food_obs:
            if x == -1 or (max_food_level is not None and level > max_food_level):
                continue
            distance = (agent_pos[0] - y) ** 2 + (agent_pos[1] - x) ** 2
            if distance not in food_distances:
                food_distances[distance] = []
            food_distances[distance].append((y, x))

        if len(food_distances) == 0:
            # No food in sight
            return None

        return self._rng.choice(food_distances[min(food_distances)])

    def _center_of_agents(self, agent_pos: List[Coord]) -> Coord:
        y_mean = sum(coord[0] for coord in agent_pos) / len(agent_pos)
        x_mean = sum(coord[1] for coord in agent_pos) / len(agent_pos)
        return round(x_mean), round(y_mean)

    def _move_towards(
        self, agent_pos: Coord, target: Coord, allowed_actions: List[LBFAction]
    ) -> List[LBFAction]:
        y, x = agent_pos
        r, c = target

        valid_actions = []
        if r < y and LBFAction.NORTH in allowed_actions:
            valid_actions.append(LBFAction.NORTH)
        if r > y and LBFAction.SOUTH in allowed_actions:
            valid_actions.append(LBFAction.SOUTH)
        if c > x and LBFAction.EAST in allowed_actions:
            valid_actions.append(LBFAction.EAST)
        if c < x and LBFAction.WEST in allowed_actions:
            valid_actions.append(LBFAction.WEST)

        if valid_actions:
            return valid_actions
        else:
            raise ValueError("No simple path found")

    def _get_valid_move_actions(
        self, agent_obs: List[Tuple[int, int, int]]
    ) -> List[LBFAction]:
        y, x = agent_obs[0][:2]
        other_agent_pos = {o[:2] for o in agent_obs[1:] if o[0] > -1}

        valid_actions = []
        if y > 0 and (y - 1, x) not in other_agent_pos:
            valid_actions.append(LBFAction.NORTH)
        if y < self.field_height - 1 and (y + 1, x) not in other_agent_pos:
            valid_actions.append(LBFAction.SOUTH)
        if x < self.field_width - 1 and (y, x + 1) not in other_agent_pos:
            valid_actions.append(LBFAction.EAST)
        if x > 0 and (y, x - 1) not in other_agent_pos:
            valid_actions.append(LBFAction.WEST)

        return valid_actions

    def _get_actions_towards_food(
        self,
        agent_obs: List[Tuple[int, int, int]],
        center_pos: Coord,
        food_obs: List[Tuple[int, int, int]],
        max_food_level: Optional[int] = None,
    ) -> List[LBFAction]:
        closest_food = self._closest_food(center_pos, food_obs, max_food_level)
        if closest_food is None:
            actions = self._get_valid_move_actions(agent_obs)
            if actions:
                return actions
            return [LBFAction.NONE]

        food_y, food_x = closest_food
        y, x = agent_obs[0][:2]
        if (abs(food_y - y) + abs(food_x - x)) == 1:
            return [LBFAction.LOAD]

        valid_move_actions = self._get_valid_move_actions(agent_obs)
        try:
            return self._move_towards((y, x), (food_y, food_x), valid_move_actions)
        except ValueError:
            if valid_move_actions:
                return valid_move_actions
            return [LBFAction.NONE]

    def _parse_obs(
        self, obs: LBFObs
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        model = cast(LevelBasedForagingModel, self.model)
        return model.parse_obs(obs)


class LBFHeuristicPolicy1(LBFHeuristicPolicy):
    """Level-Based Foraging Heuristic Policy 1.

    This policy always goes to the closest observed food, irrespective of
    the foods level.
    """

    def _get_action_from_obs(self, obs: LBFObs) -> LBFAction:
        agent_obs, food_obs = self._parse_obs(obs)
        agent_pos = agent_obs[0][:2]
        actions = self._get_actions_towards_food(agent_obs, agent_pos, food_obs)
        return self._rng.choice(actions)

    def _get_pi_from_obs(self, obs: LBFObs) -> Dict[LBFAction, float]:
        agent_obs, food_obs = self._parse_obs(obs)
        agent_pos = agent_obs[0][:2]
        actions = self._get_actions_towards_food(agent_obs, agent_pos, food_obs)
        action_dist = {a: 0.0 for a in LBFAction}
        for a in actions:
            action_dist[a] = 1.0 / len(actions)
        return action_dist


class LBFHeuristicPolicy2(LBFHeuristicPolicy):
    """Level-Based Foraging Heuristic Policy 2.

    This policy goes towards the visible food that is closest to the centre of
    visible players, irrespective of food level.
    """

    def _get_action_from_obs(self, obs: LBFObs) -> LBFAction:
        agent_obs, food_obs = self._parse_obs(obs)
        other_agent_pos = [o[:2] for o in agent_obs[1:] if o[0] > -1]

        if not other_agent_pos:
            actions = self._get_valid_move_actions(agent_obs)
            if actions:
                return self._rng.choice(actions)
            return LBFAction.NONE

        center_pos = self._center_of_agents(other_agent_pos)
        actions = self._get_actions_towards_food(agent_obs, center_pos, food_obs)
        return self._rng.choice(actions)

    def _get_pi_from_obs(self, obs: LBFObs) -> Dict[LBFAction, float]:
        agent_obs, food_obs = self._parse_obs(obs)
        other_agent_pos = [o[:2] for o in agent_obs[1:] if o[0] > -1]

        if not other_agent_pos:
            # no visible agents
            actions = self._get_valid_move_actions(agent_obs)
            if not actions:
                actions = [LBFAction.NONE]
        else:
            center_pos = self._center_of_agents(other_agent_pos)
            actions = self._get_actions_towards_food(agent_obs, center_pos, food_obs)

        action_dist = {a: 0.0 for a in LBFAction}
        for a in actions:
            action_dist[a] = 1.0 / len(actions)
        return action_dist


class LBFHeuristicPolicy3(LBFHeuristicPolicy):
    """Level-Based Foraging Heuristic Policy 3.

    This policy goes towards the closest visible food with a compatible level.
    """

    def _get_action_from_obs(self, obs: LBFObs) -> LBFAction:
        agent_obs, food_obs = self._parse_obs(obs)
        agent_pos = agent_obs[0][:2]
        agent_level = agent_obs[0][2]
        actions = self._get_actions_towards_food(
            agent_obs, agent_pos, food_obs, agent_level
        )
        return self._rng.choice(actions)

    def _get_pi_from_obs(self, obs: LBFObs) -> Dict[LBFAction, float]:
        agent_obs, food_obs = self._parse_obs(obs)
        agent_pos = agent_obs[0][:2]
        agent_level = agent_obs[0][2]
        actions = self._get_actions_towards_food(
            agent_obs, agent_pos, food_obs, agent_level
        )
        action_dist = {a: 0.0 for a in LBFAction}
        for a in actions:
            action_dist[a] = 1.0 / len(actions)
        return action_dist


class LBFHeuristicPolicy4(LBFHeuristicPolicy):
    """Level-Based Foraging Heuristic Policy 4.

    This policy goes towards the visible food which is closest to all visible
    agents such that the sum of their and the policy agent's level is
    sufficient to load the food.
    """

    def _get_action_from_obs(self, obs: LBFObs) -> LBFAction:
        agent_obs, food_obs = self._parse_obs(obs)
        other_agent_pos = [o[:2] for o in agent_obs[1:] if o[0] > -1]

        if not other_agent_pos:
            actions = self._get_valid_move_actions(agent_obs)
            if actions:
                return self._rng.choice(actions)
            return LBFAction.NONE

        agent_level_sum = sum([o[2] for o in agent_obs if o[0] > -1])
        center_pos = self._center_of_agents(other_agent_pos)
        actions = self._get_actions_towards_food(
            agent_obs, center_pos, food_obs, agent_level_sum
        )
        return self._rng.choice(actions)

    def _get_pi_from_obs(self, obs: LBFObs) -> Dict[LBFAction, float]:
        agent_obs, food_obs = self._parse_obs(obs)
        other_agent_pos = [o[:2] for o in agent_obs[1:] if o[0] > -1]

        if not other_agent_pos:
            # no visible agents
            actions = self._get_valid_move_actions(agent_obs)
            if not actions:
                actions = [LBFAction.NONE]
        else:
            agent_level_sum = sum([o[2] for o in agent_obs if o[0] > -1])
            center_pos = self._center_of_agents(other_agent_pos)
            actions = self._get_actions_towards_food(
                agent_obs, center_pos, food_obs, agent_level_sum
            )

        action_dist = {a: 0.0 for a in LBFAction}
        for a in actions:
            action_dist[a] = 1.0 / len(actions)
        return action_dist
