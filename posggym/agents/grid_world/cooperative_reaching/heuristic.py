"""Heuristic Bot Policies for the Cooperative Reaching Environment.

Adapted from:
https://github.com/uoe-agents/BRDiv/tree/master/envs/CooperativeReaching/coopreaching/coopreaching/agents

Reference:
Rahman et al (2023) Generating Teammates for Training Robust Ad Hoc Teamwork Agents via
Best-Response Diversity https://openreview.net/pdf?id=l5BzfQhROl

"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Optional

from posggym.agents.policy import Policy, PolicyID, PolicyState
from posggym.agents.utils import action_distributions
from posggym.envs.grid_world.cooperative_reaching import (
    DO_NOTHING,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    CooperativeReachingModel,
    CRAction,
    CRObs,
)

if TYPE_CHECKING:
    from posggym.envs.grid_world.core import Coord


class CRHeuristicPolicy(Policy[CRAction, CRObs]):
    """Heuristic policy for the Cooperative Reaching Environment.

    This is the base class for all heuristic policies. Concrete implementations must
    implement the `_get_target_pos` method.
    """

    def __init__(
        self, model: CooperativeReachingModel, agent_id: str, policy_id: PolicyID
    ):
        super().__init__(model, agent_id, policy_id)
        self._rng = random.Random()
        self.grid_size = model.size
        self.goals = model.goals

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        state["last_obs"] = None
        state["target_goal"] = None
        return state

    def get_next_state(
        self,
        obs: CRObs,
        state: PolicyState,
    ) -> PolicyState:
        return {
            "last_obs": obs,
            "target_goal": self._get_target_pos(obs, state["target_goal"]),
        }

    def sample_action(self, state: PolicyState) -> CRAction:
        possible_actions = self._move_towards(
            state["target_goal"], state["last_obs"][0]
        )
        return self._rng.choice(possible_actions)

    def get_pi(self, state: PolicyState) -> action_distributions.ActionDistribution:
        possible_actions = self._move_towards(
            state["target_goal"], tuple(state["last_obs"][0])
        )
        return action_distributions.DiscreteActionDistribution(
            {a: 1 / len(possible_actions) for a in possible_actions}, self._rng
        )

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` no implemented by {self.__class__.__name__} policy"
        )

    def _move_towards(self, target_pos: Coord, agent_pos: Coord) -> List[CRAction]:
        """Get list of actions that move towards target_pos from agent_pos."""
        valid_actions = []
        if target_pos[1] < agent_pos[1]:
            valid_actions.append(UP)
        elif target_pos[1] > agent_pos[1]:
            valid_actions.append(DOWN)

        if target_pos[0] < agent_pos[0]:
            valid_actions.append(LEFT)
        elif target_pos[0] > agent_pos[0]:
            valid_actions.append(RIGHT)

        if not valid_actions:
            valid_actions.append(DO_NOTHING)
        return valid_actions

    def _get_target_pos(self, obs: CRObs, target_goal: Optional[Coord]) -> Coord:
        """Get target position from observation."""
        raise NotImplementedError

    def _get_target_goal(
        self,
        obs: CRObs,
        closest: bool | None,
        optimal: bool | None,
    ) -> Coord:
        """Get target goal position from observation.

        Conditions selection on whether to go to the closest (closest=True), furthest
        (closest=False), or any distance (closest=None) goal, and whether to go to the
        optimal (highest value, optimal=True), suboptimal (non-max value, optimal=False)
        or any value (optimal=None) goal.
        """
        agent_pos = obs[0]

        if optimal is None:
            goal_list = list(self.goals)
        elif optimal:
            max_reward = max(self.goals.values())
            goal_list = [
                goal for goal, reward in self.goals.items() if reward == max_reward
            ]
        else:
            max_reward = max(self.goals.values())
            goal_list = [
                goal for goal, reward in self.goals.items() if reward != max_reward
            ]
            if not goal_list:
                # all goals are equal
                goal_list = list(self.goals)

        if closest is None:
            return self._rng.choice(goal_list)

        dist_to_goal = [
            abs(agent_pos[0] - goal[0]) + abs(agent_pos[1] - goal[1])
            for goal in goal_list
        ]
        desired_dist_to_goal = min(dist_to_goal) if closest else max(dist_to_goal)
        valid_target_goals = [
            g
            for g, dist in zip(goal_list, dist_to_goal)
            if dist == desired_dist_to_goal
        ]
        return self._rng.choice(valid_target_goals)


class CRHeuristicPolicy1(CRHeuristicPolicy):
    """Cooperative Reaching Heuristic Policy 1.

    H1 always goes to the closest rewarding goal.
    """

    def _get_target_pos(self, obs: CRObs, target_goal: Optional[Coord]) -> Coord:
        if target_goal is None:
            target_goal = self._get_target_goal(obs, closest=True, optimal=None)
        return target_goal


class CRHeuristicPolicy2(CRHeuristicPolicy):
    """Cooperative Reaching Heuristic Policy 2.

    H2 always goes to the furthest rewarding goal.
    """

    def _get_target_pos(self, obs: CRObs, target_goal: Optional[Coord]) -> Coord:
        if target_goal is None:
            target_goal = self._get_target_goal(obs, closest=False, optimal=None)
        return target_goal


class CRHeuristicPolicy3(CRHeuristicPolicy):
    """Cooperative Reaching Heuristic Policy 3

    H3 always goes to the closest optimal goal.
    """

    def _get_target_pos(self, obs: CRObs, target_goal: Optional[Coord]) -> Coord:
        if target_goal is None:
            target_goal = self._get_target_goal(obs, closest=True, optimal=True)
        return target_goal


class CRHeuristicPolicy4(CRHeuristicPolicy):
    """Cooperative Reaching Heuristic Policy 4

    H4 always goes to the furthest optimal goal.
    """

    def _get_target_pos(self, obs: CRObs, target_goal: Optional[Coord]) -> Coord:
        if target_goal is None:
            target_goal = self._get_target_goal(obs, closest=False, optimal=True)
        return target_goal


class CRHeuristicPolicy5(CRHeuristicPolicy):
    """Cooperative Reaching Heuristic Policy 5

    H5 always goes to the closest suboptimal goal.
    """

    def _get_target_pos(self, obs: CRObs, target_goal: Optional[Coord]) -> Coord:
        if target_goal is None:
            target_goal = self._get_target_goal(obs, closest=True, optimal=False)
        return target_goal


class CRHeuristicPolicy6(CRHeuristicPolicy):
    """Cooperative Reaching Heuristic Policy 6

    H6 always goes to the furthest suboptimal goal.
    """

    def _get_target_pos(self, obs: CRObs, target_goal: Optional[Coord]) -> Coord:
        if target_goal is None:
            target_goal = self._get_target_goal(obs, closest=False, optimal=False)
        return target_goal


class CRHeuristicPolicy7(CRHeuristicPolicy):
    """Cooperative Reaching Heuristic Policy 7

    H7 goes to a randomly selected goal.
    """

    def _get_target_pos(self, obs: CRObs, target_goal: Optional[Coord]) -> Coord:
        if target_goal is None:
            target_goal = self._rng.choice(list(self.goals))
        return target_goal


class CRHeuristicPolicy8(CRHeuristicPolicy):
    """Cooperative Reaching Heuristic Policy 8

    H8 goes to the goal closest to the other agent at each time step.
    """

    def _get_target_pos(self, obs: CRObs, target_goal: Optional[Coord]) -> Coord:
        other_pos = obs[1]
        if other_pos == (self.grid_size, self.grid_size):
            # cannot see other agent, so just go towards a random goal
            return self._rng.choice(list(self.goals))

        dist_to_goal = [
            abs(other_pos[0] - goal[0]) + abs(other_pos[1] - goal[1])
            for goal in self.goals
        ]
        min_dist_to_goal = min(dist_to_goal)
        closest_goals = [
            g for g, dist in zip(self.goals, dist_to_goal) if dist == min_dist_to_goal
        ]
        return self._rng.choice(closest_goals)


class CRHeuristicPolicy9(CRHeuristicPolicy):
    """Cooperative Reaching Heuristic Policy 9

    H9 goes to the optimal goal closest to the other agent.
    """

    def _get_target_pos(self, obs: CRObs, target_goal: Optional[Coord]) -> Coord:
        other_pos = obs[1]
        if other_pos == (self.grid_size, self.grid_size):
            # cannot see other agent, so just go towards a random optimal goal
            return self._get_target_goal(obs, closest=None, optimal=True)

        goal_list = [
            goal
            for goal, reward in self.goals.items()
            if reward == max(self.goals.values())
        ]

        dist_to_goal = [
            abs(other_pos[0] - goal[0]) + abs(other_pos[1] - goal[1])
            for goal in goal_list
        ]
        min_dist_to_goal = min(dist_to_goal)
        closest_goals = [
            g for g, dist in zip(goal_list, dist_to_goal) if dist == min_dist_to_goal
        ]
        return self._rng.choice(closest_goals)


class CRHeuristicPolicy10(CRHeuristicPolicy):
    """Cooperative Reaching Heuristic Policy 10

    H10 follows the other agent.
    """

    def _get_target_pos(self, obs: CRObs, target_goal: Optional[Coord]) -> Coord:
        other_pos = obs[1]
        if other_pos == (self.grid_size, self.grid_size):
            # cannot see other agent, so just go towards a random goal
            return self._rng.choice(list(self.goals))
        return other_pos


# def get_cr_vec_bot(
#     bot_class: type[CRHeuristicPolicy],
#     num_envs: int | None,
#     goals: dict[Coord, float],
# ) -> Callable[[], CRHeuristicPolicy]:
#     """Get vectorized Cooperative Reaching bot."""

#     def thunk():
#         return bot_class(goals)

#     return VecBot(thunk, num_envs)


# if __name__ == "__main__":
#     import time
#     from itertools import product

#     from posggym.envs.grid_world.cooperative_reaching import CooperativeReachingEnv

#     def run_env(env, policies, n_steps=100):
#         obs, _ = env.reset()
#         env.render()
#         for pi in policies.values():
#             pi.reset()
#         ep_return = 0
#         for _ in range(n_steps):
#             actions = {
#                 i: policies[i].step(obs[i], np.array([False])) for i in env.agents
#             }
#             obs, rewards, _, _, _ = env.step(actions)
#             ep_return += rewards[env.possible_agents[0]]

#             env.render()
#             time.sleep(0.05)

#             if not env.agents:
#                 print(f"Episode return: {ep_return}")
#                 obs, _ = env.reset()
#                 for pi in policies.values():
#                     pi.reset()
#                 ep_return = 0

#         return ep_return

#     for (m, n), s, bot_class in product(
#         [
#             ("original", 4),
#         ],
#         [10],
#         ALL_CR_BOTS,
#     ):
#         print(
#             f"\nRunning mode: {m}, n_goals: {n}, size: {s}, bot: {bot_class.__name__}"
#         )
#         if bot_class.__doc__ is not None:
#             print(bot_class.__doc__.split("\n")[0])
#             input("Press enter to continue...")
#         env = CooperativeReachingEnv(size=s, n_goals=n, mode=m, render_mode="human")

#         env.reset()
#         policies = {
#             env.possible_agents[0]: bot_class(env.possible_agents[0], s, env.goals),
#             env.possible_agents[1]: bot_class(env.possible_agents[1], s, env.goals),
#         }
#         run_env(env, policies)

#         env.close()

#         env.close()
#         env.close()

#         env.close()
