"""Shortest path policy for PursuitEvasion env."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, cast

from posggym.agents.policy import Policy, PolicyID, PolicyState
from posggym.agents.utils import action_distributions
from posggym.envs.grid_world.pursuit_evasion import (
    ACTION_TO_DIR,
    INITIAL_DIR,
    PEAction,
    PEObs,
    PursuitEvasionModel,
)
from posggym.utils import seeding

if TYPE_CHECKING:
    from posggym.envs.grid_world.core import Coord, Direction
    from posggym.model import POSGModel
    from posggym.utils.history import AgentHistory


class PEShortestPathPolicy(Policy[PEAction, PEObs]):
    """Shortest Path Policy for the Pursuit Evasion environment.

    This policy sets the preferred action as the one which is on the shortest
    path to the evaders goal and which doesn't leave agent in same position.

    """

    def __init__(self, model: POSGModel, agent_id: str, policy_id: PolicyID, **kwargs):
        super().__init__(model, agent_id, policy_id)
        self.model = cast(PursuitEvasionModel, model)
        assert all(p == 1.0 for p in self.model.action_probs), (
            f"{self.__class__.__name__} only supported for deterministic versions of "
            "the PursuitEvasion environment."
        )

        self._grid = self.model.grid
        self._is_evader = self.agent_id == self.model.EVADER_IDX

        self._action_space = list(
            range(self.model.action_spaces[self.agent_id].n)  # type: ignore
        )

        evader_end_coords = list(
            set(self._grid.evader_start_coords + self._grid.all_goal_coords)
        )
        self._dists = self._grid.get_all_shortest_paths(evader_end_coords)

        self._rng, _ = seeding.std_random()

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng, _ = seeding.std_random(seed=seed)

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        state.update(
            {
                "action": None,
                "update_num": 0,
                "facing_dir": INITIAL_DIR,
                "coord": (0, 0),
                "prev_coord": (0, 0),
                "target_coord": (0, 0),
            }
        )
        return state

    def get_next_state(
        self,
        obs: PEObs,
        state: PolicyState,
    ) -> PolicyState:
        if state["update_num"] == 0:
            # I.e. first observation with no action performed yet
            next_update_num = 1
            next_facing_dir = INITIAL_DIR
            next_coord = obs[1] if self._is_evader else obs[2]
            next_prev_coord = (0, 0)
            target_coord = obs[3] if self._is_evader else obs[1]
        else:
            next_update_num = state["update_num"] + 1
            next_facing_dir = ACTION_TO_DIR[state["action"]][state["facing_dir"]]
            next_coord = self._grid.get_next_coord(
                state["coord"], next_facing_dir, False
            )
            next_prev_coord = state["coord"]
            target_coord = state["target_coord"]

        next_action = self._get_sp_action(
            next_facing_dir,
            next_coord,
            next_prev_coord,
            target_coord,
        )
        next_state = {
            "update_num": next_update_num,
            "facing_dir": next_facing_dir,
            "coord": next_coord,
            "prev_coord": next_prev_coord,
            "target_coord": target_coord,
            "action": next_action,
        }
        return next_state

    def get_state_from_history(self, history: AgentHistory) -> PolicyState:
        last_action, obs = history.get_last_step()
        assert obs is not None
        facing_dir = INITIAL_DIR
        coord = obs[1] if self._is_evader else obs[2]
        prev_coord = coord
        target_coord = obs[3] if self._is_evader else obs[1]
        for a, _ in history:
            if a is None:
                continue
            facing_dir = ACTION_TO_DIR[a][facing_dir]
            prev_coord = coord
            coord = self._grid.get_next_coord(coord, facing_dir, False)
        return {
            "action": last_action,
            "update_num": len(history),
            "facing_dir": facing_dir,
            "coord": coord,
            "prev_coord": prev_coord,
            "target_coord": target_coord,
        }

    def sample_action(self, state: PolicyState) -> PEAction:
        return state["action"]

    def get_pi(self, state: PolicyState) -> action_distributions.ActionDistribution:
        dist = self._get_pi_from_coords(
            state["facing_dir"],
            state["coord"],
            state["prev_coord"],
            state["target_coord"],
        )
        return action_distributions.DiscreteActionDistribution(dist, self._rng)

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` no implemented by {self.__class__.__name__} policy"
        )

    def _get_pi_from_coords(
        self, facing_dir: Direction, coord: Coord, prev_coord: Coord, goal_coord: Coord
    ) -> Dict[PEAction, float]:
        dists = self._get_action_sp_dists(facing_dir, coord, prev_coord, goal_coord)

        pi = {a: 1.0 for a in self._action_space}
        max_dist, min_dist = max(dists), min(dists)
        dist_gap = max(1, max_dist - min_dist)

        weight_sum = 0.0
        for i, a in enumerate(self._action_space):
            weight = 1 - ((dists[i] - min_dist) / dist_gap)
            pi[a] = weight
            weight_sum += weight

        for a in self._action_space:
            pi[a] /= weight_sum

        return pi

    def _get_sp_action(
        self, facing_dir: Direction, coord: Coord, prev_coord: Coord, goal_coord: Coord
    ) -> PEAction:
        sp_action = self._action_space[0]
        sp_dist = float("inf")
        for a in self._action_space:
            a_dir = ACTION_TO_DIR[a][facing_dir]
            next_coord = self._grid.get_next_coord(coord, a_dir, False)
            if next_coord == prev_coord:
                continue

            dist = self._dists[goal_coord][next_coord]
            if dist < sp_dist:
                sp_dist = dist
                sp_action = a
        return sp_action

    def _get_action_sp_dists(
        self, facing_dir: Direction, coord: Coord, prev_coord: Coord, goal_coord: Coord
    ) -> List[int]:
        dists = []
        for a in self._action_space:
            a_dir = ACTION_TO_DIR[a][facing_dir]
            new_coord = self._grid.get_next_coord(coord, a_dir, False)
            if new_coord == prev_coord:
                # action leaves position unchanged or reverses direction
                # so penalize distance by setting it to max possible dist
                d = self._dists[goal_coord][coord]
                d += 2
            else:
                d = self._dists[goal_coord][new_coord]
            dists.append(d)
        return dists
