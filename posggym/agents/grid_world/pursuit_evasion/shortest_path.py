"""Shortest path policy for PursuitEvasion env."""
from __future__ import annotations

from queue import PriorityQueue
from typing import TYPE_CHECKING, Dict, Set, Tuple, cast

import posggym.envs.grid_world.pursuit_evasion as pe
from posggym.agents.policy import Policy, PolicyID, PolicyState
from posggym.agents.utils import action_distributions
from posggym.envs.grid_world.core import Coord, Direction
from posggym.utils import seeding

if TYPE_CHECKING:
    from posggym.model import POSGModel
    from posggym.utils.history import AgentHistory


# Current coord, facing direction
Pos = Tuple[Coord, Direction]


class PEShortestPathPolicy(Policy[pe.PEAction, pe.PEObs]):
    """Shortest Path Policy for the Pursuit Evasion environment.

    This policy sets the preferred action as the one which is on the shortest
    path to a target location evaders goal.

    For the evader the target location is the evader's goal.

    For the pursuer the target location is the evader's start location. If the pursuer
    reaches the evader's start location without the episode ending, the target location
    changes to the next closest unexplored evader start or goal location.

    """

    def __init__(self, model: POSGModel, agent_id: str, policy_id: PolicyID):
        super().__init__(model, agent_id, policy_id)
        self.model = cast(pe.PursuitEvasionModel, model)
        self._grid = self.model.grid
        self._is_evader = int(self.agent_id) == self.model.EVADER_IDX
        self._action_space = list(range(self.model.action_spaces[self.agent_id].n))

        self.shortest_paths = {}
        for dest_coord in set(
            self._grid.evader_start_coords + self._grid.all_goal_coords
        ):
            self.shortest_paths[dest_coord] = self.get_all_shortest_paths(dest_coord)

        self._rng, _ = seeding.std_random()

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng, _ = seeding.std_random(seed=seed)

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        state.update(
            {
                "update_num": 0,
                "facing_dir": pe.INITIAL_DIR,
                "coord": (0, 0),
                "prev_coord": (0, 0),
                "target_coord": (0, 0),
                "visited_target_coords": None,
            }
        )
        return state

    def get_next_state(
        self,
        action: pe.PEAction | None,
        obs: pe.PEObs,
        state: PolicyState,
    ) -> PolicyState:
        if state["update_num"] == 0:
            # I.e. first observation with no action performed yet
            next_update_num = 1
            next_facing_dir = pe.INITIAL_DIR
            next_coord = obs[1] if self._is_evader else obs[2]
            next_prev_coord = (0, 0)
            target_coord = obs[3] if self._is_evader else obs[1]
            visited_target_coords = None
        else:
            next_update_num = state["update_num"] + 1
            next_facing_dir = pe.ACTION_TO_DIR[action][state["facing_dir"]]
            next_coord = self._grid.get_next_coord(
                state["coord"], next_facing_dir, False
            )
            next_prev_coord = state["coord"]
            target_coord = state["target_coord"]

            if not self._is_evader and next_coord == target_coord:
                if state["visited_target_coords"] is None:
                    visited_target_coords = []
                else:
                    visited_target_coords = list(state["visited_target_coords"])

                visited_target_coords.append(target_coord)
                unvisited_target_coords = set(
                    self._grid.evader_start_coords + self._grid.all_goal_coords
                ) - set(visited_target_coords)

                if unvisited_target_coords:
                    target_coord = min(
                        unvisited_target_coords,
                        key=lambda c: self.get_dest_shortest_path_dist(
                            c, (next_coord, next_facing_dir)
                        ),
                    )
                else:
                    # All evader start and goal locations have been visited so just keep
                    # target as current target (this is unlikely to happen within
                    # episode step limits)
                    target_coord = next_coord
                visited_target_coords = tuple(visited_target_coords)
            else:
                visited_target_coords = state["visited_target_coords"]

        next_state = {
            "update_num": next_update_num,
            "facing_dir": next_facing_dir,
            "coord": next_coord,
            "prev_coord": next_prev_coord,
            "target_coord": target_coord,
            "visited_target_coords": visited_target_coords,
        }
        return next_state

    def get_state_from_history(self, history: AgentHistory) -> PolicyState:
        last_action, obs = history.get_last_step()
        assert obs is not None
        facing_dir = pe.INITIAL_DIR
        coord = obs[1] if self._is_evader else obs[2]
        prev_coord = coord
        target_coord = obs[3] if self._is_evader else obs[1]
        for a, _ in history:
            if a is None:
                continue
            facing_dir = pe.ACTION_TO_DIR[a][facing_dir]
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

    def sample_action(self, state: PolicyState) -> pe.PEAction:
        return self.get_pi(state).sample()

    def get_pi(self, state: PolicyState) -> action_distributions.ActionDistribution:
        dists = []
        coord, target_coord = state["coord"], state["target_coord"]
        for a in self._action_space:
            a_dir = pe.ACTION_TO_DIR[a][state["facing_dir"]]
            next_coord = self._grid.get_next_coord(coord, a_dir, False)
            d = self.get_dest_shortest_path_dist(target_coord, (next_coord, a_dir))
            dists.append(d)

        min_dist = min(dists)
        num_min = dists.count(min_dist)
        pi = {}
        for a in self._action_space:
            if dists[a] == min_dist:
                pi[a] = 1 / num_min
            else:
                pi[a] = 0

        return action_distributions.DiscreteActionDistribution(pi, self._rng)

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` no implemented by {self.__class__.__name__} policy"
        )

    def get_dest_shortest_path_dist(self, dest_coord: Coord, pos: Pos) -> int:
        """Get shortest path distance to destination coord from given position."""
        if dest_coord not in self.shortest_paths:
            self.shortest_paths[dest_coord] = self.get_all_shortest_paths(dest_coord)

        min_dist = float("inf")
        for dists in self.shortest_paths[dest_coord].values():
            min_dist = min(min_dist, dists.get(pos, float("inf")))
        return min_dist

    def get_all_shortest_paths(self, origin: Coord) -> Dict[Pos, Dict[Pos, int]]:
        """Get shortest paths from given origin to all other positions.

        Note, this is a search over agent configurations (coord, facing_dir), rather
        than just agent coordinate. This yields a shortest path distance in terms of
        number of actions, rather than number of cells.
        """
        src_dists = {}
        for facing_dir in Direction:
            pos = (origin, facing_dir)
            src_dists[pos] = self.dijkstra(pos)
        return src_dists

    def dijkstra(self, origin: Pos) -> Dict[Pos, int]:
        """Get shortest path distance to origin from all other positions."""
        dist = {origin: 0}
        pq = PriorityQueue()  # type: ignore
        pq.put((dist[origin], origin))

        visited = {origin}

        while not pq.empty():
            _, pos = pq.get()

            for adj_pos in self.get_prev_positions(pos):
                if dist[pos] + 1 < dist.get(adj_pos, float("inf")):
                    dist[adj_pos] = dist[pos] + 1

                    if adj_pos not in visited:
                        pq.put((dist[adj_pos], adj_pos))
                        visited.add(adj_pos)
        return dist

    def get_prev_positions(self, pos: Pos) -> Set[Pos]:
        """Get all positions reachable from given position."""
        coord, facing_dir = pos
        prev_positions = set()
        for prev_a in [pe.FORWARD, pe.BACKWARD, pe.LEFT, pe.RIGHT]:
            if prev_a == pe.FORWARD:
                prev_dir = facing_dir
                move_dir = Direction((facing_dir + 2) % len(Direction))
            elif prev_a == pe.BACKWARD:
                prev_dir = facing_dir
                move_dir = facing_dir
            elif prev_a == pe.LEFT:
                prev_dir = Direction((facing_dir + 1) % len(Direction))
                move_dir = Direction((facing_dir + 2) % len(Direction))
            else:
                # RIGHT
                prev_dir = Direction((facing_dir - 1) % len(Direction))
                move_dir = Direction((facing_dir + 2) % len(Direction))

            next_coord = self._grid.get_next_coord(
                coord,
                Direction((move_dir + 2) % len(Direction)),
                ignore_blocks=False,
            )
            if next_coord == coord:
                # if next coord is blocked then may have moved to current
                # coord from 0 or 1 cells away
                prev_positions.add((coord, prev_dir))

            prev_coord = prev_coord = self._grid.get_next_coord(
                coord, move_dir, ignore_blocks=False
            )
            prev_positions.add((prev_coord, prev_dir))

        return prev_positions
