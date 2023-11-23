"""Shortest path policy for Driving envs."""
from __future__ import annotations

from itertools import product
from queue import PriorityQueue
from typing import TYPE_CHECKING, Dict, Set, Tuple, cast

from posggym.agents.policy import Policy, PolicyID, PolicyState
from posggym.agents.utils import action_distributions
from posggym.envs.grid_world.core import Coord, Direction
from posggym.envs.grid_world.driving import (
    ACCELERATE,
    DECELERATE,
    DO_NOTHING,
    INIT_DIR,
    INIT_SPEED,
    TURN_LEFT,
    TURN_RIGHT,
    VEHICLE,
    DAction,
    DObs,
    DrivingGrid,
    DrivingModel,
    Speed,
)
from posggym.utils import seeding


if TYPE_CHECKING:
    from posggym.model import POSGModel
    from posggym.utils.history import AgentHistory


# Current coord, speed, facing direction
Pos = Tuple[Coord, Speed, Direction]


class DrivingShortestPathPolicy(Policy[DAction, DObs]):
    """Shortest Path Policy for the Driving environment.

    This policy selects actions that lead it to taking the shortest path to the agent's
    goal. If there are multiple actions on the shortest path then selects uniformly
    at random from those actions.

    Arguments
    ---------
    aggressiveness : float
        The aggressiveness of the policy towards other vehicles. A value of 0.0 means
        the policy will always stop when it sees another vehicle, while a value of 1.0
        means the policy will ignore other vehicles and always take the shortest path
        action. Values in between will change how far away another vehicle needs to
        be before the policy will stop. Furthermore, if `aggressiveness < 0.5` vehicles
        will never accelerate to max speed (Default=`1.0`).

    """

    # shared shortest paths lookup table for all class instances
    # this shares shortest path computation and storage between all instances of class
    # which is useful if running a vectorized environment or with many shortest path
    # agents
    # shortest_paths: Dict[Coord, Dict[Pos, Dict[Pos, int]]] = {}
    shortest_paths: Dict[Coord, Dict[Pos, int]] = {}

    def __init__(
        self,
        model: POSGModel,
        agent_id: str,
        policy_id: PolicyID,
        aggressiveness: float = 1.0,
    ):
        super().__init__(model, agent_id, policy_id)
        self.model = cast(DrivingModel, model)
        assert (
            0.0 <= aggressiveness <= 1.0
        ), f"Aggressiveness must be between 0.0 and 1.0, got {aggressiveness}"
        self.aggressiveness = aggressiveness
        self._grid = self.model.grid
        self._action_space = list(range(self.model.action_spaces[agent_id].n))
        self._rng, _ = seeding.std_random()

        obs_front, obs_back, obs_side = self.model.obs_dim
        self.obs_width = (2 * obs_side) + 1
        self.agent_obs_idx = (obs_front * self.obs_width) + obs_side
        self.agent_obs_coord = (obs_side, obs_front)
        self.max_obs_dist = max(obs_front, obs_back) + obs_side

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng, _ = seeding.std_random(seed=seed)

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        state.update(
            {
                "update_num": 0,
                "speed": INIT_SPEED,
                "facing_dir": INIT_DIR,
                "coord": (0, 0),
                # store previous coord to avoid getting stuck in same position
                "prev_coord": (0, 0),
                "dest_coord": (0, 0),
                "min_other_vehicle_dist": 0,
            }
        )
        return state

    def get_next_state(
        self,
        action: DAction | None,
        obs: DObs,
        state: PolicyState,
    ) -> PolicyState:
        if state["update_num"] == 0:
            # I.e. first observation with no action performed yet
            next_facing_dir = INIT_DIR
        else:
            next_facing_dir = self.model.get_next_direction(
                action, obs[1], state["facing_dir"]
            )

        next_state = {
            "update_num": state["update_num"] + 1,
            "speed": obs[1],
            "facing_dir": next_facing_dir,
            "coord": obs[2],
            "prev_coord": state["coord"],
            "dest_coord": obs[3],
            "min_other_vehicle_dist": self.get_min_other_vehicle_dist(obs[0]),
        }
        return next_state

    def get_state_from_history(self, history: AgentHistory) -> PolicyState:
        _, obs = history.get_last_step()
        assert obs is not None
        facing_dir = INIT_DIR
        for a, _ in history:
            if a is None:
                continue
            facing_dir = self.model.get_next_direction(a, facing_dir)
        return {
            "update_num": len(history),
            "speed": obs[1],
            "facing_dir": facing_dir,
            "coord": obs[2],
            "prev_coord": (0, 0) if len(history) == 1 else history[-2][1][2],
            "dest_coord": obs[3],
            "min_other_vehicle_dist": self.get_min_other_vehicle_dist(obs[0]),
        }

    def sample_action(self, state: PolicyState) -> DAction:
        return self.get_pi(state).sample()

    def get_pi(self, state: PolicyState) -> action_distributions.ActionDistribution:
        vehicle_dist = state["min_other_vehicle_dist"]
        if vehicle_dist <= self.max_obs_dist:
            proximity = (vehicle_dist - 1) / (self.max_obs_dist - 1)
            if proximity < (1 - self.aggressiveness):
                if state["speed"] > Speed.STOPPED:
                    pi = {DECELERATE: 1.0}
                elif state["speed"] < Speed.STOPPED:
                    pi = {ACCELERATE: 1.0}
                else:
                    pi = {DO_NOTHING: 1.0}
                return action_distributions.DiscreteActionDistribution(pi, self._rng)

        dists = []
        for a in self._action_space:
            if (
                self.aggressiveness < 0.5
                and state["speed"] >= Speed.FORWARD_SLOW
                and a == ACCELERATE
            ):
                # don't accelerate if aggressiveness is low and agent is already
                # moving at FORWARD_SLOW or faster
                dists.append(float("inf"))
                continue

            a_speed = self.model.get_next_speed(a, state["speed"])
            a_facing_dir = self.model.get_next_direction(
                a, a_speed, state["facing_dir"]
            )
            a_move_dir = self.model.get_move_direction(a, a_speed, state["facing_dir"])
            next_coord = state["coord"]
            for _ in range(abs(a_speed - Speed.STOPPED)):
                next_coord = self._grid.get_next_coord(
                    next_coord, a_move_dir, ignore_blocks=False
                )
            if next_coord == state["coord"]:
                # hit a wall
                a_speed = Speed.STOPPED

            d = self.get_dest_shortest_path_dist(
                state["dest_coord"], (next_coord, a_speed, a_facing_dir)
            )
            dists.append(d)

        min_dist = min(dists)
        num_min = dists.count(min_dist)
        pi = {}
        for a in self._action_space:
            if dists[a] == min_dist:
                pi[a] = 1.0 / num_min
            else:
                pi[a] = 0.0

        return action_distributions.DiscreteActionDistribution(pi, self._rng)

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` no implemented by {self.__class__.__name__} policy"
        )

    def get_min_other_vehicle_dist(self, local_obs: Tuple[int, ...]) -> int:
        """Get minimum distance to other vehicle in local observation."""
        min_other_vehicle_dist = self.max_obs_dist + 1
        for idx, cell_obs in enumerate(local_obs):
            if idx != self.agent_obs_idx and cell_obs == VEHICLE:
                col, row = idx % self.obs_width, idx // self.obs_width
                dist = abs(col - self.agent_obs_coord[0]) + abs(
                    row - self.agent_obs_coord[1]
                )
                min_other_vehicle_dist = min(dist, min_other_vehicle_dist)
        return min_other_vehicle_dist

    def get_dest_shortest_path_dist(self, dest_coord: Coord, pos: Pos) -> int:
        """Get shortest path distance to destination coord from given position."""
        cls = DrivingShortestPathPolicy
        if dest_coord not in cls.shortest_paths:
            cls.shortest_paths[dest_coord] = {}
        return cls.get_shortest_path(
            pos, dest_coord, self._grid, cls.shortest_paths[dest_coord]
        )

    @staticmethod
    def get_shortest_path(
        origin: Pos,
        dest: Coord,
        grid: DrivingGrid,
        lookup_table: Dict[Pos, int],
    ) -> int:
        """Get shortest path to given origin to given destination.

        Note, this is a search over vehicle configurations, i.e. (coord, speed,
        facing_dir), rather than just vehicle coordinate. This yields a shortest path
        distance in terms of number of actions, rather than number of cells.

        This method modifies the lookup table in place. Adding any new positions
        encountered to the lookup table.

        """
        if (dest, INIT_SPEED, INIT_DIR) not in lookup_table:
            for speed, facing_dir in product(Speed, Direction):
                lookup_table[(dest, speed, facing_dir)] = 0

        if origin in lookup_table:
            return lookup_table[origin]

        # Run djikstra from origin till it reaches a position in the lookup table
        move_dist = {origin: 0}
        prev_pos = {origin: None}
        pq = PriorityQueue()
        pq.put((move_dist[origin], origin))
        visited = {origin}

        last_pos = None
        while not pq.empty():
            _, pos = pq.get()
            if pos in lookup_table:
                last_pos = pos
                break

            for adj_pos in DrivingShortestPathPolicy.get_next_positions(pos, grid):
                if move_dist[pos] + 1 < move_dist.get(adj_pos, float("inf")):
                    move_dist[adj_pos] = move_dist[pos] + 1
                    prev_pos[adj_pos] = pos

                    if adj_pos not in visited:
                        pq.put((move_dist[adj_pos], adj_pos))
                        visited.add(adj_pos)

        if last_pos is None:
            raise ValueError(
                f"Could not find shortest path from {origin} to {dest} in grid"
            )

        # convert move_dist to shortest path dist from origin to dest
        # by starting at last_pos and working backwards
        d = lookup_table[last_pos]
        while last_pos is not None:
            lookup_table[last_pos] = d
            last_pos = prev_pos[last_pos]
            d += 1

        return lookup_table[origin]

    @staticmethod
    def get_next_positions(pos: Pos, grid: DrivingGrid) -> Set[Pos]:
        coord, speed, facing_dir = pos

        next_positions = set()
        for a in [DO_NOTHING, TURN_LEFT, TURN_RIGHT, ACCELERATE, DECELERATE]:
            next_speed = DrivingModel.get_next_speed(a, speed)
            move_dir = DrivingModel.get_move_direction(a, next_speed, facing_dir)
            next_dir = DrivingModel.get_next_direction(a, next_speed, facing_dir)

            next_coord = coord
            for _ in range(abs(next_speed - Speed.STOPPED)):
                next_coord = grid.get_next_coord(
                    next_coord, move_dir, ignore_blocks=False
                )

            if next_coord == coord:
                # hit a wall
                next_speed = Speed.STOPPED
            next_positions.add((next_coord, next_speed, next_dir))
        return next_positions
