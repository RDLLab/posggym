"""Shortest path policy for Driving envs."""
from __future__ import annotations

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

    def __init__(
        self,
        model: POSGModel,
        agent_id: str,
        policy_id: PolicyID,
        aggressiveness: float = 1.0,
        precompute_shortest_paths: bool = False,
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

        self.shortest_paths = {}
        if precompute_shortest_paths:
            for dest_coord in set.union(*self._grid.dest_coords):
                self.shortest_paths[dest_coord] = self.get_all_shortest_paths(
                    dest_coord
                )

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

        # get distances to dest for each action
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

            d = self.get_dest_shortest_path_dist(
                state["dest_coord"], (next_coord, a_speed, a_facing_dir)
            )
            dists.append(d)

        # get pi from dists
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
        if dest_coord not in self.shortest_paths:
            self.shortest_paths[dest_coord] = self.get_all_shortest_paths(dest_coord)

        min_dist = float("inf")
        for dists in self.shortest_paths[dest_coord].values():
            if dists.get(pos, float("inf")) < min_dist:
                min_dist = dists[pos]
        return min_dist

    def get_all_shortest_paths(self, origin: Coord) -> Dict[Pos, Dict[Pos, int]]:
        """Get shortest paths from given origin to all other positions.

        Note, this is a search over vehicle configurations, i.e. (coord, speed,
        facing_dir), rather than just vehicle coordinate. This yields a shortest path
        distance in terms of number of actions, rather than number of cells.
        """
        src_dists = {}
        for speed in Speed:
            for facing_dir in Direction:
                pos = (origin, speed, facing_dir)
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
        coord, speed, facing_dir = pos
        prev_positions = set()
        for prev_a in [DO_NOTHING, TURN_LEFT, TURN_RIGHT, ACCELERATE, DECELERATE]:
            if prev_a in [DO_NOTHING, ACCELERATE, DECELERATE] or speed == Speed.REVERSE:
                # agent moved in straight line from prev pos
                if prev_a == DECELERATE:
                    prev_speed = Speed(min(speed + 1, Speed.FORWARD_FAST))
                    prev_dir = facing_dir
                elif prev_a == ACCELERATE:
                    prev_speed = Speed(max(speed - 1, Speed.REVERSE))
                    prev_dir = facing_dir
                else:
                    # DO_NOTHING or (TURN_LEFT or TURN_RIGHT in REVERSE)
                    # Agent cannot turn while reversing, so action is same as DO_NOTHING
                    prev_speed = speed
                    prev_dir = facing_dir

                if speed == Speed.REVERSE:
                    move_dir = prev_dir
                else:
                    move_dir = Direction((prev_dir + 2) % len(Direction))

                next_coord = self._grid.get_next_coord(
                    coord,
                    Direction((move_dir + 2) % len(Direction)),
                    ignore_blocks=False,
                )
                next_coord_blocked = next_coord == coord

                prev_coord = coord
                for _ in range(abs(prev_speed - Speed.STOPPED)):
                    if next_coord_blocked:
                        # if next coord is blocked then may have moved to current
                        # coord from 0, 1 or 2 cells away
                        prev_positions.add((prev_coord, prev_speed, prev_dir))
                    prev_coord = self._grid.get_next_coord(
                        prev_coord, move_dir, ignore_blocks=False
                    )
                prev_positions.add((prev_coord, prev_speed, prev_dir))
            else:
                # TURN_RIGHT or TURN_LEFT (and not in REVERSE)
                # agent turned and maybe moved from prev pos
                if prev_a == TURN_RIGHT:
                    prev_dir = Direction((facing_dir - 1) % len(Direction))
                else:
                    prev_dir = Direction((facing_dir + 1) % len(Direction))

                if speed == Speed.STOPPED:
                    # agent turned in place
                    next_coord = coord
                    prev_speed = speed
                    prev_positions.add((coord, prev_speed, prev_dir))
                elif speed == Speed.FORWARD_FAST:
                    # not possible, since turning reduces speed to FORWARD_SLOW
                    continue
                else:
                    # agent turned and moved
                    # if current speed is FORWARD_SLOW, then agent could have
                    # been moving at FORWARD_FAST or FORWARD_SLOW
                    prev_speeds = [speed]
                    if speed == Speed.FORWARD_SLOW:
                        prev_speeds.append(Speed.FORWARD_FAST)

                    prev_coords = set()
                    # Need to check if next coord is blocked, since agent may have
                    # turned into a wall and so stayed at same coord, but changed dir
                    next_coord = self._grid.get_next_coord(
                        coord, facing_dir, ignore_blocks=False
                    )
                    if next_coord == coord:
                        prev_coords.add(coord)

                    # agent moved from cell in opposite direction to facing dir
                    move_dir = Direction((facing_dir + 2) % len(Direction))
                    prev_coord = self._grid.get_next_coord(
                        coord, move_dir, ignore_blocks=False
                    )
                    prev_coords.add(prev_coord)

                    for prev_coord in prev_coords:
                        for prev_speed in prev_speeds:
                            prev_positions.add((prev_coord, prev_speed, prev_dir))

        return prev_positions
