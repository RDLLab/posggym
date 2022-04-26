"""The POSG Model for the Driving Problem."""
import enum
import random
from itertools import product
from typing import Tuple, Sequence, Dict, Set, List, NamedTuple, Optional

from gym import spaces

import posggym.model as M

from posggym.envs.grid_world.utils import Direction, Coord

from posggym.envs.grid_world.driving.grid import DrivingGrid


# TODO Try using named tuple for vehicle state, see if it has a noticable
#      impact on speed


class Speed(enum.IntEnum):
    """A speed setting for a vehicle."""
    REVERSE = 0
    STOPPED = 1
    FORWARD_SLOW = 2
    FORWARD_FAST = 3


class CollisionType(enum.IntEnum):
    """Type of collision for a vehicle."""
    NONE = 0
    OBSTACLE = 1
    VEHICLE = 2


# vehicle state = (
#   current_coord, Direction, Speed, dest_coord, dest_reached
# )
# N.B. dest_reached needed so agent isn't rewarded twice for reaching dest

class VehicleState(NamedTuple):
    """The state of a vehicle in the Driving Environment."""
    coord: Coord
    facing_dir: Direction
    speed: Speed
    dest_coord: Coord
    dest_reached: int
    crashed: int


DState = Tuple[VehicleState, ...]

# Initial direction and speed of each vehicle
INIT_DIR = Direction.NORTH
INIT_SPEED = Speed.STOPPED

DAction = int
DJointAction = Tuple[DAction, ...]

# The actions
DO_NOTHING = 0
ACCELERATE = 1
DECELERATE = 2
TURN_RIGHT = 3
TURN_LEFT = 4

ACTIONS = [DO_NOTHING, ACCELERATE, DECELERATE, TURN_RIGHT, TURN_LEFT]
ACTIONS_STR = ["0", "acc", "dec", "tr", "tl"]

# Obs = (adj_obs, speed, dest_coord, dest_reached, crashed)
DObs = Tuple[Tuple[int, ...], Speed, Coord, int, int]
DJointObs = Tuple[DObs, ...]

# Cell obs
VEHICLE = 0
WALL = 1
EMPTY = 2

CELL_OBS = [VEHICLE, WALL, EMPTY]
CELL_OBS_STR = ["V", "#", "0"]


class DB0(M.Belief):
    """The initial belief in a Driving problem."""

    def __init__(self,
                 n_agents: int,
                 grid: DrivingGrid,
                 rng: random.Random,
                 dist_res: int = 1000):
        assert n_agents <= grid.supported_num_agents
        assert n_agents <= grid.supported_num_agents
        self._n_agents = n_agents
        self._grid = grid
        self._rng = rng
        self._dist_res = dist_res

    def sample(self) -> M.State:
        state = []
        chosen_start_coords: Set[Coord] = set()
        chosen_dest_coords: Set[Coord] = set()
        for i in range(self._n_agents):
            start_coords_i = self._grid.start_coords[i]
            avail_start_coords = start_coords_i.difference(chosen_start_coords)
            start_coord = self._rng.choice(list(avail_start_coords))
            chosen_start_coords.add(start_coord)

            dest_coords_i = self._grid.dest_coords[i]
            avail_dest_coords = dest_coords_i.difference(chosen_dest_coords)
            dest_coord = self._rng.choice(list(avail_dest_coords))
            chosen_dest_coords.add(dest_coord)

            # VehicleState = Tuple[Coord, Direction, Speed, Coord]
            state_i = VehicleState(
                coord=start_coord,
                facing_dir=INIT_DIR,
                speed=INIT_SPEED,
                dest_coord=dest_coord,
                dest_reached=int(False),
                crashed=int(False)
            )
            state.append(state_i)
        return tuple(state)

    def sample_k(self, k: int) -> Sequence[M.State]:
        return [self.sample() for _ in range(k)]

    def get_dist(self) -> Dict[M.State, float]:
        return self.sample_belief_dist(self._dist_res)


class DrivingModel(M.POSGModel):
    """Driving Problem Model.

    Parameters
    ----------
    grid : DrivingGrid
        the grid environment for the model scenario
    num_agents : int
        the number of agents in the model scenario
    obs_dims : (int, int, int)
        number of cells in front, behind, and to the side that each agent
        can observe
    obstacle_collisions : bool
        whether cars can crash into wall and other obstacles, on top of
        crashing into other vehicles
    infinite_horizon : bool
        whether problem should terminate once a terminal state is reached
        (default, False) or reset to start position and continue (True).

    """

    R_ACTION = 0.00
    R_CRASH_OBJECT = -0.5
    R_CRASH_VEHICLE = -1.0
    R_DESTINATION_REACHED = 1.0

    def __init__(self,
                 grid: DrivingGrid,
                 num_agents: int,
                 obs_dim: Tuple[int, int, int],
                 obstacle_collisions: bool,
                 infinite_horizon: bool = False,
                 **kwargs):
        assert 0 < num_agents <= grid.supported_num_agents
        assert obs_dim[0] > 0 and obs_dim[1] >= 0 and obs_dim[2] >= 0
        super().__init__(num_agents, **kwargs)
        self.grid = grid
        self._obs_front, self._obs_back, self._obs_side = obs_dim
        self._obstacle_collisions = obstacle_collisions
        self._infinite_horizon = infinite_horizon

        self._rng = random.Random(None)

    @property
    def state_space(self) -> spaces.Space:
        vehicle_state_space = spaces.Tuple((
            # coord
            spaces.Tuple((
                spaces.Discrete(self.grid.width),
                spaces.Discrete(self.grid.height)
            )),
            # direction
            spaces.Discrete(len(Direction)),
            # speed
            spaces.Discrete(len(Speed)),
            # destination coord
            spaces.Tuple((
                spaces.Discrete(self.grid.width),
                spaces.Discrete(self.grid.height)
            )),
            # destination reached
            spaces.Discrete(2),
            # crashed
            spaces.Discrete(2)
        ))

        return spaces.Tuple(
            tuple(vehicle_state_space for _ in range(self.n_agents))
        )

    @property
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        return tuple(
            spaces.Discrete(len(ACTIONS)) for _ in range(self.n_agents)
        )

    @property
    def obs_spaces(self) -> Tuple[spaces.Space, ...]:
        obs_depth = self._obs_front + self._obs_back + 1
        obs_width = (2 * self._obs_side) + 1
        local_cell_obs = spaces.Tuple(tuple(
            spaces.Discrete(len(CELL_OBS))
            for _ in range(obs_depth * obs_width)
        ))

        return tuple(
            spaces.Tuple((
                local_cell_obs,
                # speed
                spaces.Discrete(len(Speed)),
                # dest coord
                spaces.Tuple((
                    spaces.Discrete(self.grid.width),
                    spaces.Discrete(self.grid.height)
                )),
                # dest reached
                spaces.Discrete(2),
                # crashed
                spaces.Discrete(2)
            ))
            for _ in range(self.n_agents)
        )

    @property
    def reward_ranges(self) -> Tuple[Tuple[M.Reward, M.Reward], ...]:
        return tuple(
            (self.R_CRASH_VEHICLE, self.R_DESTINATION_REACHED)
            for _ in range(self.n_agents)
        )

    @property
    def initial_belief(self) -> M.Belief:
        return DB0(self.n_agents, self.grid, self._rng)

    def get_agent_initial_belief(self,
                                 agent_id: int,
                                 obs: M.Observation) -> M.Belief:
        return self.initial_belief

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        return self._get_obs(state)

    def step(self,
             state: M.State,
             actions: M.JointAction
             ) -> M.JointTimestep:
        next_state, collision_types = self._get_next_state(state, actions)
        rewards = self._get_rewards(state, next_state, collision_types)
        done = self._state_is_terminal(next_state)

        outcomes = self.get_outcome(next_state) if done else None

        # TODO only reset vehicles that reach a terminal state
        if self._infinite_horizon and done:
            next_state = self.sample_initial_state()   # type: ignore
            done = False

        obs = self._get_obs(next_state)

        return M.JointTimestep(next_state, obs, rewards, done, outcomes)

    def _get_next_state(self,
                        state: DState,
                        actions: DJointAction
                        ) -> Tuple[DState, List[CollisionType]]:
        exec_order = list(range(self.n_agents))
        self._rng.shuffle(exec_order)

        next_state = [vs for vs in state]
        vehicle_coords: Set[Coord] = set([vs.coord for vs in state])
        collision_types = [CollisionType.NONE] * self.n_agents

        for i in exec_order:
            action_i = actions[i]
            state_i = state[i]
            next_state_i = next_state[i]
            # check next_state in case agent i was crashed into this step
            if state_i.dest_reached or next_state_i.crashed:
                continue

            vehicle_coords.remove(state_i.coord)

            next_speed = self._get_next_speed(action_i, state_i.speed)
            move_dir = self._get_move_direction(
                action_i, next_speed, state_i.facing_dir)
            next_dir = self._get_next_direction(action_i, state_i.facing_dir)
            next_speed = self._get_next_speed(action_i, state_i.speed)
            next_coord, collision_type, hit_vehicle = self._get_next_coord(
                state_i.coord, next_speed, move_dir, vehicle_coords
            )

            crashed = False
            if collision_type == CollisionType.VEHICLE:
                # update state of vehicle that was hit
                crashed = True
                collision_types[i] = collision_type
                for j in range(self.n_agents):
                    next_state_j = next_state[j]
                    if next_state_j.coord == hit_vehicle:
                        collision_types[j] = CollisionType.VEHICLE
                        next_state[j] = VehicleState(
                            coord=next_state_j.coord,
                            facing_dir=next_state_j.facing_dir,
                            speed=next_state_j.speed,
                            dest_coord=next_state_j.dest_coord,
                            dest_reached=next_state_j.dest_reached,
                            crashed=int(True)
                        )
                        break
            elif (
                self._obstacle_collisions
                and collision_type == CollisionType.OBSTACLE
            ):
                crashed = True
                collision_types[i] = collision_type

            next_state[i] = VehicleState(
                coord=next_coord,
                facing_dir=next_dir,
                speed=next_speed,
                dest_coord=state_i.dest_coord,
                dest_reached=int(next_coord == state_i.dest_coord),
                crashed=int(crashed)
            )

            vehicle_coords.add(next_coord)

        return tuple(next_state), collision_types

    def _get_move_direction(self,
                            action: DAction,
                            speed: Speed,
                            curr_dir: Direction) -> Direction:
        if speed == Speed.REVERSE:
            # No turning while in reverse,
            # so movement dir is just the opposite of current direction
            return Direction((curr_dir + 2) % len(Direction))
        return self._get_next_direction(action, curr_dir)

    @staticmethod
    def _get_next_direction(action: DAction, curr_dir: Direction) -> Direction:
        if action == TURN_RIGHT:
            return Direction((curr_dir + 1) % len(Direction))
        if action == TURN_LEFT:
            return Direction((curr_dir - 1) % len(Direction))
        return curr_dir

    @staticmethod
    def _get_next_speed(action: DAction, curr_speed: Speed) -> Speed:
        if action == DO_NOTHING:
            return curr_speed
        if action in (TURN_LEFT, TURN_RIGHT):
            if curr_speed == Speed.FORWARD_FAST:
                return Speed.FORWARD_SLOW
            return curr_speed
        if action == ACCELERATE:
            return Speed(min(curr_speed + 1, Speed.FORWARD_FAST))
        return Speed(max(curr_speed - 1, Speed.REVERSE))

    def _get_next_coord(self,
                        curr_coord: Coord,
                        speed: Speed,
                        move_dir: Direction,
                        vehicle_coords: Set[Coord]
                        ) -> Tuple[Coord, CollisionType, Optional[Coord]]:
        # assumes curr_coord isn't in vehicle coords
        next_coord = curr_coord
        collision = CollisionType.NONE
        hit_vehicle_coord = None
        for i in range(abs(speed - Speed.STOPPED)):
            next_coord = self.grid.get_next_coord(
                curr_coord, move_dir, ignore_blocks=False
            )
            if next_coord == curr_coord:
                collision = CollisionType.OBSTACLE
                break
            elif next_coord in vehicle_coords:
                collision = CollisionType.VEHICLE
                hit_vehicle_coord = next_coord
                next_coord = curr_coord
                break
            curr_coord = next_coord

        return (next_coord, collision, hit_vehicle_coord)

    def _get_obs(self, state: DState) -> M.JointObservation:
        agent_obs = tuple(
            self._get_agent_obs(state, i) for i in range(self.n_agents)
        )
        return agent_obs

    def _get_agent_obs(self, state: DState, agent_id: M.AgentID) -> DObs:
        local_cell_obs = self._get_local_cell__obs(state, agent_id)
        state_i = state[agent_id]
        return (
            local_cell_obs,
            state_i.speed,
            state_i.dest_coord,
            state_i.dest_reached,
            state_i.crashed
        )

    def _get_local_cell__obs(self,
                             state: DState,
                             agent_id: M.AgentID) -> Tuple[int, ...]:
        state_i = state[agent_id]
        obs_depth = self._obs_front + self._obs_back + 1
        obs_width = (2 * self._obs_side) + 1
        vehicle_coords = set([vs.coord for vs in state])

        cell_obs: List[int] = []
        for col, row in product(range(obs_width), range(obs_depth)):
            obs_grid_coord = self._map_obs_to_grid_coord(
                (col, row), state_i.coord, state_i.facing_dir
            )
            if (
                obs_grid_coord is None
                or obs_grid_coord in self.grid.block_coords
            ):
                cell_obs.append(WALL)
            elif obs_grid_coord in vehicle_coords:
                cell_obs.append(VEHICLE)
            else:
                cell_obs.append(EMPTY)
        return tuple(cell_obs)

    def _map_obs_to_grid_coord(self,
                               obs_coord: Coord,
                               agent_coord: Coord,
                               facing_dir: Direction) -> Optional[Coord]:
        grid_row = -1
        grid_col = -1
        if facing_dir == Direction.NORTH:
            grid_row = agent_coord[1] + obs_coord[1] - self._obs_front
            grid_col = agent_coord[0] + obs_coord[0] - self._obs_side
        elif facing_dir == Direction.EAST:
            grid_row = agent_coord[1] + obs_coord[0] - self._obs_side
            grid_col = agent_coord[0] - obs_coord[1] + self._obs_front
        elif facing_dir == Direction.SOUTH:
            grid_row = agent_coord[1] - obs_coord[1] + self._obs_front
            grid_col = agent_coord[0] - obs_coord[0] + self._obs_side
        else:
            grid_row = agent_coord[1] - obs_coord[0] + self._obs_side
            grid_col = agent_coord[0] + obs_coord[1] - self._obs_front

        if (
            0 <= grid_row < self.grid.height
            and 0 <= grid_col < self.grid.width
        ):
            return (grid_col, grid_row)
        return None

    def get_obs_coords(self,
                       origin: Coord,
                       facing_dir: Direction) -> List[Coord]:
        """Get the list of coords observed from agent at origin."""
        obs_depth = self._obs_front + self._obs_back + 1
        obs_width = (2 * self._obs_side) + 1
        obs_coords: List[Coord] = []
        for col, row in product(range(obs_width), range(obs_depth)):
            obs_grid_coord = self._map_obs_to_grid_coord(
                (col, row), origin, facing_dir
            )
            if obs_grid_coord is not None:
                obs_coords.append(obs_grid_coord)
        return obs_coords

    def _get_rewards(self,
                     state: DState,
                     next_state: DState,
                     collision_types: List[CollisionType]) -> M.JointReward:
        rewards = []
        for i in range(self.n_agents):
            if state[i].crashed or state[i].dest_reached:
                # already in terminal/rewarded state
                r_i = 0.0
            elif (
                self._obstacle_collisions
                and collision_types[i] == CollisionType.OBSTACLE
            ):
                r_i = self.R_CRASH_OBJECT
            elif collision_types[i] == CollisionType.VEHICLE:
                r_i = self.R_CRASH_VEHICLE
            elif next_state[i].dest_reached:
                r_i = self.R_DESTINATION_REACHED
            else:
                r_i = self.R_ACTION
            rewards.append(r_i)

        return tuple(rewards)

    def is_done(self, state: M.State) -> bool:
        if self._infinite_horizon:
            return False
        return self._state_is_terminal(state)

    def get_outcome(self, state: M.State) -> Tuple[M.Outcome, ...]:
        if self._infinite_horizon:
            return (M.Outcome.NA, M.Outcome.NA)

        outcomes = []
        for i in range(self.n_agents):
            if state[i].dest_reached:
                outcome_i = M.Outcome.WIN
            elif state[i].crashed:
                outcome_i = M.Outcome.LOSS
            else:
                outcome_i = M.Outcome.DRAW
            outcomes.append(outcome_i)
        return tuple(outcomes)

    def set_seed(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def _vehicle_state_is_terminal(self, vehicle_state: VehicleState) -> bool:
        return bool(vehicle_state.dest_reached or vehicle_state.crashed)

    def _state_is_terminal(self, state: DState) -> bool:
        return all(self._vehicle_state_is_terminal(vs) for vs in state)
