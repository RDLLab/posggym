import math
import random
from itertools import product
from typing import Tuple, NamedTuple, Set, Optional, Dict, Sequence, List

from gym import spaces

import posggym.model as M

from posggym.envs.grid_world.core import Coord, Direction
from posggym.envs.grid_world.predator_prey.grid import PPGrid


class PPState(NamedTuple):
    """A state in the Predator-Prey Environment."""
    predator_coords: Tuple[Coord, ...]
    prey_coords: Tuple[Coord, ...]
    prey_caught: Tuple[int, ...]


PPAction = int
PPJointAction = Tuple[PPAction, ...]

# The actions
DO_NOTHING = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

ACTIONS = [DO_NOTHING, UP, DOWN, LEFT, RIGHT]
ACTIONS_STR = ["0", "U", "D", "L", "R"]
ACTION_TO_DIR = [
    None, Direction.NORTH, Direction.SOUTH, Direction.WEST, Direction.EAST
]

# Obs = (adj_obs)
PPObs = Tuple[int, ...]
PPJointObs = Tuple[PPObs, ...]

# Cell Obs
EMPTY = 0
WALL = 1
PREDATOR = 2
PREY = 3

CELL_OBS = [EMPTY, WALL, PREDATOR, PREY]
CELL_OBS_STR = ["0", "#", "P", "p"]


class PPB0(M.Belief):
    """The initial belief in a Predator-Prey problem."""

    def __init__(self,
                 num_predators: int,
                 num_prey: int,
                 grid: PPGrid,
                 rng: random.Random,
                 dist_res: int = 1000):
        assert grid.prey_start_coords is not None
        self._num_predators = num_predators
        self._num_prey = num_prey
        self._grid = grid
        self._rng = rng

        center_coord = (self._grid.size // 2, self._grid.size // 2)
        dist_from_center = math.ceil(math.sqrt(self._num_prey)) - 1
        self._prey_start_coords = self._grid.get_coords_within_dist(
            center_coord,
            dist_from_center,
            ignore_blocks=False,
            include_origin=True
        )

    def sample(self) -> M.State:
        predator_start_coords = [*self._grid.predator_start_coords]
        self._rng.shuffle(predator_start_coords)
        predator_start_coords = tuple(
            predator_start_coords[:self._num_predators]
        )

        prey_start_coords = set(self._grid.prey_start_coords)
        prey_start_coords.difference_update(predator_start_coords)
        prey_start_coords = list(prey_start_coords)
        self._rng.shuffle(prey_start_coords)
        prey_start_coords = tuple(prey_start_coords[:self._num_prey])

        prey_caught = (0, ) * self._num_prey

        return PPState(predator_start_coords, prey_start_coords, prey_caught)

    def sample_k(self, k: int) -> Sequence[M.State]:
        return [self.sample() for _ in range(k)]

    def get_dist(self) -> Dict[M.State, float]:
        return self.sample_belief_dist(self._dist_res)


class PPModel(M.POSGModel):
    """Predator-Prey Problem Model.

    Parameters
    ----------
    size : int
        the size of the grid (height and width)
    num_predators : int
        the number of predator (and thus controlled agents)
    num_prey : int
        the number of prey
    cooperative : bool
        whether environment rewards are fully shared (True) or only awarded to
        capturing predators (i.e. mixed) (False)
    prey_strenth : int
        the minimum number of predators needed to capture a prey
    obs_dims : int
        number of cells in each direction around the agent that the agent can
        observe

    """

    R_MAX = 1.0
    PREY_CAUGHT_COORD = (0, 0)

    def __init__(self,
                 grid: PPGrid,
                 num_predators: int,
                 num_prey: int,
                 cooperative: bool,
                 prey_strength: int,
                 obs_dim: int,
                 **kwargs):
        assert 1 < num_predators <= 8
        assert 0 < num_prey
        assert 0 < obs_dim
        assert 0 < prey_strength <= min(4, num_predators)
        assert (
            grid.prey_start_coords is None
            or len(grid.prey_start_coords) >= num_prey
        )
        super().__init__(num_predators, **kwargs)

        self.grid = grid
        self.obs_dim = obs_dim
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.cooperative = cooperative
        self.prey_strength = prey_strength
        self._per_prey_reward = self.R_MAX / self.num_prey

        self._rng = random.Random(kwargs.get("seed", None))

        if self.grid.prey_start_coords is None:
            center_coords = self.grid.get_unblocked_center_coords(num_prey)
            self.grid.prey_start_coords = center_coords

    @property
    def observation_first(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def state_space(self) -> spaces.Space:
        # coords of each agent
        coord_space = spaces.Tuple((
            spaces.Discrete(self.grid.width), spaces.Discrete(self.grid.height)
        ))
        predator_coord_space = spaces.Tuple(tuple(
            coord_space for _ in range(self.num_predators)
        ))
        prey_coord_space = spaces.Tuple(tuple(
            coord_space for _ in range(self.num_prey)
        ))
        caught_space = spaces.Tuple(tuple(
            spaces.Discrete(2) for _ in range(self.num_prey)
        ))
        return spaces.Tuple((
            predator_coord_space, prey_coord_space, caught_space
        ))

    @property
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        return tuple(
            spaces.Discrete(len(ACTIONS)) for _ in range(self.n_agents)
        )

    @property
    def observation_spaces(self) -> Tuple[spaces.Space, ...]:
        local_cell_obs = spaces.Tuple(tuple(
            spaces.Discrete(len(CELL_OBS))
            for _ in range((2 * self.obs_dim + 1)**2)
        ))
        return tuple(local_cell_obs for _ in range(self.n_agents))

    @property
    def reward_ranges(self) -> Tuple[Tuple[M.Reward, M.Reward], ...]:
        return tuple((0.0, self.R_MAX) for _ in range(self.n_agents))

    @property
    def initial_belief(self) -> M.Belief:
        return PPB0(self.num_predators, self.num_prey, self.grid, self._rng)

    def get_agent_initial_belief(self,
                                 agent_id: int,
                                 obs: M.Observation) -> M.Belief:
        raise NotImplementedError

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        return self._get_obs(state, state)

    def step(self,
             state: M.State,
             actions: M.JointAction
             ) -> M.JointTimestep:
        next_state = self._get_next_state(state, actions)
        obs = self._get_obs(state, next_state)
        rewards = self._get_rewards(state, next_state)
        dones, all_done = self._is_done(next_state)
        outcomes = self._get_outcome(all_done)
        return M.JointTimestep(
            next_state, obs, rewards, dones, all_done, outcomes
        )

    def _get_next_state(self,
                        state: PPState,
                        actions: PPJointAction
                        ) -> PPState:
        # prey move first
        prey_coords = self._get_next_prey_state(state)
        predator_coords = self._get_next_predator_state(
            state, actions, prey_coords
        )
        prey_caught = self._get_next_prey_caught(
            state, prey_coords, predator_coords
        )
        return PPState(predator_coords, prey_coords, prey_caught)

    def _get_next_prey_state(self, state: PPState) -> Tuple[Coord, ...]:
        next_prey_coords = [None] * self.num_prey
        occupied_coords = set(
            state.predator_coords
            + tuple(
                state.prey_coords[i] for i in range(self.num_prey)
                if not state.prey_caught[i]
            )
        )

        # handle moving away from predators for all prey
        for i in range(self.num_prey):
            prey_coord = state.prey_coords[i]
            if state.prey_caught[i]:
                next_prey_coords[i] = prey_coord
                continue

            next_coord = self._move_away_from_predators(
                prey_coord, state.predator_coords, occupied_coords
            )
            if next_coord:
                next_prey_coords[i] = next_coord
                occupied_coords.remove(prey_coord)
                occupied_coords.add(next_coord)

        if self.num_prey - sum(state.prey_caught) > 1:
            # handle moving away from other prey
            for i in range(self.num_prey):
                if next_prey_coords[i] is not None:
                    # already moved or caught
                    continue

                prey_coord = state.prey_coords[i]
                next_coord = self._move_away_from_preys(
                    prey_coord, state, occupied_coords
                )
                if next_coord:
                    next_prey_coords[i] = next_coord
                    occupied_coords.remove(prey_coord)
                    occupied_coords.add(next_coord)

        # Handle random moving prey for those that are out of obs range
        # of all predators and other prey
        for i in range(self.num_prey):
            if next_prey_coords[i] is not None:
                continue
            # No visible prey or predator
            prey_coord = state.prey_coords[i]
            neighbours = self.grid.get_neighbours(prey_coord)
            if self.obs_dim > 1:
                # no chance of moving randomly into an occupied cell
                next_prey_coords[i] = self._rng.choice(neighbours)
            else:
                # possibility for collision between random moving prey
                neighbours = [
                    c for c in neighbours if c not in occupied_coords
                ]
                if len(neighbours) == 0:
                    next_coord = prey_coord
                else:
                    next_coord = self._rng.choice(neighbours)
                next_prey_coords[i] = next_coord
                occupied_coords.remove(prey_coord)
                occupied_coords.add(next_coord)

        return tuple(next_prey_coords)

    def _move_away_from_predators(self,
                                  prey_coord: Coord,
                                  predator_coords: Tuple[Coord, ...],
                                  occupied_coords: Set[Coord]
                                  ) -> Optional[Coord]:
        # get any predators within obs distance
        predator_dists = [
            self.grid.manhattan_dist(prey_coord, c)
            for c in predator_coords
        ]
        min_predator_dist = min(predator_dists)
        all_closest_predator_coords = [
            predator_coords[i] for i in range(self.num_predators)
            if predator_dists[i] == min_predator_dist
        ]
        closest_predator_coord = self._rng.choice(all_closest_predator_coords)

        if (all(
            abs(a-b) > self.obs_dim
            for a, b in zip(prey_coord, closest_predator_coord)
        )):
            # closes predator out of obs range
            return None

        # move into furthest away free cell, includes current coord
        neighbours = list(
            (self.grid.manhattan_dist(c, closest_predator_coord), c)
            for c in self.grid.get_neighbours(prey_coord) + [prey_coord]
        )
        neighbours.sort()
        for (d, c) in reversed(neighbours):
            if (
                c == prey_coord
                or self._coord_available_for_prey(c, occupied_coords)
            ):
                return c

        raise AssertionError("Something has gone wrong, please investigate.")

    def _move_away_from_preys(self,
                              prey_coord: Coord,
                              state: PPState,
                              occupied_coords: Set[Coord]
                              ) -> Optional[Coord]:
        prey_dists = [
            self.grid.manhattan_dist(prey_coord, c)
            if (c != prey_coord and not state.prey_caught[i]) else float("inf")
            for i, c in enumerate(state.prey_coords)
        ]
        min_prey_dist = min(prey_dists)
        all_closest_prey_coords = [
            state.prey_coords[i] for i in range(self.num_prey)
            if prey_dists[i] == min_prey_dist
        ]
        closest_prey_coord = self._rng.choice(all_closest_prey_coords)

        if (all(
            abs(a-b) > self.obs_dim
            for a, b in zip(prey_coord, closest_prey_coord)
        )):
            # closes predator out of obs range
            return None

        # move into furthest away free cell, includes current coord
        neighbours = list(
            (self.grid.manhattan_dist(c, closest_prey_coord), c)
            for c in self.grid.get_neighbours(prey_coord) + [prey_coord]
        )
        neighbours.sort()
        for (d, c) in reversed(neighbours):
            if (
                c == prey_coord
                or self._coord_available_for_prey(c, occupied_coords)
            ):
                return c

        raise AssertionError("Something has gone wrong, please investigate.")

    def _coord_available_for_prey(self,
                                  coord: Coord,
                                  occupied_coords: Set[Coord]) -> bool:
        if coord in occupied_coords:
            return False
        neighbours = self.grid.get_neighbours(
            coord, ignore_blocks=False, include_out_of_bounds=False
        )
        for c in neighbours:
            if c in occupied_coords:
                neighbours.remove(c)
        return len(neighbours) >= self.prey_strength

    def _get_next_predator_state(self,
                                 state: PPState,
                                 actions: PPJointAction,
                                 next_prey_coords: Tuple[Coord, ...]
                                 ) -> Tuple[Coord, ...]:
        potential_next_coords = []
        for i, coord in enumerate(state.predator_coords):
            if actions[i] == 0:
                next_coord = coord
            else:
                a_dir = ACTION_TO_DIR[actions[i]]
                next_coord = self.grid.get_next_coord(
                    coord, a_dir, ignore_blocks=False
                )
                if next_coord in next_prey_coords:
                    next_coord = coord
            potential_next_coords.append(next_coord)

        # handle collisions
        next_coords = []
        for i in range(self.num_predators):
            coord_i = potential_next_coords[i]
            collision = False
            for j in range(self.num_predators):
                if i == j:
                    continue
                elif coord_i == potential_next_coords[j]:
                    collision = True
                    break
            if collision:
                next_coords.append(state.predator_coords[i])
            else:
                next_coords.append(coord_i)

        return tuple(next_coords)

    def _get_next_prey_caught(self,
                              state: PPState,
                              next_prey_coords: Tuple[Coord, ...],
                              next_predator_coords: Tuple[Coord, ...]
                              ) -> Tuple[int, ...]:
        prey_caught = []
        for i in range(self.num_prey):
            if state.prey_caught[i]:
                prey_caught.append(1)
            else:
                predator_dists = [
                    self.grid.manhattan_dist(next_prey_coords[i], c)
                    for c in next_predator_coords
                ]
                num_adj_predators = sum(d <= 1 for d in predator_dists)
                prey_caught.append(
                    int(num_adj_predators >= self.prey_strength)
                )
        return tuple(prey_caught)

    def _get_obs(self, state: PPState, next_state: PPState) -> PPJointObs:
        return tuple(
            self._get_local_cell__obs(i, state, next_state)
            for i in range(self.num_predators)
        )

    def _get_local_cell__obs(self,
                             agent_id: M.AgentID,
                             state: PPState,
                             next_state: PPState) -> Tuple[int, ...]:
        obs_size = (2 * self.obs_dim) + 1
        agent_coord = next_state.predator_coords[agent_id]

        cell_obs = []
        for col, row in product(range(obs_size), repeat=2):
            obs_grid_coord = self._map_obs_to_grid_coord(
                (col, row), agent_coord
            )
            if (
                obs_grid_coord is None
                or obs_grid_coord in self.grid.block_coords
            ):
                cell_obs.append(WALL)
            elif obs_grid_coord in next_state.predator_coords:
                cell_obs.append(PREDATOR)
            elif obs_grid_coord in next_state.prey_coords:
                prey_obs_added = False
                for i, c in enumerate(next_state.prey_coords):
                    if c != obs_grid_coord or state.prey_caught[i]:
                        continue
                    cell_obs.append(PREY)
                    prey_obs_added = True
                    break
                if not prey_obs_added:
                    # coord is for previously caught prey and is now empty
                    cell_obs.append(EMPTY)
            else:
                cell_obs.append(EMPTY)
        return tuple(cell_obs)

    def _map_obs_to_grid_coord(self,
                               obs_coord: Coord,
                               agent_coord: Coord) -> Optional[Coord]:
        grid_col = agent_coord[0] + obs_coord[0] - self.obs_dim
        grid_row = agent_coord[1] + obs_coord[1] - self.obs_dim

        if (
            0 <= grid_row < self.grid.height
            and 0 <= grid_col < self.grid.width
        ):
            return (grid_col, grid_row)
        return None

    def get_obs_coords(self, origin: Coord) -> List[Coord]:
        """Get the list of coords observed from agent at origin."""
        obs_size = (2 * self.obs_dim) + 1
        obs_coords: List[Coord] = []
        for col, row in product(range(obs_size), repeat=2):
            obs_grid_coord = self._map_obs_to_grid_coord((col, row), origin)
            if obs_grid_coord is not None:
                obs_coords.append(obs_grid_coord)
        return obs_coords

    def _get_rewards(self,
                     state: PPState,
                     next_state: PPState) -> M.JointReward:
        new_caught_prey = []
        for i in range(self.num_prey):
            if not state.prey_caught[i] and next_state.prey_caught[i]:
                new_caught_prey.append(next_state.prey_coords[i])

        if len(new_caught_prey) == 0:
            return (0.0, ) * self.num_predators

        if self.cooperative:
            reward = len(new_caught_prey) * (self._per_prey_reward)
            return (reward, ) * self.num_predators

        rewards = [0.0] * self.num_predators
        for prey_coord in new_caught_prey:
            adj_coords = self.grid.get_neighbours(
                prey_coord, ignore_blocks=False, include_out_of_bounds=False
            )
            involved_predators = []
            for coord in adj_coords:
                try:
                    predator_i = next_state.predator_coords.index(coord)
                    involved_predators.append(predator_i)
                except ValueError:
                    pass

            predator_reward = self._per_prey_reward / len(involved_predators)
            for i in involved_predators:
                rewards[i] += predator_reward

        return tuple(rewards)

    def _is_done(self, state: PPState) -> Tuple[Tuple[bool, ...], bool]:
        done = all(state.prey_caught)
        return (done, ) * self.num_predators, done

    def _get_outcome(self, all_done: bool) -> Tuple[M.Outcome, ...]:
        if all_done:
            return (M.Outcome.WIN, ) * self.num_predators
        return (M.Outcome.NA, ) * self.num_predators

    def set_seed(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
