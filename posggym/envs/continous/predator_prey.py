"""The Predator-Prey Grid World Environment.

A co-operative 2D grid world problem involving multiple predator agents working together
to catch prey agents in the environment.

Reference
---------
- Ming Tan. 1993. Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents
  In Proceedings of the Tenth International Conference on Machine Learning. 330–337.
- J. Z. Leibo, V. F. Zambaldi, M. Lanctot, J. Marecki, and T. Graepel. 2017. Multi-Agent
  Reinforcement Learning in Sequential Social Dilemmas. In AAMAS, Vol. 16. ACM, 464–473

"""
import math
from itertools import product
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    SupportsFloat,
    Tuple,
)

from gymnasium import spaces

import posggym.envs.grid_world.render as render_lib
import posggym.model as M
from posggym.core import DefaultEnv
from posggym.envs.grid_world.core import ContinousWorld, Object, Position
from posggym.utils import seeding
import math
class PPState(NamedTuple):
    """A state in the Predator-Prey Environment."""

    predator_coords: Tuple[Position, ...]
    prey_coords: Tuple[Position, ...]
    prey_caught: Tuple[int, ...]


# Actions
PPAction = float

PREY = 0
PREDATOR = 1
AGENT_TYPE = [PREDATOR, PREY] 


# Observations
# Obs = (adj_obs)
PPObs = Tuple[float, ...]
# Cell Obs

collision_distance = 5

class PPContinousEnv(DefaultEnv[PPState, PPObs, PPAction]):
    """The Predator-Prey Grid World Environment.

    A co-operative 2D grid world problem involving multiple predator agents
    working together to catch prey agent/s in the environment.

    Agents
    ------
    Varied number

    State
    -----
    Each state consists of:

    1. tuple of the (x, y) position of all predators
    2. tuple of the (x, y) position of all preys
    3. tuple of whether each prey has been caught or not (0=no, 1=yes)

    For the coordinate x=column, y=row, with the origin (0, 0) at the
    top-left square of the grid.

    Actions
    -------
    Each agent has 5 actions: DO_NOTHING=0, UP=1, DOWN=2, LEFT=3, RIGHT=4

    Observation
    -----------
    Each agent observes the contents of local cells. The size of the
    local area observed is controlled by the `obs_dims` parameter. For each
    cell in the observed are the agent observes whether they are one of four
    things: EMPTY=0, WALL=1, PREDATOR=2, PREY=3.

    Reward
    ------
    There are two modes of play:

    1. Fully cooperative: All predators share a reward and each agent recieves
    a reward of 1.0 / `num_prey` for each prey capture, independent of which
    predator agent/s were responsible for the capture.

    2. Mixed cooperative: Predators only recieve a reward if they were part
    of the prey capture, recieving 1.0 / `num_prey`.

    In both modes prey can only been captured when at least `prey_strength`
    predators are in adjacent cells,
    where 1 <= `prey_strength` <= `num_predators`.

    Transition Dynamics
    -------------------
    Actions of the predator agents are deterministic and consist of moving in
    to the adjacent cell in each of the four cardinal directions. If two or
    more predators attempt to move into the same cell then no agent moves.

    Prey move according to the following rules (in order of priority):

    1. if predator is within `obs_dim` cells, moves away from closest predator
    2. if another prey is within `obs_dim` cells, moves away from closest prey
    3. else move randomly

    Prey always move first and predators and prey cannot occupy the same cell.
    The only exception being if a prey has been caught their final coord is
    recorded in the state but predator and prey agents will be able to move
    into the final coord.

    Episodes ends when all prey have been captured or the episode step limit is
    reached.

    Initial Conditions
    ------------------
    Predators start from random seperate locations along the edge of the grid
    (either in a corner, or half-way along a side), while prey start together
    in the middle.

    Reference
    ---------
    - Ming Tan. 1993. Multi-Agent Reinforcement Learning: Independent vs. Cooperative
      Agents. In Proceedings of the Tenth International Conference on Machine Learning.
      330–337.
    - J. Z. Leibo, V. F. Zambaldi, M. Lanctot, J. Marecki, and T. Graepel. 2017.
      Multi-Agent Reinforcement Learning in Sequential Social Dilemmas. In AAMAS,
      Vol. 16. ACM, 464–473

    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 4,
    }

    def __init__(
        self,
        num_predators: int,
        num_prey: int,
        cooperative: bool,
        prey_strength: int,
        obs_dim: int,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            PPModel(
                num_predators,
                num_prey,
                cooperative,
                prey_strength,
                obs_dim,
                **kwargs,
            ),
            render_mode=render_mode,
        )
        self._obs_dim = obs_dim
        self._viewer = None
        self._renderer: Optional[render_lib.GWContinousRender] = None
        self.render_mode = render_mode

    def render(self):
        if self.render_mode == "human":
            import posggym.envs.grid_world.render as render_lib

            if self._renderer is None:
                # print("recreate")
                # input()
                self._renderer = render_lib.GWContinousRender(
                    self.render_mode,
                    self.model.grid,
                    render_fps=self.metadata["render_fps"],
                    env_name="PredatorPreyContinous",
                    domain_size=self.model.grid.width,
                    num_colors=2
                )
            colored_prey =  tuple(t + (0,) for t in self._state.prey_coords)
            colored_pred =  tuple(t + (1,) for t in self._state.predator_coords)
            self._renderer.render(colored_prey + colored_pred)





    def close(self) -> None:
        pass

class PPModel(M.POSGModel[PPState, PPObs, PPAction]):
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

    def __init__(
        self,
        # grid: Union[str, "PPGrid"],
        num_predators: int,
        num_prey: int,
        cooperative: bool,
        prey_strength: int,
        obs_dim: int,
        **kwargs,
    ):
        # if isinstance(grid, str):
        # grid = load_grid("5x5")

        # print(num_predators)

        assert 1 < num_predators <= 8
        assert 0 < num_prey
        assert 0 < obs_dim
        assert 0 < prey_strength <= min(4, num_predators)
        # assert grid.prey_start_coords is None or len(grid.prey_start_coords) >= num_prey

        self.grid = PPWorld(grid_size=6, block_coords=None)
        self.obs_dim = obs_dim
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.cooperative = cooperative
        self.prey_strength = prey_strength
        self._per_prey_reward = self.R_MAX / self.num_prey

        # if self.grid.prey_start_coords is None:
        center_coords = self.grid.get_unblocked_center_coords(num_prey)
        self.grid.prey_start_coords = center_coords

        def _coord_space():
            return [spaces.Box(low=0, high=self.grid.width), spaces.Box(low=0, high=self.grid.height), spaces.Box(low=0, high=math.pi*2)]

        def _coord_space2():
            return spaces.Tuple((spaces.Box(low=0, high=self.grid.width), spaces.Box(low=0, high=self.grid.height), spaces.Box(low=0, high=math.pi*2)))
        
        self.possible_agents = tuple(range(self.num_predators))
        self.state_space = spaces.Tuple(
            (
                # coords of each agent
                spaces.Tuple(tuple(_coord_space2() for _ in range(self.num_predators))),
                # prey coords
                spaces.Tuple(tuple(_coord_space2() for _ in range(self.num_prey))),
                # prey caught/not
                spaces.Tuple(tuple(spaces.Discrete(2) for _ in range(self.num_prey))),
            )
        )

        self.action_spaces = {
            i: spaces.Box(low=-1.0, high=1.0) for i in self.possible_agents
        }

        # Observe everyones location
        nested_space = [_coord_space() for _ in range(self.num_predators + self.num_prey)]                     
        agent_obs = spaces.Tuple(tuple([item for sublist in nested_space for item in sublist]))


        self.observation_spaces = {
            i: agent_obs
            for i in self.possible_agents
        }
        self.observation_first = True
        self.is_symmetric = True

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[SupportsFloat, SupportsFloat]]:
        return {i: (0.0, self.R_MAX) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: PPState) -> List[M.AgentID]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> PPState:

        assert self.grid.prey_start_coords is not None

        predator_coords = [*self.grid.predator_start_coords]
        self.rng.shuffle(predator_coords)
        predator_coords = predator_coords[: self.num_predators]

        prey_coords = set(self.grid.prey_start_coords)
        prey_coords.difference_update(predator_coords)
        prey_coords_list = list(prey_coords)
        self.rng.shuffle(prey_coords_list)
        prey_coords_list = prey_coords_list[: self.num_prey]

        prey_caught = (0,) * self.num_prey

        return PPState(tuple(predator_coords), tuple(prey_coords_list), prey_caught)

    def sample_initial_obs(self, state: PPState) -> Dict[M.AgentID, PPObs]:
        # print(state, state)
        return self._get_obs(state, state)

    def step(
        self, state: PPState, actions: Dict[M.AgentID, PPAction]
    ) -> M.JointTimestep[PPState, PPObs]:

        next_state = self._get_next_state(state, actions)
        obs = self._get_obs(state, next_state)
        rewards = self._get_rewards(state, next_state)

        # print(rewards)

        all_done = all(next_state.prey_caught)
        truncated = {i: False for i in self.possible_agents}
        terminated = {i: all_done for i in self.possible_agents}

        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}
        if all_done:
            for i in self.possible_agents:
                info[i]["outcome"] = M.Outcome.WIN

        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, info
        )

    def _get_next_state(
        self, state: PPState, actions: Dict[M.AgentID, PPAction]
    ) -> PPState:
        # prey move first
        prey_coords = self._get_next_prey_state(state)
        predator_coords = self._get_next_predator_state(state, actions, prey_coords)
        prey_caught = self._get_next_prey_caught(state, prey_coords, predator_coords)
        return PPState(predator_coords, prey_coords, prey_caught)

    def _get_next_prey_state(self, state: PPState) -> Tuple[Position, ...]:
        next_prey_coords: List[Optional[Position]] = [None] * self.num_prey
        occupied_coords = set(
            state.predator_coords
            + tuple(
                state.prey_coords[i]
                for i in range(self.num_prey)
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
                prey_coord, state.predator_coords, list(occupied_coords)
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
                    prey_coord, state, list(occupied_coords)
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
                next_prey_coords[i] = self.rng.choice(neighbours)
            else:
                # possibility for collision between random moving prey
                neighbours = [c for c in neighbours if c not in occupied_coords]
                if len(neighbours) == 0:
                    next_coord = prey_coord
                else:
                    next_coord = self.rng.choice(neighbours)
                next_prey_coords[i] = next_coord
                occupied_coords.remove(prey_coord)
                occupied_coords.add(next_coord)

        return tuple(next_prey_coords)  # type: ignore

    def _move_away_from_predators(
        self,
        prey_coord: Position,
        predator_coords: Tuple[Position, ...],
        occupied_coords: List[Position],
    ) -> Optional[Position]:
        # get any predators within obs distance
        predator_dists = [
            self.grid.manhattan_dist(prey_coord, c) for c in predator_coords
        ]
        min_predator_dist = min(predator_dists)
        all_closest_predator_coords = [
            predator_coords[i]
            for i in range(self.num_predators)
            if predator_dists[i] == min_predator_dist
        ]
        closest_predator_coord = self.rng.choice(all_closest_predator_coords)

        if all(
            abs(a - b) > self.obs_dim
            for a, b in zip(prey_coord, closest_predator_coord)
        ):
            # closes predator out of obs range
            return None

        # move into furthest away free cell, includes current coord
        neighbours = list(
            (self.grid.manhattan_dist(c, closest_predator_coord), c)
            for c in self.grid.get_neighbours(prey_coord) + [prey_coord]
        )


        neighbours.sort()
        for (d, c) in reversed(neighbours):
            if c == prey_coord or self._coord_available_for_prey(c, occupied_coords):
                return c

        raise AssertionError("Something has gone wrong, please investigate.")

    def _move_away_from_preys(
        self, prey_coord: Position, state: PPState, occupied_coords: List[Position]
    ) -> Optional[Position]:
        prey_dists = [
            self.grid.manhattan_dist(prey_coord, c)
            if (c != prey_coord and not state.prey_caught[i])
            else float("inf")
            for i, c in enumerate(state.prey_coords)
        ]
        min_prey_dist = min(prey_dists)
        all_closest_prey_coords = [
            state.prey_coords[i]
            for i in range(self.num_prey)
            if prey_dists[i] == min_prey_dist
        ]
        closest_prey_coord = self.rng.choice(all_closest_prey_coords)

        if all(
            abs(a - b) > self.obs_dim for a, b in zip(prey_coord, closest_prey_coord)
        ):
            # closes predator out of obs range
            return None

        # move into furthest away free cell, includes current coord
        neighbours = list(
            (self.grid.manhattan_dist(c, closest_prey_coord), c)
            for c in self.grid.get_neighbours(prey_coord) + [prey_coord]
        )
        neighbours.sort()
        for (d, c) in reversed(neighbours):
            if c == prey_coord or self._coord_available_for_prey(c, occupied_coords):
                return c

        raise AssertionError("Something has gone wrong, please investigate.")

    def _coord_available_for_prey(
        self, coord: Position, occupied_coords: List[Position]
    ) -> bool:
        if coord in occupied_coords:
            return False
        neighbours = self.grid.get_neighbours(
            coord, ignore_blocks=False, include_out_of_bounds=False
        )
        for c in neighbours:
            for c2 in occupied_coords:
                if self.grid.agents_collide(c, c2):
                    neighbours.remove(c)

        return len(neighbours) >= self.prey_strength

    def _get_next_predator_state(
        self,
        state: PPState,
        actions: Dict[M.AgentID, PPAction],
        next_prey_coords: Tuple[Position, ...],
    ) -> Tuple[Position, ...]:
        
        potential_next_coords = []
        for i, coord in enumerate(state.predator_coords):

            next_coord = self.grid.get_next_coord(
                coord, actions[i][0], ignore_blocks=False  # type: ignore
            )


            if self.grid.check_collision((coord, self.grid.agent_size)):
                next_coord = coord
                # print("in collision")
            potential_next_coords.append(next_coord)

        # handle collisions
        next_coords = []
        for i in range(self.num_predators):
            coord_i = potential_next_coords[i]
            collision = False
            for j in range(self.num_predators):
                if i == j:
                    continue
                elif self.grid.agents_collide(coord_i, potential_next_coords[j]):
                    # print("in collision2")
                    collision = True
                    break
            if collision:
                next_coords.append(state.predator_coords[i])
            else:
                next_coords.append(coord_i)

        return tuple(next_coords)

    def _get_next_prey_caught(
        self,
        state: PPState,
        next_prey_coords: Tuple[Position, ...],
        next_predator_coords: Tuple[Position, ...],
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
                num_adj_predators = sum(d <= collision_distance for d in predator_dists)
                # print(predator_dists)
                prey_caught.append(int(num_adj_predators >= self.prey_strength))
        return tuple(prey_caught)

    def _get_obs(self, state: PPState, next_state: PPState) -> Dict[M.AgentID, PPObs]:

        a = {
            i: self._get_local_obs(i, state, next_state)
            for i in self.possible_agents
        }
        print("a[0]", a[0])
        return a

    def _get_local_obs(
        self, agent_id: M.AgentID, state: PPState, next_state: PPState
    ) -> Tuple[float, ...]:
        assert isinstance(agent_id, int)

        obs : List[float] = []
        
        obs += state.predator_coords[agent_id]
        for i in range(len(state.predator_coords)):
            if i == agent_id:
                continue
            obs += state.predator_coords[i]
        
        for i in range(len(state.prey_coords)):
            # print(state.prey_coords)
            obs += state.prey_coords[i]

        return tuple(obs)

    def _get_rewards(
        self, state: PPState, next_state: PPState
    ) -> Dict[M.AgentID, SupportsFloat]:
        new_caught_prey = []
        for i in range(self.num_prey):
            if not state.prey_caught[i] and next_state.prey_caught[i]:
                new_caught_prey.append(next_state.prey_coords[i])

        if len(new_caught_prey) == 0:
            return {i: 0.0 for i in self.possible_agents}

        if self.cooperative:
            reward = len(new_caught_prey) * (self._per_prey_reward)
            return {i: reward for i in self.possible_agents}

        rewards = {i: 0.0 for i in self.possible_agents}
        for prey_coord in new_caught_prey:
            involved_predators = []
            for index, predator in enumerate(next_state.predator_coords):
                if self.grid.manhattan_dist(predator, prey_coord) < collision_distance:
                    involved_predators.append(index)
                                         
            predator_reward = self._per_prey_reward / len(involved_predators)
            for i in involved_predators:
                rewards[i] += predator_reward

        return rewards  # type: ignore

import random
class PPWorld(ContinousWorld):
    """A grid for the Predator-Prey Problem."""

    def __init__(
        self,
        grid_size: int,
        block_coords: Optional[List[Object]],
        predator_start_coords: Optional[List[Position]] = None,
        prey_start_coords: Optional[List[Position]] = None,
        predator_angles: Optional[List[float]] = None,
    ):
        assert grid_size >= 3
        super().__init__(grid_size, grid_size, block_coords)
        self.size = grid_size
        # predators start in corners or half-way along a side
        if predator_start_coords is None:
            predator_start_coords_ : List[Tuple[float, float]] = list(
                (float(c[0]), float(c[1]))  # type: ignore
                for c in product([0, grid_size // 2, grid_size - 1], repeat=2)
                if c[0] in (0, grid_size - 1) or c[1] in (0, grid_size - 1)
            )
            predator_angles = [random.random() * math.pi for _ in range(len(predator_start_coords_))]

            predator_start_coords = [x + (y,) for x,y in zip(predator_start_coords_, predator_angles)]

        self.predator_start_coords = predator_start_coords
        self.prey_start_coords = prey_start_coords


    def get_unblocked_center_coords(self, num: int) -> List[Position]:
        """Get at least num closest coords to the center of grid.

        May return more than num, since can be more than one coord at equal
        distance from the center.
        """
        # assert num < self.n_coords - len(self.block_coords)
        center = (self.width / 2, self.height / 2, 0)
        min_dist_from_center = math.ceil(math.sqrt(num)) - 1

        coords = [self.sample_coords_within_dist(center, min_dist_from_center, ignore_blocks=False) for _ in range(num)]

        return coords
