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
    Sequence,
    Set,
    SupportsFloat,
    Tuple,
    Union,
)

from gymnasium import spaces

import posggym.envs.grid_world.render as render_lib
import posggym.model as M
from posggym.core import DefaultEnv
from posggym.envs.grid_world.core import Coord, Direction, Grid
from posggym.utils import seeding


class PPState(NamedTuple):
    """A state in the Predator-Prey Environment."""

    predator_coords: Tuple[Coord, ...]
    prey_coords: Tuple[Coord, ...]
    prey_caught: Tuple[int, ...]


# Actions
PPAction = int
DO_NOTHING = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
ACTIONS = [DO_NOTHING, UP, DOWN, LEFT, RIGHT]
ACTIONS_STR = ["0", "U", "D", "L", "R"]
ACTION_TO_DIR = [None, Direction.NORTH, Direction.SOUTH, Direction.WEST, Direction.EAST]

# Observations
# Obs = (adj_obs)
PPObs = Tuple[int, ...]
# Cell Obs
EMPTY = 0
WALL = 1
PREDATOR = 2
PREY = 3
CELL_OBS = [EMPTY, WALL, PREDATOR, PREY]
CELL_OBS_STR = ["0", "#", "P", "p"]


class PPEnv(DefaultEnv[PPState, PPObs, PPAction]):
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
        "render_modes": ["human", "ansi", "rgb_array", "rgb_array_dict"],
        "render_fps": 4,
    }

    def __init__(
        self,
        grid: Union[str, "PPGrid"],
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
                grid,
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
        self._renderer: Optional[render_lib.GWRenderer] = None

    def render(self):
        # cast for typehinting
        model: PPModel = self.model  # type: ignore
        grid = model.grid
        if self.render_mode == "ansi":
            uncaught_prey_coords = [
                self._state.prey_coords[i]
                for i in range(self.model.num_prey)
                if not self._state.prey_caught[i]
            ]
            grid_str = grid.get_ascii_repr(
                self._state.predator_coords, uncaught_prey_coords
            )
            output = [
                f"Step: {self._step_num}",
                grid_str,
            ]
            if self._last_actions is not None:
                action_str = ", ".join(
                    [ACTIONS_STR[a] for a in self._last_actions.values()]
                )
                output.insert(1, f"Actions: <{action_str}>")
                output.append(f"Rewards: <{self._last_rewards}>")

            return "\n".join(output) + "\n"

        if self.render_mode == "human" and self._viewer is None:
            # pylint: disable=[import-outside-toplevel]
            from posggym.envs.grid_world import viewer

            self._viewer = viewer.GWViewer(
                "Predator-Prey Env",
                (min(grid.width, 9), min(grid.height, 9)),
                num_agent_displays=len(self.possible_agents),
            )
            self._viewer.show(block=False)

        if self._renderer is None:
            self._renderer = render_lib.GWRenderer(
                len(self.possible_agents), grid, [], render_blocks=True
            )

        agent_obs_coords = tuple(
            self.model.get_obs_coords(c) for c in self._state.predator_coords
        )
        agent_coords = self._state.predator_coords

        prey_objs = [
            render_lib.GWObject(
                c,
                "cyan",
                render_lib.Shape.CIRCLE,
                # alpha=0.25
            )
            for i, c in enumerate(self._state.prey_coords)
            if not self._state.prey_caught[i]
        ]

        env_img = self._renderer.render(
            agent_coords,
            agent_obs_coords,
            agent_dirs=None,
            other_objs=prey_objs,
            agent_colors=None,
        )
        agent_obs_imgs = self._renderer.render_all_agent_obs(
            env_img,
            agent_coords,
            Direction.NORTH,
            agent_obs_dims=(self._obs_dim, self._obs_dim, self._obs_dim),
            out_of_bounds_obj=render_lib.GWObject(
                (0, 0), "grey", render_lib.Shape.RECTANGLE
            ),
        )

        if self.render_mode == "human":
            self._viewer.update_img(env_img, agent_idx=None)
            for i, obs_img in enumerate(agent_obs_imgs):
                self._viewer.update_img(obs_img, agent_idx=i)
            self._viewer.display_img()
        elif self.render_mode == "rgb_array":
            return env_img
        else:
            # rgb_array_dict
            return dict(enumerate(agent_obs_imgs))

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()  # type: ignore
            self._viewer = None


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
        grid: Union[str, "PPGrid"],
        num_predators: int,
        num_prey: int,
        cooperative: bool,
        prey_strength: int,
        obs_dim: int,
        **kwargs,
    ):
        if isinstance(grid, str):
            grid = load_grid(grid)

        assert 1 < num_predators <= 8
        assert 0 < num_prey
        assert 0 < obs_dim
        assert 0 < prey_strength <= min(4, num_predators)
        assert grid.prey_start_coords is None or len(grid.prey_start_coords) >= num_prey

        self.grid = grid
        self.obs_dim = obs_dim
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.cooperative = cooperative
        self.prey_strength = prey_strength
        self._per_prey_reward = self.R_MAX / self.num_prey

        if self.grid.prey_start_coords is None:
            center_coords = self.grid.get_unblocked_center_coords(num_prey)
            self.grid.prey_start_coords = center_coords

        def _coord_space():
            return spaces.Tuple(
                (spaces.Discrete(self.grid.width), spaces.Discrete(self.grid.height))
            )

        self.possible_agents = tuple(range(self.num_predators))
        self.state_space = spaces.Tuple(
            (
                # coords of each agent
                spaces.Tuple(tuple(_coord_space() for _ in range(self.num_predators))),
                # prey coords
                spaces.Tuple(tuple(_coord_space() for _ in range(self.num_prey))),
                # prey caught/not
                spaces.Tuple(tuple(spaces.Discrete(2) for _ in range(self.num_prey))),
            )
        )
        self.action_spaces = {
            i: spaces.Discrete(len(ACTIONS)) for i in self.possible_agents
        }
        self.observation_spaces = {
            i: spaces.Tuple(
                tuple(
                    spaces.Discrete(len(CELL_OBS))
                    for _ in range((2 * self.obs_dim + 1) ** 2)
                )
            )
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
        return self._get_obs(state, state)

    def step(
        self, state: PPState, actions: Dict[M.AgentID, PPAction]
    ) -> M.JointTimestep[PPState, PPObs]:
        next_state = self._get_next_state(state, actions)
        obs = self._get_obs(state, next_state)
        rewards = self._get_rewards(state, next_state)

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

    def _get_next_prey_state(self, state: PPState) -> Tuple[Coord, ...]:
        next_prey_coords: List[Optional[Coord]] = [None] * self.num_prey
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
        prey_coord: Coord,
        predator_coords: Tuple[Coord, ...],
        occupied_coords: Set[Coord],
    ) -> Optional[Coord]:
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
        self, prey_coord: Coord, state: PPState, occupied_coords: Set[Coord]
    ) -> Optional[Coord]:
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
        self, coord: Coord, occupied_coords: Set[Coord]
    ) -> bool:
        if coord in occupied_coords:
            return False
        neighbours = self.grid.get_neighbours(
            coord, ignore_blocks=False, include_out_of_bounds=False
        )
        for c in neighbours:
            if c in occupied_coords:
                neighbours.remove(c)
        return len(neighbours) >= self.prey_strength

    def _get_next_predator_state(
        self,
        state: PPState,
        actions: Dict[M.AgentID, PPAction],
        next_prey_coords: Tuple[Coord, ...],
    ) -> Tuple[Coord, ...]:
        potential_next_coords = []
        for i, coord in enumerate(state.predator_coords):
            if actions[i] == 0:
                next_coord = coord
            else:
                a_dir = ACTION_TO_DIR[actions[i]]
                next_coord = self.grid.get_next_coord(
                    coord, a_dir, ignore_blocks=False  # type: ignore
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

    def _get_next_prey_caught(
        self,
        state: PPState,
        next_prey_coords: Tuple[Coord, ...],
        next_predator_coords: Tuple[Coord, ...],
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
                prey_caught.append(int(num_adj_predators >= self.prey_strength))
        return tuple(prey_caught)

    def _get_obs(self, state: PPState, next_state: PPState) -> Dict[M.AgentID, PPObs]:
        return {
            i: self._get_local_cell__obs(i, state, next_state)
            for i in self.possible_agents
        }

    def _get_local_cell__obs(
        self, agent_id: M.AgentID, state: PPState, next_state: PPState
    ) -> Tuple[int, ...]:
        assert isinstance(agent_id, int)
        obs_size = (2 * self.obs_dim) + 1
        agent_coord = next_state.predator_coords[agent_id]

        cell_obs = []
        for col, row in product(range(obs_size), repeat=2):
            obs_grid_coord = self._map_obs_to_grid_coord((col, row), agent_coord)
            if obs_grid_coord is None or obs_grid_coord in self.grid.block_coords:
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

    def _map_obs_to_grid_coord(
        self, obs_coord: Coord, agent_coord: Coord
    ) -> Optional[Coord]:
        grid_col = agent_coord[0] + obs_coord[0] - self.obs_dim
        grid_row = agent_coord[1] + obs_coord[1] - self.obs_dim

        if 0 <= grid_row < self.grid.height and 0 <= grid_col < self.grid.width:
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
        return rewards  # type: ignore


class PPGrid(Grid):
    """A grid for the Predator-Prey Problem."""

    def __init__(
        self,
        grid_size: int,
        block_coords: Optional[Set[Coord]],
        predator_start_coords: Optional[List[Coord]] = None,
        prey_start_coords: Optional[List[Coord]] = None,
    ):
        assert grid_size >= 3
        super().__init__(grid_size, grid_size, block_coords)
        self.size = grid_size
        # predators start in corners or half-way along a side
        if predator_start_coords is None:
            predator_start_coords = list(
                c  # type: ignore
                for c in product([0, grid_size // 2, grid_size - 1], repeat=2)
                if c[0] in (0, grid_size - 1) or c[1] in (0, grid_size - 1)
            )
        self.predator_start_coords = predator_start_coords
        self.prey_start_coords = prey_start_coords

    def get_ascii_repr(
        self,
        predator_coords: Optional[Sequence[Coord]],
        prey_coords: Optional[Sequence[Coord]],
    ) -> str:
        """Get ascii repr of grid."""
        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                coord = (col, row)
                if coord in self.block_coords:
                    row_repr.append("#")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)

        if predator_coords is not None:
            for c in predator_coords:
                grid_repr[c[0]][c[1]] = "P"
        if prey_coords is not None:
            for c in prey_coords:
                grid_repr[c[0]][c[1]] = "p"

        return (
            str(self) + "\n" + "\n".join(list(list((" ".join(r) for r in grid_repr))))
        )

    def get_unblocked_center_coords(self, num: int) -> List[Coord]:
        """Get at least num closest coords to the center of grid.

        May return more than num, since can be more than one coord at equal
        distance from the center.
        """
        assert num < self.n_coords - len(self.block_coords)
        center = (self.width // 2, self.height // 2)
        min_dist_from_center = math.ceil(math.sqrt(num)) - 1
        coords = self.get_coords_within_dist(
            center, min_dist_from_center, ignore_blocks=False, include_origin=True
        )

        while len(coords) < num:
            # not the most efficient as it repeats work,
            # but function should only be called once when model is initialized
            # and for small num
            for c in coords:
                coords.update(
                    self.get_neighbours(
                        c, ignore_blocks=False, include_out_of_bounds=False
                    )
                )

        return list(coords)

    def num_unblocked_neighbours(self, coord: Coord) -> int:
        """Get number of neighbouring coords that are unblocked."""
        return len(
            self.get_neighbours(coord, ignore_blocks=False, include_out_of_bounds=False)
        )


def parse_grid_str(grid_str: str) -> PPGrid:
    """Parse a str representation of a grid into a grid object.

    Notes on grid str representation:

    . = empty/unallocated cell
    # = a block
    P = starting location for predator agents [optional] (defaults to edges)
    p = starting location for prey agent [optional] (defaults to center)

    Examples (" " quotes and newline chars ommited):

    1. A 10x10 grid with 4 groups of blocks and using the default predator
       and prey start locations.

    ..........
    ..........
    ..##..##..
    ..##..##..
    ..........
    ..........
    ..##..##..
    ..##..##..
    ..........
    ..........

    2. Same as above but with predator and prey start locations defined for
    up to 8 predators and 4 prey. (This would be the default layout for the
    scenario where there are between 2 and 4 prey, i.e. if prey and predator
    start locations were left unspecified as in example 1.)

    P....P...P
    ..........
    ..##..##..
    ..##..##..
    ....pp....
    P...pp...P
    ..##..##..
    ..##..##..
    ..........
    P....P...P

    """
    row_strs = grid_str.splitlines()
    assert len(row_strs) > 1
    assert all(len(row) == len(row_strs[0]) for row in row_strs)
    assert len(row_strs[0]) > 1
    assert len(row_strs) == len(row_strs[0])

    grid_size = len(row_strs)
    block_coords = set()
    predator_coords = set()
    prey_coords = set()
    for r, c in product(range(grid_size), repeat=2):
        coord = (c, r)
        char = row_strs[r][c]

        if char == "#":
            block_coords.add(coord)
        elif char == "P":
            predator_coords.add(coord)
        elif char == "p":
            prey_coords.add(coord)
        else:
            assert char == "."

    return PPGrid(
        grid_size,
        block_coords,
        None if len(predator_coords) == 0 else list(predator_coords),
        None if len(prey_coords) == 0 else list(prey_coords),
    )


def get_5x5_grid() -> PPGrid:
    """Generate 5x5 grid layout."""
    return PPGrid(grid_size=5, block_coords=None)


def get_5x5_blocks_grid() -> PPGrid:
    """Generate 5x5 Blocks grid layout."""
    grid_str = ".....\n" ".#.#.\n" ".....\n" ".#.#.\n" ".....\n"
    return parse_grid_str(grid_str)


def get_10x10_grid() -> PPGrid:
    """Generate 10x10 grid layout."""
    return PPGrid(grid_size=10, block_coords=None)


def get_10x10_blocks_grid() -> PPGrid:
    """Generate 10x10 Blocks grid layout."""
    grid_str = (
        "..........\n"
        "..........\n"
        "..##..##..\n"
        "..##..##..\n"
        "..........\n"
        "..........\n"
        "..##..##..\n"
        "..##..##..\n"
        "..........\n"
        "..........\n"
    )
    return parse_grid_str(grid_str)


def get_15x15_grid() -> PPGrid:
    """Generate 15x15 grid layout."""
    return PPGrid(grid_size=15, block_coords=None)


def get_15x15_blocks_grid() -> PPGrid:
    """Generate 10x10 Blocks grid layout."""
    grid_str = (
        "...............\n"
        "...............\n"
        "...............\n"
        "...###...###...\n"
        "...###...###...\n"
        "...###...###...\n"
        "...............\n"
        "...............\n"
        "...............\n"
        "...###...###...\n"
        "...###...###...\n"
        "...###...###...\n"
        "...............\n"
        "...............\n"
        "...............\n"
    )
    return parse_grid_str(grid_str)


def get_20x20_grid() -> PPGrid:
    """Generate 20x20 grid layout."""
    return PPGrid(grid_size=20, block_coords=None)


def get_20x20_blocks_grid() -> PPGrid:
    """Generate 20x20 Blocks grid layout."""
    grid_str = (
        "....................\n"
        "....................\n"
        "....................\n"
        "....................\n"
        "....####....####....\n"
        "....####....####....\n"
        "....####....####....\n"
        "....####....####....\n"
        "....................\n"
        "....................\n"
        "....................\n"
        "....................\n"
        "....####....####....\n"
        "....####....####....\n"
        "....####....####....\n"
        "....####....####....\n"
        "....................\n"
        "....................\n"
        "....................\n"
        "....................\n"
    )
    return parse_grid_str(grid_str)


#  (grid_make_fn, step_limit)
SUPPORTED_GRIDS = {
    "5x5": (get_5x5_grid, 25),
    "5x5Blocks": (get_5x5_blocks_grid, 50),
    "10x10": (get_10x10_grid, 50),
    "10x10Blocks": (get_10x10_blocks_grid, 50),
    "15x15": (get_15x15_grid, 100),
    "15x15Blocks": (get_15x15_blocks_grid, 100),
    "20x20": (get_20x20_grid, 200),
    "20x20Blocks": (get_20x20_blocks_grid, 200),
}


def load_grid(grid_name: str) -> PPGrid:
    """Load grid with given name."""
    grid_name = grid_name
    assert grid_name in SUPPORTED_GRIDS, (
        f"Unsupported grid name '{grid_name}'. Grid name must be one of: "
        f"{SUPPORTED_GRIDS.keys()}."
    )
    return SUPPORTED_GRIDS[grid_name][0]()
