"""The Predator-Prey Grid World Environment."""

import math
from itertools import product
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

from gymnasium import spaces

import posggym.model as M
from posggym import logger
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


class PredatorPreyEnv(DefaultEnv[PPState, PPObs, PPAction]):
    """The Predator-Prey Grid World Environment.

    A co-operative 2D grid world problem involving multiple predator agents
    working together to catch prey agent/s in the environment.

    Possible Agents
    ---------------
    From two to eight agents, with all agents always active.

    State Space
    -----------
    Each state consists of:

    1. tuple of the `(x, y)` position of all predators
    2. tuple of the `(x, y)` position of all preys
    3. tuple of whether each prey has been caught or not (`0=no`, `1=yes`)

    For the coordinate `x`=column, `y`=row, with the origin (0, 0) at the top-left
    square of the grid.

    Action Space
    ------------
    Each agent has 5 actions: `DO_NOTHING=0`, `UP=1`, `DOWN=2`, `LEFT=3`, `RIGHT=4`

    Observation Space
    -----------------
    Each agent observes the contents of cells in a local area aroun the agent. The size
    of the local area observed is controlled by the `obs_dims` parameter which specifies
    how many cells in each direction is observed (the default value is `2`, which means
    the agent observes a 5x5 area). For each observed cell the agent receives one of
    of four values depending on the contents of the cell: `EMPTY=0`, `WALL=1`,
    `PREDATOR=2`, `PREY=3`.

    Rewards
    -------
    There are two modes of play:

    1. *Fully cooperative*: All predators share a reward and each agent receives a
    reward of `1.0 / num_prey` for each prey capture, independent of which predator
    agent/s were responsible for the capture.
    2. *Mixed cooperative*: Predators only receive a reward if they were part of the
    prey capture, receiving `1.0 / num_prey` for each prey capture they were apart of.

    Dynamics
    --------
    Actions of the predator agents are deterministic and consist of moving to the
    adjacent cell in each of the four cardinal directions. If two or more predators
    attempt to move into the same cell then no agent moves.

    Prey move according to the following rules (in order of priority):

    1. if predator is within `obs_dim` cells, moves away from closest predator
    2. if another prey is within `obs_dim` cells, moves away from closest prey
    3. else move randomly

    Prey always move first and predators and prey cannot occupy the same cell.
    The only exception being if a prey has been caught their final coordinate is
    recorded in the state but predator and still alive prey will be able to move into
    the final coordinate of caught prey.

    Prey are captured when at least `prey_strength` predators are in adjacent cells,
    where `1 <= prey_strength <= min(4, num_predators)`.

    Starting State
    --------------
    Predators start from random separate locations along the edge of the grid
    (either in a corner, or half-way along a side), while prey start together
    in the middle.

    Episodes End
    ------------
    Episodes ends when all prey have been captured. By default a `max_episode_steps`
    limit of `50` steps is also set. This may need to be adjusted when using larger
    grids (this can be done by manually specifying a value for `max_episode_steps` when
    creating the environment with `posggym.make`).

    Arguments
    ---------

    - `grid` - the grid layout to use. This can either be a string specifying one of
        the supported grids, or a custom :class:`PredatorPreyGrid` object
        (default = `"10x10"`).
    - `num_predators` - the number of predator (and thus controlled agents)
        (default = `2`).
    - `num_prey` - the number of prey (default = `3`)
    - `cooperative` - whether agents share all rewards or only get rewards for prey they
        are involved in capturing (default = 'True`)
    - `prey_strength` - how many predators are required to capture each prey, minimum is
        `1` and maximum is `min(4, num_predators)`. If `None` this is set to
        `min(4, num_predators)` (default = 'None`)
    - `obs_dim` - the local observation dimensions, specifying how many cells in each
        direction each predator and prey agent observes (default = `2`, resulting in
        the agent observing a 5x5 area)

    Available variants
    ------------------
    The PredatorPrey environment comes with a number of pre-built grid layouts which can
    be passed as an argument to `posggym.make`, to create different grids. All layouts
    support 2 to 8 agents.

    | Grid name         | Grid size |
    |-------------------|-----------|
    | `5x5`             | 5x5       |
    | `5x5Blocks`       | 5x5       |
    | `10x10`           | 10x10     |
    | `10x10Blocks`     | 10x10     |
    | `15x15`           | 15x15     |
    | `15x15Blocks`     | 15x15     |
    | `20x20`           | 20x20     |
    | `20x20Blocks`     | 20x20     |


    For example to use the Predator Prey environment with the `15x15Blocks` grid, 4
    predators, 4 prey, and episode step limit of 100, and the default values for the
    other parameters (`cooperative`, `obs_dim`, `prey_strength`) you would use:

    ```python
    import posggym
    env = posggym.make(
        'PredatorPrey-v0',
        max_episode_steps=100,
        grid="15x15Blocks",
        num_predators=4,
        num_prey=4
    )
    ```

    Version History
    ---------------
    - `v0`: Initial version

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
        "render_fps": 15,
    }

    def __init__(
        self,
        grid: Union[str, "PredatorPreyGrid"] = "10x10",
        num_predators: int = 2,
        num_prey: int = 3,
        cooperative: bool = True,
        prey_strength: Optional[int] = None,
        obs_dim: int = 2,
        render_mode: Optional[str] = None,
    ):
        super().__init__(
            PredatorPreyModel(
                grid,
                num_predators,
                num_prey,
                cooperative,
                prey_strength,
                obs_dim,
            ),
            render_mode=render_mode,
        )
        self._obs_dim = obs_dim
        self.renderer = None
        self._predator_imgs = None
        self._prey_imgs = None

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. posggym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.render_mode == "ansi":
            return self._render_ansi()
        return self._render_img()

    def _render_ansi(self):
        model: PredatorPreyModel = self.model  # type: ignore
        grid = model.grid
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

    def _render_img(self):
        model: PredatorPreyModel = self.model  # type: ignore

        import posggym.envs.grid_world.render as render_lib

        if self.renderer is None:
            self.renderer = render_lib.GWRenderer(
                self.render_mode,
                model.grid,
                render_fps=self.metadata["render_fps"],
                env_name="PredatorPrey",
            )

        if self._predator_imgs is None:
            img_path = Path(__file__).parent / "img" / "robot.png"
            agent_img = render_lib.load_img_file(img_path, self.renderer.cell_size)
            self._predator_imgs = {
                i: render_lib.GWImage((0, 0), self.renderer.cell_size, agent_img)
                for i in self.possible_agents
            }

        observed_coords = []
        for coord in self._state.predator_coords:
            observed_coords.extend(model.get_obs_coords(coord))

        render_objects = []
        # Add prey
        for i, coord in enumerate(self._state.prey_coords):
            if self._state.prey_caught[i]:
                continue
            render_objects.append(
                render_lib.GWCircle(
                    coord, self.renderer.cell_size, render_lib.get_color("cyan")
                )
            )

        # Add agents
        for i, coord in enumerate(self._state.predator_coords):
            agent_obj = self._predator_imgs[str(i)]
            agent_obj.coord = coord
            render_objects.append(agent_obj)

        agent_coords_and_dirs = {
            str(i): (coord, Direction.NORTH)
            for i, coord in enumerate(self._state.predator_coords)
        }

        if self.render_mode in ("human", "rgb_array"):
            return self.renderer.render(render_objects, observed_coords)
        return self.renderer.render_agents(
            render_objects,
            agent_coords_and_dirs,
            agent_obs_dims=self._obs_dim,
            observed_coords=observed_coords,
        )

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class PredatorPreyModel(M.POSGModel[PPState, PPObs, PPAction]):
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
    prey_strenth : int, optional
        how many predators are required to capture each prey, minimum is `1` and maximum
        is `min(4, num_predators)`. If `None` this is set to `min(4, num_predators)`
    obs_dims : int
        number of cells in each direction around the agent that the agent can
        observe

    """

    R_MAX = 1.0
    PREY_CAUGHT_COORD = (0, 0)

    def __init__(
        self,
        grid: Union[str, "PredatorPreyGrid"],
        num_predators: int,
        num_prey: int,
        cooperative: bool,
        prey_strength: Optional[int],
        obs_dim: int,
    ):
        if isinstance(grid, str):
            assert grid in SUPPORTED_GRIDS, (
                f"Unsupported grid name '{grid}'. Grid name must be one of: "
                f"{SUPPORTED_GRIDS.keys()}."
            )
            grid = SUPPORTED_GRIDS[grid][0]()

        if prey_strength is None:
            prey_strength = min(4, num_predators)

        assert 1 < num_predators <= 8
        assert num_prey > 0
        assert obs_dim > 0
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

        self.possible_agents = tuple(str(i) for i in range(self.num_predators))
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
        self.is_symmetric = True

    @property
    def reward_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {i: (0.0, self.R_MAX) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, _ = seeding.std_random()
        return self._rng

    def get_agents(self, state: PPState) -> List[str]:
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

    def sample_initial_obs(self, state: PPState) -> Dict[str, PPObs]:
        return self._get_obs(state, state)

    def step(
        self, state: PPState, actions: Dict[str, PPAction]
    ) -> M.JointTimestep[PPState, PPObs]:
        next_state = self._get_next_state(state, actions)
        obs = self._get_obs(state, next_state)
        rewards = self._get_rewards(state, next_state)

        all_done = all(next_state.prey_caught)
        truncated = {i: False for i in self.possible_agents}
        terminated = {i: all_done for i in self.possible_agents}

        info: Dict[str, Dict] = {i: {} for i in self.possible_agents}
        if all_done:
            for i in self.possible_agents:
                info[i]["outcome"] = M.Outcome.WIN

        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, info
        )

    def _get_next_state(self, state: PPState, actions: Dict[str, PPAction]) -> PPState:
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
        neighbours = [
            (self.grid.manhattan_dist(c, closest_predator_coord), c)
            for c in self.grid.get_neighbours(prey_coord) + [prey_coord]
        ]
        neighbours.sort()
        for d, c in reversed(neighbours):
            if c == prey_coord or self._coord_available_for_prey(c, occupied_coords):
                return c

        raise AssertionError("Something has gone wrong, please investigate.")

    def _move_away_from_preys(
        self, prey_coord: Coord, state: PPState, occupied_coords: Set[Coord]
    ) -> Optional[Coord]:
        prey_dists = [
            (
                self.grid.manhattan_dist(prey_coord, c)
                if (c != prey_coord and not state.prey_caught[i])
                else float("inf")
            )
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
        neighbours = [
            (self.grid.manhattan_dist(c, closest_prey_coord), c)
            for c in self.grid.get_neighbours(prey_coord) + [prey_coord]
        ]
        neighbours.sort()
        for d, c in reversed(neighbours):
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
        actions: Dict[str, PPAction],
        next_prey_coords: Tuple[Coord, ...],
    ) -> Tuple[Coord, ...]:
        potential_next_coords = []
        occupied_prey_coords = {
            c for i, c in enumerate(next_prey_coords) if not state.prey_caught[i]
        }
        for i, coord in enumerate(state.predator_coords):
            if actions[str(i)] == 0:
                next_coord = coord
            else:
                a_dir = ACTION_TO_DIR[actions[str(i)]]
                next_coord = self.grid.get_next_coord(
                    coord, a_dir, ignore_blocks=False  # type: ignore
                )
                if next_coord in occupied_prey_coords:
                    next_coord = coord
            potential_next_coords.append(next_coord)

        # handle collisions
        next_coords = []
        for i in range(self.num_predators):
            coord_i = potential_next_coords[i]
            for j in range(self.num_predators):
                if i == j:
                    continue
                elif coord_i == potential_next_coords[j]:
                    # collision, stay in current cell
                    coord_i = state.predator_coords[i]
                    break
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

    def _get_obs(self, state: PPState, next_state: PPState) -> Dict[str, PPObs]:
        return {
            i: self._get_local_cell__obs(int(i), state, next_state)
            for i in self.possible_agents
        }

    def _get_local_cell__obs(
        self, agent_idx: int, state: PPState, next_state: PPState
    ) -> Tuple[int, ...]:
        obs_size = (2 * self.obs_dim) + 1
        agent_coord = next_state.predator_coords[agent_idx]

        cell_obs = []
        for row, col in product(range(obs_size), repeat=2):
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

    def _get_rewards(self, state: PPState, next_state: PPState) -> Dict[str, float]:
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
                rewards[str(i)] += predator_reward
        return rewards  # type: ignore


class PredatorPreyGrid(Grid):
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
            predator_start_coords = [
                c
                for c in product([0, grid_size // 2, grid_size - 1], repeat=2)
                if c[0] in (0, grid_size - 1) or c[1] in (0, grid_size - 1)
            ]
        self.predator_start_coords: List[Coord] = predator_start_coords
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

        return str(self) + "\n" + "\n".join([" ".join(r) for r in grid_repr])

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


def parse_grid_str(grid_str: str) -> PredatorPreyGrid:
    """Parse a str representation of a grid into a grid object.

    Notes on grid str representation:

    . = empty/unallocated cell
    # = a block
    P = starting location for predator agents [optional] (defaults to edges)
    p = starting location for prey agent [optional] (defaults to center)

    Examples (" " quotes and newline chars omitted):

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

    return PredatorPreyGrid(
        grid_size,
        block_coords,
        None if len(predator_coords) == 0 else list(predator_coords),
        None if len(prey_coords) == 0 else list(prey_coords),
    )


def get_5x5_grid() -> PredatorPreyGrid:
    """Generate 5x5 grid layout."""
    return PredatorPreyGrid(grid_size=5, block_coords=None)


def get_5x5_blocks_grid() -> PredatorPreyGrid:
    """Generate 5x5 Blocks grid layout."""
    grid_str = ".....\n" ".#.#.\n" ".....\n" ".#.#.\n" ".....\n"
    return parse_grid_str(grid_str)


def get_10x10_grid() -> PredatorPreyGrid:
    """Generate 10x10 grid layout."""
    return PredatorPreyGrid(grid_size=10, block_coords=None)


def get_10x10_blocks_grid() -> PredatorPreyGrid:
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


def get_15x15_grid() -> PredatorPreyGrid:
    """Generate 15x15 grid layout."""
    return PredatorPreyGrid(grid_size=15, block_coords=None)


def get_15x15_blocks_grid() -> PredatorPreyGrid:
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


def get_20x20_grid() -> PredatorPreyGrid:
    """Generate 20x20 grid layout."""
    return PredatorPreyGrid(grid_size=20, block_coords=None)


def get_20x20_blocks_grid() -> PredatorPreyGrid:
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


def load_grid(grid_name: str) -> PredatorPreyGrid:
    """Load grid with given name."""
    grid_name = grid_name
    assert grid_name in SUPPORTED_GRIDS, (
        f"Unsupported grid name '{grid_name}'. Grid name must be one of: "
        f"{SUPPORTED_GRIDS.keys()}."
    )
    return SUPPORTED_GRIDS[grid_name][0]()
