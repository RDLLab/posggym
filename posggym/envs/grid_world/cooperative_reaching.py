"""The Cooperative Reaching Grid World Environment."""
from itertools import product
from os import path
from typing import Dict, List, Optional, Tuple

from gymnasium import spaces

import posggym.model as M
from posggym import logger
from posggym.core import DefaultEnv
from posggym.envs.grid_world.core import Coord, Direction, Grid
from posggym.utils import seeding


# State = (coord_0, coord_1)
CRState = Tuple[Coord, Coord]

# The actions
CRAction = int
DO_NOTHING = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

ACTIONS = [DO_NOTHING, UP, DOWN, LEFT, RIGHT]
ACTIONS_STR = ["0", "U", "D", "L", "R"]
ACTION_TO_DIR = [None, Direction.NORTH, Direction.SOUTH, Direction.WEST, Direction.EAST]

# Obs = (ego_coord, other_coord)
CRObs = Tuple[Coord, Coord]


class CooperativeReachingEnv(DefaultEnv[CRState, CRObs, CRAction]):
    """The Cooperative Reaching Grid World Environment.

    A cooperative 2D grid world problem where two agents must coordinate to go to the
    same goal. This environment tests an agent's ability to coordinate with another
    agent.

    Possible Agents
    ---------------
    The environment supports two agents, with both agents always beginning and ending
    each episode at the same time.

    State Space
    -----------
    Each state is made up of the the `(x, y)` coordinate of each agent. For the
    coordinate `x`=column, `y`=row, with the origin (0, 0) at the top-left square of
    the grid.

    Action Space
    ------------
    Each agent has 5 actions: `DO_NOTHING=0`, `UP=1`, `DOWN=2`, `LEFT=3`, `RIGHT=4`

    Observation Space
    -----------------
    Each agent observes their `(x, y)` coordinate, as well as the `(x, y)` coordinate
    of the other agent, as long as the other agent is within the observation range.
    If the other agent is outside the observation range then their observed coordinate
    will be `(size, size)` (i.e. outside of the grid).

    All together each agent's observation is tuple of the form:

        (ego coord, other coord)

    Rewards
    -------
    Both agents receive a reward when they simoultaneously reach the same goal cell.
    The reward they receive will depend on the value of the goal cell, which is
    determined by the scenario. For all other steps the agents receive a reward of
    `0.0`.

    Dynamics
    --------
    Actions are deterministic and consist of moving to the adjacent cell in each of the
    four cardinal directions. If an agent attempts to move out of bounds of the grid
    then they remain in their current cell. Agents can occupy the same cell at the same
    time.

    Starting State
    --------------
    Agents start from random locations in the middle of the grid.

    Episodes End
    ------------
    Episodes end when both agents simoultaneously reach the same goal cell. By default
    a `max_episode_steps` is also set. The default value is `50` steps, but this may
    need to be adjusted when using larger grids (this can be done by manually specifying
    a value for `max_episode_steps` when creating the environment with `posggym.make`).

    Arguments
    ---------

    - `size` - the size (width and height) of grid.
    - `num_goals` - the number of goal cells in the grid.
    - `mode` - the mode of the environment, which determines the layout of goals in the
        grid as well as their values. The available modes are: ["square", "line",
        "original"]
    - `obs_distance` - the number of cells in each direction that each agent can
        observe. This determines how close agents need to be to each other to be able
        to observe each other's location. Setting this to be `2*size` will make the
        environment fully observable (default = `None` = `2*size`).

    Available variants
    ------------------
    The Cooperative Reaching environment comes with a number of benchmark grid layouts
    which can be passed as an argument to `posggym.make` using the `mode` argument.

    - `original` - the original grid layout from the paper. This is a grid with four
        goals, one in each corner of the grid. The goal values are: top-left = 1,
        top-right = 0.75, bottom-right = 1, bottom-left = 0.75. Note, that this mode
        only supports having four goals (`num_goals=4`).
    - `square` - goals are spaced out evenly along the border of the grid, starting
        from the top-left corner and moving clockwise. Supports any number of goals
        `1 <= num_goals <= (size-1)*4` and all goals have the same value of `1.0`.
    - `line` - goals are laid out in a line evenly along the middle column of the grid
        (or one of the middle columns if the grid has an even number of columns).
        Supports any number of goals `1 <= num_goals <= size` and all goals have the
        same value of `1.0`.

    The following table are some standard benchmark layouts that have been used in
    papers or are similar to those studied in paper:

    | Name              | `size` | `num_goals` | `mode`   |
    |-------------------|--------|------------ | -------- |
    | `original_5`      | 5      | 4           | original |
    | `original_10`     | 10     | 4           | original |
    | `square_5_n4`     | 5      | 4           | square   |
    | `square_10_n4`    | 10     | 4           | square   |
    | `square_10_n8`    | 10     | 8           | square   |
    | `line_5_n3`       | 5      | 3           | line     |
    | `line_7_n4`       | 7      | 4           | line     |
    | `line_11_n6`      | 11     | 6           | line     |

    Note, "Name" here is just provided to give a label for each layout. To use one of
    these layouts the user must specify each argument.

    For example to use the Cooperative Reaching environment with the `square_5_n5`
    benchmark layout, you would use:

    ```python
    import posggym
    env = posggym.make('CooperativeReaching-v0', size=5, num_goals=4, mode="square")
    ```

    Version History
    ---------------
    - `v0`: Initial version

    References
    ----------
    - Arrasy Rahman, Elliot Fosong, Ignacio Carlucho, and Stefano V. Albrecht. 2023.
      Generating Teammates for Training Robust Ad Hoc Teamwork Agents via Best-Response
      Diversity. Transactions on Machine Learning Research.
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array", "rgb_array_dict"],
        "render_fps": 15,
    }

    def __init__(
        self,
        size: int = 5,
        num_goals: int = 4,
        mode: str = "original",
        obs_distance: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__(
            CooperativeReachingModel(size, num_goals, mode, obs_distance),
            render_mode=render_mode,
        )
        self.renderer = None
        self._agent_imgs = None

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
        model: CooperativeReachingModel = self.model  # type: ignore
        grid_str = model.grid.get_ascii_repr(self._state)

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
        model: CooperativeReachingModel = self.model  # type: ignore

        import posggym.envs.grid_world.render as render_lib

        if self.renderer is None:
            self.renderer = render_lib.GWRenderer(
                self.render_mode,
                model.grid,
                render_fps=self.metadata["render_fps"],
                env_name="CooperativeReaching",
            )

        if self._agent_imgs is None:
            img_path = path.join(path.dirname(__file__), "img", "robot.png")
            agent_img = render_lib.load_img_file(img_path, self.renderer.cell_size)
            self._agent_imgs = {
                i: render_lib.GWImage((0, 0), self.renderer.cell_size, agent_img)
                for i in self.possible_agents
            }

        observed_coords = []
        if model.obs_distance < model.size * 2:
            for coord in self._state:
                observed_coords.extend(model.get_obs_coords(coord))

        render_objects = []
        # Add goal locations. Color is green but shaded based on goal value
        for coord, value in model.goals.items():
            render_objects.append(
                render_lib.GWRectangle(
                    coord,
                    self.renderer.cell_size,
                    color=(
                        int(255 * (1 - value)),
                        int(255 * value),
                        int(255 * (1 - value)),
                        int(255 * value),
                    ),
                )
            )

        # Add agents
        for i, coord in enumerate(self._state):
            agent_obj = self._agent_imgs[str(i)]
            agent_obj.coord = coord
            render_objects.append(agent_obj)

        agent_coords_and_dirs = {
            str(i): (coord, Direction.NORTH) for i, coord in enumerate(self._state)
        }

        if self.render_mode in ("human", "rgb_array"):
            return self.renderer.render(render_objects, observed_coords)
        return self.renderer.render_agents(
            render_objects,
            agent_coords_and_dirs,
            agent_obs_dims=model.obs_distance,
            observed_coords=observed_coords,
        )

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class CooperativeReachingModel(M.POSGModel[CRState, CRObs, CRAction]):
    """Driving Problem Model.

    Parameters
    ----------
    size : int
        the size (width and height) of grid.
    num_goals : int
        the number of goals in the grid.
    mode : str
        the mode of the environment, which determines the layout of goals in the
        grid as well as their values. The available modes are: ["square", "line",
        "original"]
    obs_distance : int or None
        the number of cells in each direction that each agent can observe. This
        determines how close agents need to be to each other to be able to observe each
        other's location. Setting this to be `2*size` will make the environment fully
        observable. If `None` then will be set to `2*size`.

    """

    NUM_AGENTS = 2

    MODES = ["square", "line", "original"]

    def __init__(
        self,
        size: int,
        num_goals: int,
        mode: str,
        obs_distance: Optional[int],
    ):
        assert size >= 3, "Grid size must be at least 3"
        assert num_goals >= 1, "Must have at least one goal"
        assert mode in self.MODES, f"Mode must be one of {self.MODES}"
        if obs_distance is None:
            obs_distance = 2 * size
        assert obs_distance >= 0, "Observation distance must be non-negative"

        self.size = size
        self.num_goals = num_goals
        self.mode = mode
        self.obs_distance = obs_distance

        if mode == "square":
            points = equid_points_square(num_goals, self.size)
            self.goals = {p: 1.0 for p in points}
        elif mode == "line":
            points = equid_points_line(num_goals, self.size)
            self.goals = {p: 1.0 for p in points}
        else:
            if num_goals != 4:
                logger.warn(
                    f"'original' mode only supports 4 goals, but got {num_goals}. "
                    "Continuing with 4 goals."
                )
            self.goals = {
                (0, 0): 1.0,
                (self.size - 1, self.size - 1): 1.0,
                (0, self.size - 1): 0.75,
                (self.size - 1, 0): 0.75,
            }

        self.grid = CooperativeReachingGrid(self.size, list(self.goals))
        # TODO review this. Do we want to restrict the start coords so agents start
        # within the observation range of each other?
        self.possible_start_coords = list(product(range(1, self.size - 1), repeat=2))

        def _coord_space(s):
            return spaces.Tuple((spaces.Discrete(s), spaces.Discrete(s)))

        self.possible_agents = tuple(str(i) for i in range(self.NUM_AGENTS))
        self.state_space = spaces.Tuple(
            tuple(_coord_space(size) for _ in range(len(self.possible_agents)))
        )
        self.action_spaces = {
            i: spaces.Discrete(len(ACTIONS)) for i in self.possible_agents
        }
        self.observation_spaces = {
            i: spaces.Tuple(
                (
                    _coord_space(size),
                    _coord_space(size + 1),  # +1 to account for non-observable agent
                )
            )
            for i in self.possible_agents
        }
        self.is_symmetric = True

    @property
    def reward_ranges(self) -> Dict[str, Tuple[float, float]]:
        max_goal_value = max(self.goals.values())
        return {i: (0.0, max_goal_value) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, _ = seeding.std_random()
        return self._rng

    def get_agents(self, state: CRState) -> List[str]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> CRState:
        return tuple(
            self.rng.choice(self.possible_start_coords)
            for _ in range(len(self.possible_agents))
        )

    def sample_agent_initial_state(self, agent_id: str, obs: CRObs) -> CRState:
        assert (0 <= p < self.size for p in obs[1]), (
            "Invalid other agent coord for sampling agent-centric initial state. "
            "Other agent should be observable and within grid."
        )
        agent_idx = int(agent_id)
        return (obs[0], obs[1]) if agent_idx == 0 else (obs[1], obs[0])

    def sample_initial_obs(self, state: CRState) -> Dict[str, CRObs]:
        return self._get_obs(state)

    def step(
        self, state: CRState, actions: Dict[str, CRAction]
    ) -> M.JointTimestep[CRState, CRObs]:
        assert all(0 <= a_i < len(ACTIONS) for a_i in actions.values())
        next_state = self._get_next_state(state, actions)
        obs = self._get_obs(next_state)
        rewards = self._get_rewards(next_state)

        all_done = (
            all(p == next_state[0] for p in next_state) and next_state[0] in self.goals
        )
        terminated = {i: all_done for i in self.possible_agents}
        truncated = {i: False for i in self.possible_agents}

        info: Dict[str, Dict] = {i: {} for i in self.possible_agents}
        if all_done:
            for i in self.possible_agents:
                info[i]["outcome"] = M.Outcome.WIN

        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, info
        )

    def _get_next_state(self, state: CRState, actions: Dict[str, CRAction]) -> CRState:
        next_state = list(state)
        for i, action_i in actions.items():
            if action_i == DO_NOTHING:
                continue
            idx = int(i)
            next_state[idx] = self.grid.get_next_coord(
                next_state[idx], ACTION_TO_DIR[action_i], ignore_blocks=False
            )
        return tuple(next_state)

    def _get_obs(self, state: CRState) -> Dict[str, CRObs]:
        obs: Dict[str, CRObs] = {}
        for i in self.possible_agents:
            idx = int(i)
            other_idx = (idx + 1) % 2
            state_i, state_j = state[idx], state[other_idx]

            if (
                abs(state_i[0] - state_j[0]) <= self.obs_distance
                and abs(state_i[1] - state_j[1]) <= self.obs_distance
            ):
                obs[i] = (state_i, state_j)
            else:
                obs[i] = (state_i, (self.size, self.size))
        return obs

    def get_obs_coords(self, origin: Coord) -> List[Coord]:
        """Get the list of coords observed from agent at origin."""
        obs_size = (2 * self.obs_distance) + 1
        obs_coords: List[Coord] = []
        for obs_col, obs_row in product(range(obs_size), repeat=2):
            grid_col = origin[0] + obs_col - self.obs_distance
            grid_row = origin[1] + obs_row - self.obs_distance
            if 0 <= grid_row < self.size and 0 <= grid_col < self.size:
                obs_coords.append((grid_col, grid_row))
        return obs_coords

    def _get_rewards(self, state: CRState) -> Dict[str, float]:
        all_done = all(p == state[0] for p in state) and state[0] in self.goals
        if all_done:
            goal_value = self.goals[state[0]]
            return {i: goal_value for i in self.possible_agents}
        return {i: 0.0 for i in self.possible_agents}


class CooperativeReachingGrid(Grid):
    """A grid for the Cooperative Reaching Problem."""

    def __init__(
        self,
        size: int,
        goal_coords: List[Coord],
    ):
        assert size >= 3, "Grid size must be at least 3"
        super().__init__(size, size, block_coords=set())
        self.size = size
        self.goal_coords = goal_coords
        self.shortest_paths = self.get_all_shortest_paths(goal_coords)

    def get_shortest_path_distance(self, coord: Coord, goal: Coord) -> int:
        """Get the shortest path distance from coord to goal."""
        return int(self.shortest_paths[goal][coord])

    def get_ascii_repr(self, agent_coords: Optional[CRState]) -> str:
        """Get ascii repr of grid."""
        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                coord = (col, row)
                if coord in self.block_coords:
                    row_repr.append("#")
                elif coord in self.goal_coords:
                    row_repr.append("G")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)

        if agent_coords is not None:
            for i, coord in enumerate(agent_coords):
                grid_repr[coord[0]][coord[1]] = str(i)

        return "\n".join([" ".join(r) for r in grid_repr])


def equid_points_square(n_points: int, grid_size: int) -> List[Tuple[int, int]]:
    """Return n_points equidistant points on square border of grid."""
    assert 0 < n_points <= (grid_size - 1) * 4
    perimeter_length = (grid_size - 1) * 4
    dp = perimeter_length / n_points
    perimeter_idxs = [int(0.5 * dp + i * dp) for i in range(n_points)]
    # need to shift each index so first point is at (0, 0)
    perimeter_idxs = [i - perimeter_idxs[0] for i in perimeter_idxs]
    points = []
    for i in perimeter_idxs:
        if i < grid_size - 1:
            points.append((i, 0))
        elif i < 2 * (grid_size - 1):
            points.append((grid_size - 1, i - (grid_size - 1)))
        elif i < 3 * (grid_size - 1):
            points.append((3 * (grid_size - 1) - i, grid_size - 1))
        else:
            points.append((0, 4 * (grid_size - 1) - i))
    return points


def equid_points_line(n_points: int, grid_size: int) -> List[Tuple[int, int]]:
    """Return n_points equidistant points on line in middle of grid."""
    assert 0 < n_points <= grid_size
    col = int(grid_size / 2)
    dr = grid_size / n_points
    points = [(col, int(0.5 * dr + i * dr)) for i in range(n_points)]
    return points
