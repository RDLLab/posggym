"""The Two-Paths Grid World Environment."""
import itertools
from os import path
from typing import Dict, List, Optional, Set, Tuple, Union

from gymnasium import spaces

import posggym.model as M
from posggym import logger
from posggym.core import DefaultEnv
from posggym.envs.grid_world.core import Coord, Direction, Grid
from posggym.utils import seeding

TPState = Tuple[Coord, Coord]
TPAction = int
# Obs = adj_obs
TPObs = Tuple[int, int, int, int]

# Cell obs
OPPONENT = 0
WALL = 1
EMPTY = 2

CELL_OBS = [OPPONENT, WALL, EMPTY]
CELL_OBS_STR = ["X", "#", "0"]


class TwoPathsEnv(DefaultEnv[TPState, TPObs, TPAction]):
    """The Two-Paths Grid World Environment.

    An adversarial 2D grid world problem involving two agents, a runner and
    a chaser. The runner's goal is to reach one of two goal location, with each
    goal located at the end of a separate path. The lengths of the two paths
    have different lengths. The goal of the chaser is to intercept the runner
    before it reaches a goal. The runner is considered caught if it is observed
    by the chaser, or occupies the same location. However, the chaser is only
    able to effectively cover one of the two goal locations.

    This environment requires each agent to reason about the which path the
    other agent will choose. It offers an ideal testbed for planning under
    finite-nested reasoning assumptions since it is possible to map reasoning
    level to the expected path choice.

    The two agents start at opposite ends of the maps.

    Possible Agents
    ---------------
    - Runner = '0'
    - Chaser = '1'

    State Space
    -----------
    Each state contains the `(x, y)` (x=column, y=row, with origin at the
    top-left square of the grid) of the runner and chaser agent. Specifically,
    a state is `((x_runner, y_runner), (x_chaser, y_chaser))`

    Action Space
    ------------
    Each agent has 4 actions corresponding to moving in the 4 cardinal
    directions: `NORTH=0`, `EAST=1`, `SOUTH=2`, `WEST=3`.

    Observation Space
    -----------------
    Each agent observes the adjacent cells in the four cardinal directions and
    whether they are one of three things: `OPPONENT=0`, `WALL=1`, `EMPTY=2`.

    Each observation is represented as a tuple:
    `(cell_north, cell_south, cell_east, cell_west)`

    Rewards
    -------
    Both agents receive a penalty of `-0.01` for each step. If the runner reaches the
    goal then the runner receives a reward of `1.0`, while the chaser receives a penalty
    of `-1.0`. If the runner is observed by the chaser, then the runner receives a
    penalty of `-1.0`, while the chaser receives a reward of `1.0`.

    The rewards make the environment adversarial, but not strictly zero-sum, due to the
    small penalty each step.

    Dynamics
    --------
    By default actions are deterministic and will lead to the agent moving one cell in
    the action's direction, given the cell adjacent cell in that direction is empty and
    not out-of-bounds.

    The environment can also be run in stochastic mode by changing the `action_probs`
    parameter at initialization. This controls the probability the agent will move in
    the desired direction each step, otherwise moving randomly in one of the other 3
    other possible directions.

    Starting State
    --------------
    Both runner and chaser agents start at the same location on opposite ends of the
    grid each episode, and the goal locations and grid layout are also the same. The
    specific starting state and configuration depends on the grid layout used.

    Episodes End
    ------------
    Episode ends when either the runner is caught, or reaches a goal. By default a
    `max_episode_steps` limit of `20` is also set.

    Arguments
    ---------

    - `grid_size` - the grid size to use. This can either `3`, `4`, or `7`, each size
        `n` create a TwoPaths Env with a `n`-by-`n` grid layout (default = `7`).
    - `action_probs` - the action success probability for each agent. This can be a
        single float (same value for both runner and chaser agents) or a tuple with
        separate values for the runner and chaser (default = `1.0`).

    Reference
    ---------
    Schwartz, Jonathon, Ruijia Zhou, and Hanna Kurniawati. "Online Planning for
    Interactive-POMDPs using Nested Monte Carlo Tree Search." In 2022 IEEE/RSJ
    International Conference on Intelligent Robots and Systems (IROS), pp. 8770-8777.
    IEEE, 2022.

    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array", "rgb_array_dict"],
        "render_fps": 15,
    }

    def __init__(
        self,
        grid_size: int = 7,
        action_probs: Union[float, Tuple[float, float]] = 1.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__(
            TwoPathsModel(grid_size, action_probs),
            render_mode=render_mode,
        )
        self.renderer = None
        self.chaser_img = None
        self.runner_img = None

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
        grid = self.model.grid  # type: ignore
        grid_str = grid.get_ascii_repr(self._state[0], self._state[1])

        output = [
            f"Step: {self._step_num}",
            grid_str,
        ]
        if self._last_actions is not None:
            action_str = ", ".join(
                [str(Direction(a)) for a in self._last_actions.values()]
            )
            output.insert(1, f"Actions: <{action_str}>")
            output.append(f"Rewards: <{self._last_rewards}>")

        return "\n".join(output) + "\n"

    def _render_img(self):
        grid: Grid = self.model.grid  # type: ignore

        import posggym.envs.grid_world.render as render_lib

        if self.renderer is None:
            self.renderer = render_lib.GWRenderer(
                self.render_mode,
                grid,
                render_fps=self.metadata["render_fps"],
                env_name="Two Paths",
            )

            goal_imgs = [
                render_lib.GWRectangle(
                    coord, self.renderer.cell_size, render_lib.get_color("green")
                )
                for coord in grid.goal_coords
            ]
            self.renderer.static_objects.extend(goal_imgs)

        if self.runner_img is None:
            img_path = path.join(path.dirname(__file__), "img", "robot.png")
            img = render_lib.load_img_file(img_path, self.renderer.cell_size)
            self.runner_img = render_lib.GWImage((0, 0), self.renderer.cell_size, img)

        if self.chaser_img is None:
            img_path = path.join(path.dirname(__file__), "img", "robot.png")
            img = render_lib.load_img_file(img_path, self.renderer.cell_size)
            self.chaser_img = render_lib.GWImage((0, 0), self.renderer.cell_size, img)

        self.runner_img.coord = self._state[0]
        self.chaser_img.coord = self._state[1]
        render_objects = [self.runner_img, self.chaser_img]

        observed_coords = grid.get_neighbours(self._state[0])
        observed_coords.extend(grid.get_neighbours(self._state[1]))

        if self.render_mode in ("human", "rgb_array"):
            return self.renderer.render(render_objects, observed_coords)

        return self.renderer.render_agents(
            render_objects,
            {
                "0": (self._state[0], Direction.NORTH),
                "1": (self._state[1], Direction.NORTH),
            },
            agent_obs_dims=1,
            observed_coords=observed_coords,
            # mask corners of obs square
            agent_obs_mask=[(0, 0), (0, 2), (2, 0), (2, 2)],
        )

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class TwoPathsModel(M.POSGModel[TPState, TPObs, TPAction]):
    """Two-Paths Problem Model.

    Parameters
    ----------
    grid_name : str
        determines the environment map (see posggym.envs.two_paths.grid for
        available grids)
    action_probs : float or (float, float)
        the action success probability for each agent. By default the
        environment is deterministic (action_probs=1.0).

    """

    NUM_AGENTS = 2

    RUNNER_IDX = 0
    CHASER_IDX = 1

    R_ACTION = -0.01
    R_CAPTURE = -1.0  # Runner reward, Chaser = -R_CAPTURE
    R_SAFE = 1.0  # Runner reward, Chaser = -R_SAFE

    def __init__(
        self,
        grid_size: int,
        action_probs: Union[float, Tuple[float, float]] = 1.0,
    ):
        assert grid_size in SUPPORTED_GRIDS, (
            f"Unsupported grid_size of `{grid_size}`, must be one of: "
            f"{SUPPORTED_GRIDS.keys()}."
        )
        self.grid = SUPPORTED_GRIDS[grid_size]()

        if isinstance(action_probs, float):
            action_probs = (action_probs, action_probs)
        self._action_probs = action_probs

        self.possible_agents = tuple(str(i) for i in range(self.NUM_AGENTS))
        self.state_space = spaces.Tuple(
            tuple(
                spaces.Tuple(
                    (
                        spaces.Discrete(self.grid.width),
                        spaces.Discrete(self.grid.height),
                    )
                )
                for _ in range(len(self.possible_agents))
            )
        )
        self.action_spaces = {
            i: spaces.Discrete(len(Direction)) for i in self.possible_agents
        }
        self.observation_spaces = {
            i: spaces.Tuple(
                (
                    spaces.Discrete(len(CELL_OBS)),
                    spaces.Discrete(len(CELL_OBS)),
                    spaces.Discrete(len(CELL_OBS)),
                    spaces.Discrete(len(CELL_OBS)),
                )
            )
            for i in self.possible_agents
        }
        self.is_symmetric = False

    @property
    def reward_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {i: (self.R_CAPTURE, self.R_SAFE) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: TPState) -> List[str]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> TPState:
        return (self.grid.init_runner_coord, self.grid.init_chaser_coord)

    def sample_agent_initial_state(self, agent_id: str, obs: TPObs) -> TPState:
        return self.sample_initial_state()

    def get_initial_belief_dist(self) -> Dict[TPState, float]:
        s_0 = (self.grid.init_runner_coord, self.grid.init_chaser_coord)
        return {
            s: float(s == s_0)  # type: ignore
            for s in itertools.product(self.grid.all_coords, repeat=2)
        }

    def sample_initial_obs(self, state: TPState) -> Dict[str, TPObs]:
        return self._get_obs(state)

    def step(
        self, state: TPState, actions: Dict[str, TPAction]
    ) -> M.JointTimestep[TPState, TPObs]:
        assert all(0 <= a_i < len(Direction) for a_i in actions.values())
        next_state = self._get_next_state(state, actions)
        rewards = self._get_rewards(next_state)
        all_done = self._state_is_terminal(next_state)

        info: Dict[str, Dict] = {i: {} for i in self.possible_agents}
        if all_done:
            for i, outcome in self._get_outcome(next_state).items():
                info[i]["outcome"] = outcome

        obs = self._get_obs(next_state)
        truncated = {i: False for i in self.possible_agents}
        terminated = {i: all_done for i in self.possible_agents}

        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, info
        )

    def _get_next_state(self, state: TPState, actions: Dict[str, TPAction]) -> TPState:
        runner_coord = state[self.RUNNER_IDX]
        chaser_coord = state[self.CHASER_IDX]
        runner_a = actions[str(self.RUNNER_IDX)]
        chaser_a = actions[str(self.CHASER_IDX)]

        if self.rng.random() > self._action_probs[self.CHASER_IDX]:
            other_as = [a for a in Direction if a != chaser_a]
            chaser_a = self.rng.choice(other_as)
        chaser_next_coord = self.grid.get_next_coord(
            chaser_coord, Direction(chaser_a), ignore_blocks=False
        )

        if chaser_next_coord == runner_coord:
            # Runner considered to capture Fugitive
            runner_next_coord = runner_coord
        else:
            if self.rng.random() > self._action_probs[self.RUNNER_IDX]:
                other_as = [a for a in Direction if a != runner_a]
                runner_a = self.rng.choice(other_as)
            runner_next_coord = self.grid.get_next_coord(
                runner_coord, Direction(runner_a), ignore_blocks=False
            )

        return (runner_next_coord, chaser_next_coord)

    def _get_obs(self, state: TPState) -> Dict[str, TPObs]:
        runner_coord = state[self.RUNNER_IDX]
        chaser_coord = state[self.CHASER_IDX]
        return {
            str(self.RUNNER_IDX): self._get_adj_obs(runner_coord, chaser_coord),
            str(self.CHASER_IDX): self._get_adj_obs(chaser_coord, runner_coord),
        }

    def _get_adj_obs(
        self, coord: Coord, opponent_coord: Coord
    ) -> Tuple[int, int, int, int]:
        adj_obs = []
        for d in Direction:
            next_coord = self.grid.get_next_coord(coord, d, False)
            if next_coord == opponent_coord:
                adj_obs.append(OPPONENT)
            elif next_coord == coord:
                adj_obs.append(WALL)
            else:
                adj_obs.append(EMPTY)
        return tuple(adj_obs)  # type: ignore

    def _get_rewards(self, state: TPState) -> Dict[str, float]:
        runner_coord = state[self.RUNNER_IDX]
        chaser_coord = state[self.CHASER_IDX]
        r_runner, r_chaser = (self.R_ACTION, self.R_ACTION)
        if runner_coord in self.grid.goal_coords:
            r_runner, r_chaser = (self.R_SAFE, -self.R_SAFE)
        if runner_coord == chaser_coord or runner_coord in self.grid.get_neighbours(
            chaser_coord, True
        ):
            r_runner, r_chaser = (self.R_CAPTURE, -self.R_CAPTURE)
        return {str(self.RUNNER_IDX): r_runner, str(self.CHASER_IDX): r_chaser}

    def _get_outcome(self, state: TPState) -> Dict[str, M.Outcome]:
        # Assuming state is terminal
        runner_coord = state[self.RUNNER_IDX]
        chaser_coord = state[self.CHASER_IDX]
        runner_outcome, chaser_outcome = (M.Outcome.DRAW, M.Outcome.DRAW)
        if runner_coord in self.grid.goal_coords:
            runner_outcome, chaser_outcome = (M.Outcome.WIN, M.Outcome.LOSS)
        elif runner_coord == chaser_coord or runner_coord in self.grid.get_neighbours(
            chaser_coord, True
        ):
            runner_outcome, chaser_outcome = (M.Outcome.LOSS, M.Outcome.WIN)
        return {
            str(self.RUNNER_IDX): runner_outcome,
            str(self.CHASER_IDX): chaser_outcome,
        }

    def _state_is_terminal(self, state: TPState) -> bool:
        runner_coords = state[self.RUNNER_IDX]
        chaser_coords = state[self.CHASER_IDX]
        if chaser_coords == runner_coords or runner_coords in self.grid.goal_coords:
            return True
        neighbour_coords = self.grid.get_neighbours(chaser_coords, True)
        return runner_coords in neighbour_coords


class TPGrid(Grid):
    """A grid for the Two-Paths Problem."""

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        block_coords: Set[Coord],
        goal_coords: Set[Coord],
        init_runner_coord: Coord,
        init_chaser_coord: Coord,
    ):
        super().__init__(grid_width, grid_height, block_coords)
        self.goal_coords = goal_coords
        self.init_runner_coord = init_runner_coord
        self.init_chaser_coord = init_chaser_coord

    def get_ascii_repr(
        self, runner_coord: Optional[Coord], chaser_coord: Optional[Coord]
    ) -> str:
        """Get ascii repr of grid."""
        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                coord = (col, row)
                if coord in self.goal_coords:
                    row_repr.append("G")
                elif coord in self.block_coords:
                    row_repr.append("#")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)

        if runner_coord is not None:
            grid_repr[runner_coord[1]][runner_coord[0]] = "R"
        if chaser_coord is not None:
            if chaser_coord == runner_coord:
                grid_repr[chaser_coord[1]][chaser_coord[0]] = "X"
            else:
                grid_repr[chaser_coord[1]][chaser_coord[0]] = "C"

        return "\n".join([" ".join(r) for r in grid_repr])

    def get_init_ascii_repr(self) -> str:
        """Get ascii repr of initial grid."""
        return self.get_ascii_repr(self.init_runner_coord, self.init_chaser_coord)


def get_3x3_grid() -> TPGrid:
    """Generate the Two-Paths 3-by-3 grid layout.

       012
      #####
    0 #GC #
    1 # #G#
    2 # R #
      #####

    """
    return TPGrid(
        grid_height=3,
        grid_width=3,
        block_coords={(1, 1)},
        goal_coords={(0, 0), (2, 1)},
        init_runner_coord=(1, 2),
        init_chaser_coord=(1, 0),
    )


def get_4x4_grid() -> TPGrid:
    """Generate the Choose a Path 4-by-4 grid layout.

       0123
      ######
    0 #G C #
    1 # ##G#
    2 # #  #
    3 #  R##
      ######

    """
    block_coords = {
        (1, 1),
        (2, 1),
        (1, 2),
        (3, 3),
    }

    return TPGrid(
        grid_height=4,
        grid_width=4,
        block_coords=block_coords,
        goal_coords={(0, 0), (3, 1)},
        init_runner_coord=(2, 3),
        init_chaser_coord=(2, 0),
    )


def get_7x7_grid() -> TPGrid:
    """Generate the Two-Paths 7-by-7 grid layout.

       0123456
      #########
    0 #G   C  #
    1 # ##### #
    2 # #### G#
    3 #  ### ##
    4 ## ### ##
    5 ##  #  ##
    6 ### R ###
      #########

    """
    block_coords = {
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (2, 3),
        (3, 3),
        (4, 3),
        (6, 3),
        (0, 4),
        (2, 4),
        (3, 4),
        (4, 4),
        (6, 4),
        (0, 5),
        (3, 5),
        (6, 5),
        (0, 6),
        (1, 6),
        (5, 6),
        (6, 6),
    }

    return TPGrid(
        grid_height=7,
        grid_width=7,
        block_coords=block_coords,
        goal_coords={(0, 0), (6, 2)},
        init_runner_coord=(3, 6),
        init_chaser_coord=(4, 0),
    )


# grid_size: grid_make_fn
SUPPORTED_GRIDS = {
    3: get_3x3_grid,
    4: get_4x4_grid,
    7: get_7x7_grid,
}
