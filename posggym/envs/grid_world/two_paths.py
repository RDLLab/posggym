"""The Two-Paths Grid World Environment.

An adversarial 2D grid world problem involving two agents, a runner and
a chaser. The runner's goal is to reach one of two goal location, with each
goal located at the end of a seperate path. The lengths of the two paths
have different lengths. The goal of the chaser is to intercept the runner
before it reaches a goal. The runner is considered caught if it is observed
by the chaser, or occupies the same location. However, the chaser is only
able to effectively cover one of the two goal locations.

This environment requires each agent to reason about the which path the
other agent will choose. It offers an ideal testbed for planning under
finite-nested reasoning assumptions since it is possible to map reasoning
level to the expected path choice.

Reference
---------
Schwartz, Jonathon, Ruijia Zhou, and Hanna Kurniawati. "Online Planning for
Interactive-POMDPs using Nested Monte Carlo Tree Search." In 2022 IEEE/RSJ International
Conference on Intelligent Robots and Systems (IROS), pp. 8770-8777. IEEE, 2022.

"""
import itertools
from typing import Dict, List, Optional, Set, SupportsFloat, Tuple, Union

from gymnasium import spaces

import posggym.envs.grid_world.render as render_lib
import posggym.model as M
from posggym.core import DefaultEnv
from posggym.envs.grid_world.core import Coord, Direction, Grid
from posggym.utils import seeding


TPState = Tuple[Coord, Coord]
TPAction = int
# Obs = (adj_obs, terminal)
TPObs = Tuple[Tuple[int, int, int, int], int]

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
    goal located at the end of a seperate path. The lengths of the two paths
    have different lengths. The goal of the chaser is to intercept the runner
    before it reaches a goal. The runner is considered caught if it is observed
    by the chaser, or occupies the same location. However, the chaser is only
    able to effectively cover one of the two goal locations.

    This environment requires each agent to reason about the which path the
    other agent will choose. It offers an ideal testbed for planning under
    finite-nested reasoning assumptions since it is possible to map reasoning
    level to the expected path choice.

    The two agents start at opposite ends of the maps.

    Agents
    ------
    Runner=0
    Chaser=1

    State
    -----
    Each state contains the (x, y) (x=column, y=row, with origin at the
    top-left square of the grid) of the runner and chaser agent. Specifically,
    a states is ((x_runner, y_runner), (x_chaser, y_chaser))

    Actions
    -------
    Each agent has 4 actions corresponding to moving in the 4 cardinal
    directions (NORTH=0, EAST=1, SOUTH=2, WEST=3).

    Observation
    -----------
    Each agent observes the adjacent cells in the four cardinal directions and
    whether they are one of three things: OPPONENT=0, WALL=1, EMPTY=2.
    Each agent also observes whether a terminal state was reach (0/1). This is
    necessary for the infinite horizon model of the environment.

    Each observation is represented as a tuple:
        ((cell_north, cell_south, cell_east, cell_west), terminal)

    Reward
    ------
    Both agents receive a penalty of -0.01 for each step.
    If the runner reaches the goal then the runner recieves a reward of 1.0,
    while the chaser recieves a penalty of -1.0.
    If the runner is observed by the chaser, then the runner recieves a penalty
    of -1.0, while the chaser recieves a reward of 1.0.

    The rewards make the environment adversarial, but not strictly zero-sum,
    due to the small penalty each step.

    Transition Dynamics
    -------------------
    By default actions are deterministic and an episode ends when either the
    runner is caught, the runner reaches a goal, or the step limit is reached.

    The environment can also be run in stochastic mode by changing the
    action_probs parameter at initialization. This controls the probability
    the agent will move in the desired direction each step, otherwise moving
    randomly in one of the other 3 possible directions.

    Lastly, if using infinite_horizon mode then the environment resets to the
    start state once a terminal state is reached.
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array", "rgb_array_dict"],
        "render_fps": 4,
    }

    def __init__(
        self,
        grid_name: str,
        action_probs: Union[float, Tuple[float, float]] = 1.0,
        infinite_horizon: bool = False,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        self._viewer = None
        self._renderer: Optional[render_lib.GWRenderer] = None
        super().__init__(
            TwoPathsModel(grid_name, action_probs, infinite_horizon, **kwargs),
            render_mode=render_mode,
        )

    def render(self):
        grid = self.model.grid  # type: ignore
        if self.render_mode == "ansi":
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

        if self.render_mode == "human" and self._viewer is None:
            # pylint: disable=[import-outside-toplevel]
            from posggym.envs.grid_world import viewer

            self._viewer = viewer.GWViewer(
                "Two-Paths Env",
                (min(grid.width, 9), min(grid.height, 9)),
                num_agent_displays=len(self.possible_agents),
            )
            self._viewer.show(block=False)

        if self._renderer is None:
            static_objs = [
                render_lib.GWObject(coord, "green", render_lib.Shape.RECTANGLE)
                for coord in grid.goal_coords
            ]
            self._renderer = render_lib.GWRenderer(
                len(self.possible_agents), grid, static_objs, render_blocks=True
            )

        agent_coords = self._state
        agent_obs_coords = tuple(
            grid.get_neighbours(self._state[i], True)
            for i in range(len(self.possible_agents))
        )
        for i, coord in enumerate(agent_coords):
            agent_obs_coords[i].append(coord)
        agent_dirs = tuple(Direction.NORTH for _ in range(len(self.possible_agents)))

        env_img = self._renderer.render(
            self._state,
            agent_obs_coords,
            agent_dirs,
            other_objs=None,
            agent_colors=None,
        )
        agent_obs_imgs = self._renderer.render_all_agent_obs(
            env_img,
            agent_coords,
            agent_dirs,
            agent_obs_dims=(1, 1, 1),
            out_of_bounds_obj=render_lib.GWObject(
                (0, 0), "grey", render_lib.Shape.RECTANGLE
            ),
            agent_obs_coords=agent_obs_coords,
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
            self._viewer.close()
            self._viewer = None


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
    infinite_horizon : bool
        whether problem should terminate once a terminal state is reached
        (default, False) or reset to start position and continue (True).

    """

    NUM_AGENTS = 2

    RUNNER_IDX = 0
    CHASER_IDX = 1

    R_ACTION = -0.01
    R_CAPTURE = -1.0  # Runner reward, Chaser = -R_CAPTURE
    R_SAFE = 1.0  # Runner reward, Chaser = -R_SAFE

    def __init__(
        self,
        grid_name: str,
        action_probs: Union[float, Tuple[float, float]] = 1.0,
        infinite_horizon: bool = False,
        **kwargs,
    ):
        self.grid = load_grid(grid_name)
        self._infinite_horizon = infinite_horizon
        if isinstance(action_probs, float):
            action_probs = (action_probs, action_probs)
        self._action_probs = action_probs

        self.possible_agents = tuple(range(self.NUM_AGENTS))
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
                    spaces.Tuple(
                        (
                            spaces.Discrete(len(CELL_OBS)),
                            spaces.Discrete(len(CELL_OBS)),
                            spaces.Discrete(len(CELL_OBS)),
                            spaces.Discrete(len(CELL_OBS)),
                        )
                    ),
                    spaces.Discrete(2),
                )
            )
            for i in self.possible_agents
        }
        self.observation_first = True
        self.is_symmetric = False

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[SupportsFloat, SupportsFloat]]:
        return {i: (self.R_CAPTURE, self.R_SAFE) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: TPState) -> List[M.AgentID]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> TPState:
        return (self.grid.init_runner_coord, self.grid.init_chaser_coord)

    def sample_agent_initial_state(self, agent_id: M.AgentID, obs: TPObs) -> TPState:
        return self.sample_initial_state()

    def get_initial_belief_dist(self) -> Dict[TPState, float]:
        s_0 = (self.grid.init_runner_coord, self.grid.init_chaser_coord)
        return {
            s: float(s == s_0)  # type: ignore
            for s in itertools.product(self.grid.all_coords, repeat=2)
        }

    def sample_initial_obs(self, state: TPState) -> Dict[M.AgentID, TPObs]:
        return self._get_obs(state, False)

    def step(
        self, state: TPState, actions: Dict[M.AgentID, TPAction]
    ) -> M.JointTimestep[TPState, TPObs]:
        assert all(0 <= a_i < len(Direction) for a_i in actions.values())
        next_state = self._get_next_state(state, actions)
        rewards = self._get_rewards(next_state)
        terminal = self._state_is_terminal(next_state)

        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}
        if terminal:
            for i, outcome in self._get_outcome(next_state).items():
                info[i]["outcome"] = outcome

        if self._infinite_horizon and terminal:
            next_state = self.sample_initial_state()
            terminal = False

        obs = self._get_obs(next_state, terminal)
        truncated = {i: False for i in self.possible_agents}
        terminated = {i: terminal for i in self.possible_agents}

        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, terminal, info
        )

    def _get_next_state(
        self, state: TPState, actions: Dict[M.AgentID, TPAction]
    ) -> TPState:
        runner_coord = state[self.RUNNER_IDX]
        chaser_coord = state[self.CHASER_IDX]
        runner_a = actions[self.RUNNER_IDX]
        chaser_a = actions[self.CHASER_IDX]

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

    def _get_obs(self, state: TPState, terminal: bool) -> Dict[M.AgentID, TPObs]:
        runner_coord = state[self.RUNNER_IDX]
        chaser_coord = state[self.CHASER_IDX]
        return {
            self.RUNNER_IDX: (
                self._get_adj_obs(runner_coord, chaser_coord),
                int(terminal),
            ),
            self.CHASER_IDX: (
                self._get_adj_obs(chaser_coord, runner_coord),
                int(terminal),
            ),
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

    def _get_rewards(self, state: TPState) -> Dict[M.AgentID, SupportsFloat]:
        runner_coord = state[self.RUNNER_IDX]
        chaser_coord = state[self.CHASER_IDX]
        r_runner, r_chaser = (self.R_ACTION, self.R_ACTION)
        if runner_coord in self.grid.goal_coords:
            r_runner, r_chaser = (self.R_SAFE, -self.R_SAFE)
        if runner_coord == chaser_coord or runner_coord in self.grid.get_neighbours(
            chaser_coord, True
        ):
            r_runner, r_chaser = (self.R_CAPTURE, -self.R_CAPTURE)
        return {self.RUNNER_IDX: r_runner, self.CHASER_IDX: r_chaser}

    def _get_outcome(self, state: TPState) -> Dict[M.AgentID, M.Outcome]:
        if self._infinite_horizon:
            return {self.RUNNER_IDX: M.Outcome.NA, self.CHASER_IDX: M.Outcome.NA}

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
        return {self.RUNNER_IDX: runner_outcome, self.CHASER_IDX: chaser_outcome}

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

        return "\n".join(list(list((" ".join(r) for r in grid_repr))))

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
        block_coords=set([(1, 1)]),
        goal_coords=set([(0, 0), (2, 1)]),
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
    block_coords = set(
        [
            #
            (1, 1),
            (2, 1),
            (1, 2),
            (3, 3),
        ]
    )

    return TPGrid(
        grid_height=4,
        grid_width=4,
        block_coords=block_coords,
        goal_coords=set([(0, 0), (3, 1)]),
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
    block_coords = set(
        [
            #
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
        ]
    )

    return TPGrid(
        grid_height=7,
        grid_width=7,
        block_coords=block_coords,
        goal_coords=set([(0, 0), (6, 2)]),
        init_runner_coord=(3, 6),
        init_chaser_coord=(4, 0),
    )


# grid_name: (grid_make_fn, episode step_limit)
SUPPORTED_GRIDS = {
    "3x3": (get_3x3_grid, 20),
    "4x4": (get_4x4_grid, 20),
    "7x7": (get_7x7_grid, 20),
}


def load_grid(grid_name: str) -> TPGrid:
    """Load grid with given name."""
    grid_name = grid_name
    assert grid_name in SUPPORTED_GRIDS, (
        f"Unsupported grid name '{grid_name}'. Grid name must be one of: "
        f"{SUPPORTED_GRIDS.keys()}."
    )
    return SUPPORTED_GRIDS[grid_name][0]()
