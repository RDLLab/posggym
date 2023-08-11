"""The Pursuit-Evasion Grid World Environment."""
from collections import deque
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Set, Tuple, Union

from gymnasium import spaces

import posggym.model as M
from posggym import logger
from posggym.core import DefaultEnv
from posggym.envs.grid_world.core import Coord, Direction, Grid
from posggym.utils import seeding


# State = (e_coord, e_dir, p_coord, p_dir, e_0_coord, p_0_coord, e_goal_coord)
INITIAL_DIR = Direction.NORTH


class PEState(NamedTuple):
    """Environment state in Pursuit Evastion problem."""

    evader_coord: Coord
    evader_dir: Direction
    pursuer_coord: Coord
    pursuer_dir: Direction
    evader_start_coord: Coord
    pursuer_start_coord: Coord
    evader_goal_coord: Coord
    min_goal_dist: int  # for evader


# Action = Direction of movement
# 0 = Forward, 1 = Backward, 2 = Left, 3 = Right
PEAction = int
ACTION_TO_DIR = [
    # Forward
    [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST],
    # Backward
    [Direction.SOUTH, Direction.WEST, Direction.NORTH, Direction.EAST],
    # Left
    [Direction.WEST, Direction.NORTH, Direction.EAST, Direction.SOUTH],
    # Right
    [Direction.EAST, Direction.SOUTH, Direction.WEST, Direction.NORTH],
]

# E Obs = Tuple[WallObs, seen, heard, e_0_coord, p_0_coord, goal_coord]
#       = Tuple[Tuple[int, int, int, int, int, int], Coord, Coord, Coord]
# P Obs = Tuple[WallObs, seen, heard, e_0_coord, p_0_coord, blank_coord]
#       = Tuple[Tuple[int, int, int, int, int, int], Coord, Coord, Coord]
# Note, we use blank_coord for P Obs so Obs spaces are identical between the
# two agents. The blank_coord is always (0, 0).
PEEvaderObs = Tuple[Tuple[int, ...], Coord, Coord, Coord]
PEPursuerObs = Tuple[Tuple[int, ...], Coord, Coord, Coord]
PEObs = Union[PEEvaderObs, PEPursuerObs]


class PursuitEvasionEnv(DefaultEnv):
    """The Pursuit-Evasion Grid World Environment.

    An adversarial 2D grid world problem involving two agents: an evader and a pursuer.
    The evader's goal is to reach a goal location, on the other side of the grid, while
    the goal of the pursuer is to spot the evader before it reaches it's goal. The
    evader is considered caught if it is observed by the pursuer, or occupies the same
    location. The evader and pursuer have knowledge of each others starting locations.
    However, only the evader has knowledge of it's goal location. The pursuer only
    knowns that the evader's goal location is somewhere on the opposite side of the grid
    to the evaders start location.

    This environment requires each agent to reason about the which path the other agent
    will take through the dense grid environment.

    Possible Agents
    ---------------
    - Evader = `"0"`
    - Pursuer = `"1"`

    State Space
    -----------
    Each state is made up of:

    0. the `(x, y)` coordinate of the evader
    1. the direction the evader is facing
    2. the `(x, y)` coordinate of the pursuer
    3. the direction the pursuer is facing
    4. the `(x, y)` coordinate of the evader's start location
    5. the `(x, y)` coordinate of the pursuers's start location
    6. the `(x, y)` coordinate of the evaders's goal location
    7. the minimum distance to it's goal along the shortest path achieved by the evader
       in the current episode (this is needed to correctly reward the agent for making
       progress.)

    Action Space
    ------------
    Each agent has 4 actions corresponding to moving in the 4 available directions with
    respect to the direction the agent is currently facing: `FORWARD=0`, `BACKWARDS=1`,
    `LEFT=2`, `RIGHT=3`.

    Observation Space
    -----------------
    Each agent observes:

    1. whether there is a wall (`1`) or not (`0`) in the adjacent cells in the four
       cardinal directions,
    2. whether they see the other agent in a cone in front of them (`1`) or not (`0`).
       The cone projects forward up to 'max_obs_distance' (default=`12`) cells in front
       of the agent.
    3. whether they hear the other agent (`1`) or not (`0`). The other agent is heard if
       they are within distance 2 from the agent in any direction.
    4. the `(x, y)` coordinate of the evader's start location,
    5. the `(x, y)` coordinate of the pursuer's start location,
    6. Evader: the `(x, y)` coordinate of the evader's goal location.
       Pursuer: blank coordinate `(0, 0)`.

    Note, the goal and start coordinate observations do not change during a single
    episode, but they do change between episodes.

    Rewards
    -------
    The environment is zero-sum with the pursuer receiving the negative of the evader
    reward. Additionally, rewards are by default normalized so that returns are bounded
    between `-1` and `1` (this can be disabled by the `normalize_reward` parameter).

    The evader receives a reward of `1` for reaching it's goal location and a
    reward of `-1` if it gets captured. Additionally, the evader receives a small
    reward of `0.01` each time it's minimum distance achieved to it's goal along the
    shortest path decreases for the current episode. This is to make it so the
    environment is no longer sparesely rewarded and helps with exploration and learning
    (it can be disabled by the `use_progress_reward` parameter.)

    Dynamics
    --------
    By default actions are deterministic and will move the agent one cell in the target
    direction if the cell is empty.

    The environment can also be run in stochastic mode by changing the `action_probs`
    parameter at initialization. This controls the probability the agent will move in
    the desired direction each step, otherwise moving randomly in one of the other 3
    possible directions.

    Starting State
    --------------
    At the start of each episode the start location of the evader is selected at random
    from all possible start locations. The evader's goal location is then chosen
    randomly from the set of available goal locations given the evaders start location
    (in the default maps the goal location is always on the opposite side of the map
    from a start location). The pursuers start location is similarly chosen from it's
    set of possible start locations.

    Episode End
    -----------
    An episode ends when either the evader is caught or the evader reaches it's goal.
    By default a `max_episode_steps` limit of `100` steps is also set. This may need to
    be adjusted when using larger grids (this can be done by manually specifying a value
    for `max_episode_steps` when creating the environment with `posggym.make`).

    Arguments
    ---------

    - `grid` - the grid layout to use. This can either be a string specifying one of
         the supported grids, or a custom :class:`PEGrid` object (default = `"16x16"`).
    - `action_probs` - the action success probability for each agent. This can be a
        single float (same value for both evader and pursuer agents) or a tuple with
        separate values for each agent (default = `1.0`).
    - `max_obs_distance` - the maximum number of cells in front each agent's field of
        vision extends (default = `12`).
    - `num_prey` - the number of prey (default = `3`)
    - `normalize_reward` - whether to normalize both agents' rewards to be between `-1`
        and `1` (default = 'True`)
    - `use_progress_reward` - whether to reward the evader agent for making progress
        towards it's goal. If False the evader will only be rewarded when it reaches
        it's goal, making it a sparse reward problem (default = 'True`).

    Available variants
    ------------------

    The PursuitEvasion environment comes with a number of pre-built grid layouts which
    can be passed as an argument to `posggym.make`, to create different grids.

    | Grid name         | Grid size |
    |-------------------|-----------|
    | `5x5`             | 8x8       |
    | `16x16`           | 16x16     |
    | `32x32`           | 32x32     |

    For example to use the PursuitEvasion environment with the `32x32` grid layout, and
    episode step limit of 200, and the default values for the other parameters you would
    use:

    ```python
    import posggym
    env = posggym.make(
        'PursuitEvasion-v0',
        max_episode_steps=200,
        grid="32x32",
    )
    ```

    References
    ----------
    - [This Pursuit-Evasion implementation is directly inspired by the problem] Seaman,
      Iris Rubi, Jan-Willem van de Meent, and David Wingate. 2018. “Nested Reasoning
      About Autonomous Agents Using Probabilistic Programs.”
      ArXiv Preprint ArXiv:1812.01569.
    - Schwartz, Jonathon, Ruijia Zhou, and Hanna Kurniawati. "Online Planning for
      Interactive-POMDPs using Nested Monte Carlo Tree Search." In 2022 IEEE/RSJ
      International Conference on Intelligent Robots and Systems (IROS), pp. 8770-8777.
      IEEE, 2022.

    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 15,
    }

    def __init__(
        self,
        grid: Union[str, "PEGrid"] = "16x16",
        action_probs: Union[float, Tuple[float, float]] = 1.0,
        max_obs_distance: int = 12,
        normalize_reward: bool = True,
        use_progress_reward: bool = True,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        model = PursuitEvasionModel(
            grid,
            action_probs=action_probs,
            max_obs_distance=max_obs_distance,
            normalize_reward=normalize_reward,
            use_progress_reward=use_progress_reward,
            **kwargs,
        )
        super().__init__(model, render_mode=render_mode)

        self.max_obs_distance = max_obs_distance
        fov_width = model.grid.get_max_fov_width(
            model.FOV_EXPANSION_INCR, max_obs_distance
        )
        self._obs_dims = (
            min(max_obs_distance, max(model.grid.width, model.grid.height)),
            0,
            fov_width // 2,
            fov_width // 2,
        )
        self.renderer = None
        self._agent_imgs = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, M.ObsType], Dict[str, Dict]]:
        # reset renderer since goal location can change between episodes
        self._renderer = None
        return super().reset(seed=seed, options=options)

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
        evader_coord = self._state[0]
        pursuer_coord = self._state[2]
        goal_coord = self._state[6]
        grid = self.model.grid  # type: ignore

        grid_str = grid.get_ascii_repr(goal_coord, evader_coord, pursuer_coord)
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
        evader_coord = self._state[0]
        pursuer_coord = self._state[2]
        goal_coord = self._state[6]
        model: PursuitEvasionModel = self.model  # type: ignore

        import posggym.envs.grid_world.render as render_lib

        if self.renderer is None:
            self.renderer = render_lib.GWRenderer(
                self.render_mode,
                model.grid,
                render_fps=self.metadata["render_fps"],
                env_name="PursuitEvasion",
            )

        if self._agent_imgs is None:
            self._agent_imgs = {
                i: render_lib.GWTriangle(
                    (0, 0),
                    self.renderer.cell_size,
                    render_lib.get_agent_color(i)[0],
                    Direction.NORTH,
                )
                for i in self.possible_agents
            }

        observed_coords = []
        for i in range(len(self.agents)):
            observed_coords.extend(
                model.grid.get_fov(
                    self._state[2 * i],
                    self._state[2 * i + 1],
                    self.model.FOV_EXPANSION_INCR,
                    self.max_obs_distance,
                )
            )

        render_objects = [
            render_lib.GWRectangle(
                goal_coord, self.renderer.cell_size, render_lib.get_color("green")
            )
        ]

        agent_coords_and_dirs = {
            "0": (evader_coord, self._state[1]),
            "1": (pursuer_coord, self._state[3]),
        }
        for i, (c, d) in agent_coords_and_dirs.items():
            agent_img = self._agent_imgs[i]
            agent_img.coord = c
            agent_img.facing_dir = d
            render_objects.append(agent_img)

        return self.renderer.render(render_objects, observed_coords)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class PursuitEvasionModel(M.POSGModel[PEState, PEObs, PEAction]):
    """Discrete Pursuit-Evasion Model."""

    NUM_AGENTS = 2

    EVADER_IDX = 0
    PURSUER_IDX = 1

    # Evader-centric rewards
    R_PROGRESS = 0.01  # Reward making progress toward goal
    R_EVADER_ACTION = -0.01  # Reward each step for evader
    R_PURSUER_ACTION = -0.01  # Reward each step for pursuer
    R_CAPTURE = -1.0  # Reward for being captured
    R_EVASION = 1.0  # Reward for reaching goal

    FOV_EXPANSION_INCR = 3
    HEARING_DIST = 2

    def __init__(
        self,
        grid: Union[str, "PEGrid"],
        action_probs: Union[float, Tuple[float, float]] = 1.0,
        max_obs_distance: int = 12,
        normalize_reward: bool = True,
        use_progress_reward: bool = True,
    ):
        if isinstance(grid, str):
            assert grid in SUPPORTED_GRIDS, (
                f"Unsupported grid name '{grid}'. If grid is a string it must be one "
                f"of: {SUPPORTED_GRIDS.keys()}."
            )
            grid = SUPPORTED_GRIDS[grid][0]()

        if isinstance(action_probs, float):
            action_probs = (action_probs, action_probs)

        self._grid = grid
        self.action_probs = action_probs
        self.max_obs_distance = max_obs_distance
        self.normalize_reward = normalize_reward
        self.use_progress_reward = use_progress_reward

        self._max_sp_distance = self._grid.get_max_shortest_path_distance()
        self._max_raw_return = self.R_EVASION
        if self.use_progress_reward:
            self._max_raw_return += (self._max_sp_distance + 1) * self.R_PROGRESS
        self._min_raw_return = -self._max_raw_return

        def _coord_space():
            return spaces.Tuple(
                (spaces.Discrete(self.grid.width), spaces.Discrete(self.grid.height))
            )

        self.possible_agents = tuple(str(i) for i in range(self.NUM_AGENTS))
        # s = Tuple[Coord, Direction, Coord, Direction, Coord, Coord, Coord, int]
        # e_coord, e_dir, p_coord, p_dir, e_start, p_start, e_goal, max_sp
        self.state_space = spaces.Tuple(
            (
                _coord_space(),
                spaces.Discrete(len(Direction)),
                _coord_space(),
                spaces.Discrete(len(Direction)),
                _coord_space(),
                _coord_space(),
                _coord_space(),
                spaces.Discrete(self._max_sp_distance + 1),
            )
        )
        self.action_spaces = {
            i: spaces.Discrete(len(Direction)) for i in self.possible_agents
        }
        # o = Tuple[Tuple[WallObs, seen , heard], Coord, Coord, Coord]
        # Wall obs, seen, heard, e_start, p_start, e_goal/blank
        self.observation_spaces = {
            i: spaces.Tuple(
                (
                    spaces.Tuple(tuple(spaces.Discrete(2) for _ in range(6))),
                    _coord_space(),
                    _coord_space(),
                    _coord_space(),
                )
            )
            for i in self.possible_agents
        }
        self.is_symmetric = False

    @property
    def grid(self) -> "PEGrid":
        """The underlying grid for this model instance."""
        return self._grid

    @property
    def reward_ranges(self) -> Dict[str, Tuple[float, float]]:
        max_reward = self.R_EVASION
        if self.use_progress_reward:
            max_reward += self.R_PROGRESS
        if self.normalize_reward:
            max_reward = self._get_normalized_reward(max_reward)
        return {i: (-max_reward, max_reward) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: PEState) -> List[str]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> PEState:
        return self._sample_initial_state(None, None, None)

    def sample_agent_initial_state(self, agent_id: str, obs: PEObs) -> PEState:
        if agent_id == self.EVADER_IDX:
            return self._sample_initial_state(
                evader_coord=obs[1], pursuer_coord=obs[2], goal_coord=obs[3]
            )
        return self._sample_initial_state(
            evader_coord=obs[1], pursuer_coord=obs[2], goal_coord=None
        )

    def _sample_initial_state(
        self,
        evader_coord: Optional[Coord],
        pursuer_coord: Optional[Coord],
        goal_coord: Optional[Coord],
    ) -> PEState:
        if evader_coord is None:
            evader_coord = self.rng.choice(self.grid.evader_start_coords)
        if pursuer_coord is None:
            pursuer_coord = self.rng.choice(self.grid.pursuer_start_coords)
        if goal_coord is None:
            goal_coord = self.rng.choice(self.grid.get_goal_coords(evader_coord))
        return PEState(
            evader_coord,
            INITIAL_DIR,
            pursuer_coord,
            INITIAL_DIR,
            evader_coord,
            pursuer_coord,
            goal_coord,
            self.grid.get_shortest_path_distance(evader_coord, goal_coord),
        )

    def sample_initial_obs(self, state: PEState) -> Dict[str, PEObs]:
        return self._get_obs(state)[0]

    def step(
        self, state: PEState, actions: Dict[str, PEAction]
    ) -> M.JointTimestep[PEState, PEObs]:
        assert all(0 <= a_i < len(Direction) for a_i in actions.values())
        next_state = self._get_next_state(state, actions)
        obs, evader_detected = self._get_obs(next_state)
        rewards = self._get_reward(state, next_state, evader_detected)
        all_done = self._is_done(next_state)
        terminated = {i: all_done for i in self.possible_agents}
        truncated = {i: False for i in self.possible_agents}
        info: Dict[str, Dict] = {i: {} for i in self.possible_agents}
        if all_done:
            for i, outcome in self._get_outcome(next_state).items():
                info[i]["outcome"] = outcome
        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, info
        )

    def _get_next_state(self, state: PEState, actions: Dict[str, PEAction]) -> PEState:
        evader_a = actions[str(self.EVADER_IDX)]
        pursuer_a = actions[str(self.PURSUER_IDX)]

        if (
            self.action_probs[self.EVADER_IDX] < 1.0
            and self.rng.random() > self.action_probs[self.EVADER_IDX]
        ):
            other_as = [a for a in range(len(Direction)) if a != evader_a]
            evader_a = self.rng.choice(other_as)

        if (
            self.action_probs[self.PURSUER_IDX]
            and self.rng.random() > self.action_probs[self.PURSUER_IDX]
        ):
            other_as = [a for a in range(len(Direction)) if a != pursuer_a]
            pursuer_a = self.rng.choice(other_as)

        pursuer_next_dir = Direction(ACTION_TO_DIR[pursuer_a][state.pursuer_dir])
        pursuer_next_coord = self.grid.get_next_coord(
            state.pursuer_coord, pursuer_next_dir, ignore_blocks=False
        )

        evader_next_coord = state.evader_coord
        evader_next_dir = Direction(ACTION_TO_DIR[evader_a][state.evader_dir])
        if pursuer_next_coord != state.evader_coord:
            evader_next_coord = self.grid.get_next_coord(
                state.evader_coord, evader_next_dir, ignore_blocks=False
            )

        min_sp_distance = min(
            state.min_goal_dist,
            self.grid.get_shortest_path_distance(
                evader_next_coord, state.evader_goal_coord
            ),
        )

        return PEState(
            evader_next_coord,
            evader_next_dir,
            pursuer_next_coord,
            pursuer_next_dir,
            state.evader_start_coord,
            state.pursuer_start_coord,
            state.evader_goal_coord,
            min_sp_distance,
        )

    def _get_obs(self, state: PEState) -> Tuple[Dict[str, PEObs], bool]:
        walls, seen, heard = self._get_agent_obs(
            state.evader_coord, state.evader_dir, state.pursuer_coord
        )
        evader_obs = (
            (*walls, seen, heard),
            state.evader_start_coord,
            state.pursuer_start_coord,
            state.evader_goal_coord,
        )

        walls, seen, heard = self._get_agent_obs(
            state.pursuer_coord, state.pursuer_dir, state.evader_coord
        )
        pursuer_obs = (
            (*walls, seen, heard),
            state.evader_start_coord,
            state.pursuer_start_coord,
            # pursuer doesn't observe evader goal coord
            (0, 0),
        )
        return {
            str(self.EVADER_IDX): evader_obs,
            str(self.PURSUER_IDX): pursuer_obs,
        }, bool(seen)

    def _get_agent_obs(
        self, agent_coord: Coord, agent_dir: Direction, opp_coord: Coord
    ) -> Tuple[Tuple[int, int, int, int], int, int]:
        adj_coords = self.grid.get_neighbours(
            agent_coord, ignore_blocks=True, include_out_of_bounds=True
        )
        walls: Tuple[int, int, int, int] = tuple(  # type: ignore
            int(not self.grid.coord_in_bounds(coord) or coord in self.grid.block_coords)
            for coord in adj_coords
        )
        seen = self._get_opponent_seen(agent_coord, agent_dir, opp_coord)
        dist = self.grid.manhattan_dist(agent_coord, opp_coord)
        heard = dist <= self.HEARING_DIST
        return walls, int(seen), int(heard)

    def _get_opponent_seen(
        self, ego_coord: Coord, ego_dir: Direction, opp_coord: Coord
    ) -> bool:
        fov = self.grid.get_fov(
            ego_coord, ego_dir, self.FOV_EXPANSION_INCR, self.max_obs_distance
        )
        return opp_coord in fov

    def _get_reward(
        self, state: PEState, next_state: PEState, evader_seen: bool
    ) -> Dict[str, float]:
        evader_coord = next_state.evader_coord
        pursuer_coord = next_state.pursuer_coord
        evader_goal_coord = next_state.evader_goal_coord

        evader_reward = 0.0
        if self.use_progress_reward and next_state.min_goal_dist < state.min_goal_dist:
            evader_reward += self.R_PROGRESS

        if evader_coord == pursuer_coord or evader_seen:
            evader_reward += self.R_CAPTURE
        elif evader_coord == evader_goal_coord:
            evader_reward += self.R_EVASION

        if self.normalize_reward:
            evader_reward = self._get_normalized_reward(evader_reward)

        return {
            str(self.EVADER_IDX): evader_reward,
            str(self.PURSUER_IDX): -evader_reward,
        }

    def _is_done(self, state: PEState) -> bool:
        evader_coord, pursuer_coord = state.evader_coord, state.pursuer_coord
        pursuer_dir = state.pursuer_dir
        return (
            evader_coord == pursuer_coord
            or evader_coord == state.evader_goal_coord
            or self._get_opponent_seen(pursuer_coord, pursuer_dir, evader_coord)
        )

    def _get_outcome(self, state: PEState) -> Dict[str, M.Outcome]:
        # Assuming this method is called on final timestep
        evader_coord, pursuer_coord = state.evader_coord, state.pursuer_coord
        evader_goal_coord = state.evader_goal_coord
        pursuer_dir = state.pursuer_dir
        evader_id, pursuer_id = str(self.EVADER_IDX), str(self.PURSUER_IDX)
        # check this first before relatively expensive detection check
        if evader_coord == pursuer_coord:
            return {evader_id: M.Outcome.LOSS, pursuer_id: M.Outcome.WIN}
        if evader_coord == evader_goal_coord:
            return {evader_id: M.Outcome.WIN, pursuer_id: M.Outcome.LOSS}
        if self._get_opponent_seen(pursuer_coord, pursuer_dir, evader_coord):
            return {evader_id: M.Outcome.LOSS, pursuer_id: M.Outcome.WIN}
        return {evader_id: M.Outcome.DRAW, pursuer_id: M.Outcome.DRAW}

    def _get_normalized_reward(self, reward: float) -> float:
        """Normalize reward in [-1, 1] interval."""
        diff = self._max_raw_return - self._min_raw_return
        return 2 * (reward - self._min_raw_return) / diff - 1


class PEGrid(Grid):
    """A grid for the Pursuit Evasion Problem."""

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        block_coords: Set[Coord],
        goal_coords_map: Dict[Coord, List[Coord]],
        evader_start_coords: List[Coord],
        pursuer_start_coords: List[Coord],
    ):
        super().__init__(grid_width, grid_height, block_coords)
        self._goal_coords_map = goal_coords_map
        self.evader_start_coords = evader_start_coords
        self.pursuer_start_coords = pursuer_start_coords
        self.shortest_paths = self.get_all_shortest_paths(self.all_goal_coords)

    @property
    def all_goal_coords(self) -> List[Coord]:
        """The list of all evader goal locations."""
        all_locs = set()
        for v in self._goal_coords_map.values():
            all_locs.update(v)
        return list(all_locs)

    def get_goal_coords(self, evader_start_coord: Coord) -> List[Coord]:
        """Get list of possible evader goal coords for given start coords."""
        return self._goal_coords_map[evader_start_coord]

    def get_shortest_path_distance(self, coord: Coord, dest: Coord) -> int:
        """Get the shortest path distance from coord to destination."""
        return int(self.shortest_paths[dest][coord])

    def get_max_shortest_path_distance(self) -> int:
        """Get max shortest path distance between any start and goal coords."""
        max_dist = 0
        for start_coord, dest_coords in self._goal_coords_map.items():
            max_dist = max(
                max_dist,
                int(
                    max(
                        [
                            self.get_shortest_path_distance(start_coord, dest)
                            for dest in dest_coords
                        ]
                    )
                ),
            )
        return max_dist

    def get_ascii_repr(
        self,
        goal_coord: Union[None, Coord, List[Coord]],
        evader_coord: Union[None, Coord, List[Coord]],
        pursuer_coord: Union[None, Coord, List[Coord]],
    ) -> str:
        """Get ascii repr of grid."""
        if goal_coord is None:
            goal_coords = set()
        elif isinstance(goal_coord, list):
            goal_coords = set(goal_coord)
        else:
            goal_coords = {goal_coord}

        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                coord = (col, row)
                if coord in goal_coords:
                    row_repr.append("G")
                elif coord in self.block_coords:
                    row_repr.append("#")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)

        if evader_coord is None:
            evader_coord = []
        elif not isinstance(evader_coord, List):
            evader_coord = [evader_coord]

        for coord in evader_coord:
            grid_repr[coord[1]][coord[0]] = "R"

        if pursuer_coord is None:
            pursuer_coord = []
        elif not isinstance(pursuer_coord, List):
            pursuer_coord = [pursuer_coord]

        for coord in pursuer_coord:
            if coord in evader_coord:
                grid_repr[coord[1]][coord[0]] = "X"
            else:
                grid_repr[coord[1]][coord[0]] = "C"

        return "\n".join([" ".join(r) for r in grid_repr])

    def get_init_ascii_repr(self) -> str:
        """Get ascii repr of initial grid."""
        return self.get_ascii_repr(
            self.all_goal_coords, self.evader_start_coords, self.pursuer_start_coords
        )

    def get_fov(
        self,
        origin: Coord,
        direction: Direction,
        widening_increment: int,
        max_depth: int,
    ) -> Set[Coord]:
        """Get the Field of vision from origin looking in given direction.

        Uses BFS starting from origin and expanding in the direction, while
        accounting for obstacles blocking the field of view,
        to get the field of vision.
        """
        assert widening_increment > 0
        assert max_depth > 0
        fov = {origin}

        frontier_queue: Deque[Coord] = deque([origin])
        visited = {origin}

        while len(frontier_queue):
            coord = frontier_queue.pop()

            for next_coord in self._get_fov_successors(
                origin, direction, coord, widening_increment, max_depth
            ):
                if next_coord not in visited:
                    visited.add(next_coord)
                    frontier_queue.append(next_coord)
                    fov.add(next_coord)
        return fov

    def _get_fov_successors(
        self,
        origin: Coord,
        direction: Direction,
        coord: Coord,
        widening_increment: int,
        max_depth: int,
    ) -> List[Coord]:
        if direction in [Direction.NORTH, Direction.SOUTH]:
            depth = abs(origin[1] - coord[1])
        else:
            depth = abs(origin[0] - coord[0])

        if depth >= max_depth:
            return []

        successors = []
        forward_successor = self._get_fov_successor(coord, direction)
        if forward_successor is None:
            return []
        else:
            successors.append(forward_successor)

        # Expands FOV at depth 1 and then every widening_increment depth
        if depth != 1 and depth % widening_increment != 0:
            # Don't expand sideways
            return successors

        side_coords_list: List[Coord] = []

        if direction in [Direction.NORTH, Direction.SOUTH]:
            if 0 < coord[0] <= origin[0]:
                side_coords_list.append((coord[0] - 1, coord[1]))
            if origin[0] <= coord[0] < self.width - 1:
                side_coords_list.append((coord[0] + 1, coord[1]))
        else:
            if 0 < coord[1] <= origin[1]:
                side_coords_list.append((coord[0], coord[1] - 1))
            elif origin[1] <= coord[1] < self.height - 1:
                side_coords_list.append((coord[0], coord[1] + 1))

        side_successor: Optional[Coord] = None
        for side_coord in side_coords_list:
            if side_coord in self.block_coords:
                continue

            side_successor = self._get_fov_successor(side_coord, direction)
            if side_successor is not None:
                successors.append(side_successor)

        return successors

    def _get_fov_successor(self, coord: Coord, direction: Direction) -> Optional[Coord]:
        new_coord = self.get_next_coord(coord, direction, ignore_blocks=False)
        if new_coord == coord:
            # move in given direction is blocked or out-of-bounds
            return None
        return new_coord

    def get_max_fov_width(self, widening_increment: int, max_depth: int) -> int:
        """Get the maximum width of field of vision."""
        max_width = 1
        for d in range(max_depth):
            if d == 1 or d % widening_increment == 0:
                max_width += 2
        return min(max_depth, min(self.width, self.height))


def get_8x8_grid() -> PEGrid:
    """Generate the 8-by-8 PE grid layout.

    This is an approximate discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    - 0, 1, 2, 7, 8, 9 are possible evader start and goal locations
    - 5, 6 are possible pursuer start locations

    The evader start and goal locations are always on opposite sides of the
    map.
    """
    ascii_map = (
        "  9 #8 #"
        "# #    #"
        "# ## # 7"
        "  6   # "
        "# #5# # "
        "   #    "
        "#    #2 "
        "0 #1### "
    )

    return _convert_map_to_grid(ascii_map, 8, 8)


def get_16x16_grid() -> PEGrid:
    """Generate the 16-by-16 PE grid layout.

    This is an approximate discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    - 0, 1, 2, 7, 8, 9 are possible evader start and goal locations
    - 5, 6 are possible pursuer start locations

    The evader start and goal locations are always on opposite sides of the
    map.

    """
    ascii_map = (
        "  ## ####### ###"
        "    9##    8 ###"
        "##       # # ###"
        "## ## ##      ##"
        "## ## ##   ## ##"
        "#  #  ##   ## 7#"
        "# ## ###   #   #"
        "  #  6       # #"
        "#   ## ##  ##  #"
        "##   #5##  #   #"
        "## #   #     # #"
        "   # #   ##    #"
        " # # ##  ##   ##"
        "0#       #     #"
        "   ### #   ##2  "
        "###    1 #####  "
    )
    return _convert_map_to_grid(ascii_map, 16, 16)


def get_32x32_grid() -> PEGrid:
    """Generate the 32-by-32 PE grid layout.

    This is an approximate discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    - 0, 1, 2, 7, 8, 9 are possible evader start and goal locations
    - 5, 6 are possible pursuer start locations

    The evader start and goal locations are always on opposite sides of the
    map.

    """
    ascii_map = (
        "       #  ########### #         "
        "   #####  ########### #  #######"
        "      #   ######      8  #######"
        " #       9#               ######"
        " ##              #### ##  ######"
        " ##   #####  ##  #### ##   #####"
        " ##   #####  ##  ##        #####"
        "      ##### ####      ###  #####"
        " ### ###### ####      ###   ####"
        " ### #####  ####      ####  ####"
        " ##  ##### #####      ##### 7###"
        " ##  ####  #####       ####  ###"
        " #  #####  #####       ####  ###"
        " #  ##### ######       #     ###"
        " #       6               ##  ###"
        "    #       #   ##      ####  ##"
        "     #   ####  ###     ####   ##"
        "#### #   ####  ###    ####    ##"
        "#### #   ### 5####   ####     ##"
        "####  #      ####     ##  #   ##"
        "##### #      ####        ##   ##"
        "#                          ##  #"
        "         ###        ##    2 #  #"
        "  ###### ###      #### ## #    #"
        "########  ##      ####    #  #  "
        "########  ##       ####     ####"
        "###          ##   1##        ###"
        "          #  ####       #       "
        "   0  ###### ######   ####      "
        "  ########## #        #####     "
        "##########      ############    "
        "#                               "
    )
    return _convert_map_to_grid(ascii_map, 32, 32)


def _loc_to_coord(loc: int, grid_width: int) -> Coord:
    return (loc % grid_width, loc // grid_width)


def _convert_map_to_grid(
    ascii_map: str,
    height: int,
    width: int,
    block_symbol: str = "#",
    pursuer_start_symbols: Optional[Set[str]] = None,
    evader_start_symbols: Optional[Set[str]] = None,
    evader_goal_symbol_map: Optional[Dict] = None,
) -> PEGrid:
    assert len(ascii_map) == height * width

    if pursuer_start_symbols is None:
        pursuer_start_symbols = {"3", "4", "5", "6"}
    if evader_start_symbols is None:
        evader_start_symbols = {"0", "1", "2", "7", "8", "9"}
    if evader_goal_symbol_map is None:
        evader_goal_symbol_map = {
            "0": ["7", "8", "9"],
            "1": ["7", "8", "9"],
            "2": ["8", "9"],
            "7": ["0", "1"],
            "8": ["0", "1", "2"],
            "9": ["0", "1", "2"],
        }

    block_coords = set()
    evader_start_coords = []
    pursuer_start_coords = []
    evader_symbol_coord_map = {}

    for loc, symbol in enumerate(ascii_map):
        coord = _loc_to_coord(loc, width)
        if symbol == block_symbol:
            block_coords.add(coord)
        elif symbol in pursuer_start_symbols:
            pursuer_start_coords.append(coord)
        elif symbol in evader_start_symbols:
            evader_start_coords.append(coord)
            evader_symbol_coord_map[symbol] = coord

    evader_goal_coords_map = {}
    for start_symbol, goal_symbols in evader_goal_symbol_map.items():
        start_coord = evader_symbol_coord_map[start_symbol]
        evader_goal_coords_map[start_coord] = [
            evader_symbol_coord_map[symbol] for symbol in goal_symbols
        ]

    return PEGrid(
        grid_width=width,
        grid_height=height,
        block_coords=block_coords,
        goal_coords_map=evader_goal_coords_map,
        evader_start_coords=evader_start_coords,
        pursuer_start_coords=pursuer_start_coords,
    )


# grid_name: (grid_make_fn, step_limit)
SUPPORTED_GRIDS = {
    "8x8": (get_8x8_grid, 50),
    "16x16": (get_16x16_grid, 100),
    "32x32": (get_32x32_grid, 200),
}
