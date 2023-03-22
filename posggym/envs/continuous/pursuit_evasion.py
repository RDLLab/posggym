"""The Pursuit-Evasion Grid World Environment."""
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    Callable,
    cast,
)
import numpy as np
from gymnasium import spaces
import math
import posggym.model as M
from posggym.core import DefaultEnv

# from posggym.envs.grid_world.core import Coord, Direction, Grid
from posggym.envs.continuous.core import (
    Position,
    RectangularContinuousWorld,
    Object,
    position_to_array,
    single_item_to_position,
)
from posggym.utils import seeding


# State = (e_coord, e_dir, p_coord, p_dir, e_0_coord, p_0_coord, e_goal_coord)


class PEState(NamedTuple):
    """Environment state in Pursuit Evastion problem."""

    evader_coord: np.ndarray
    pursuer_coord: np.ndarray
    evader_start_coord: np.ndarray
    pursuer_start_coord: np.ndarray
    evader_goal_coord: np.ndarray
    min_goal_dist: int  # for evader


# Action = Direction of movement
# 0 = Forward, 1 = Backward, 2 = Left, 3 = Right
PEAction = List[float]

# E Obs = Tuple[WallObs, seen, heard, e_0_coord, p_0_coord, goal_coord]
#       = Tuple[Tuple[int, int, int, int, int, int], Coord, Coord, Coord]
# P Obs = Tuple[WallObs, seen, heard, e_0_coord, p_0_coord, blank_coord]
#       = Tuple[Tuple[int, int, int, int, int, int], Coord, Coord, Coord]
# Note, we use blank_coord for P Obs so Obs spaces are identical between the
# two agents. The blank_coord is always (0, 0).
PEEvaderObs = Tuple[Union[int, np.ndarray], ...]
PEPursuerObs = Tuple[Union[int, np.ndarray], ...]
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
    4. the `(x, y)` coordinate of the evader
    5. the `(x, y)` coordinate of the evader's start location
    6. the `(x, y)` coordinate of the pursuer's start location

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
         the supported grids, or a custom :class:`PEWorld` object (default = `"16x16"`).
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
    env = posgggym.make(
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
        "render_modes": ["human"],
        "render_fps": 15,
    }

    def __init__(
        self,
        grid: Union[str, "PEWorld"] = "16x16",
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

        self._max_obs_distance = max_obs_distance
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
    ) -> Tuple[Dict[M.AgentID, M.ObsType], Dict[M.AgentID, Dict]]:
        # reset renderer since goal location can change between episodes
        self._renderer = None
        return super().reset(seed=seed, options=options)

    def render(self):
        return self._render_img()

    def _render_img(self):
        if self.render_mode == "human":
            import posggym.envs.continuous.render as render_lib

            if self._renderer is None:
                self._renderer = render_lib.GWContinuousRender(
                    self.render_mode,
                    env_name="PursuitEvasionContinuous",
                    domain_max=self.model.grid.width,
                    render_fps=self.metadata["render_fps"],
                    num_colors=4,
                    arena_size=200,
                )

            pred = single_item_to_position(self._state.evader_coord)
            prey = single_item_to_position(self._state.pursuer_coord)

            colored_pred = (pred + (0,),)
            colored_prey = (prey + (1,),)

            num_agents = 2

            sizes = [self.model.grid.agent_size] * num_agents

            holonomic = [False] * 2

            self._renderer.clear_render()
            self._renderer.draw_arena()

            # self._renderer.render_lines(self._last_obs, self._state.predator_coords)

            self._renderer.draw_agents(
                colored_prey + colored_pred,
                sizes=sizes,
                is_holonomic=holonomic,
                alpha=255,
            )
            self._renderer.draw_blocks(self.model.grid.block_coords)
            self._renderer.render()

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
        grid: Union[str, "PEWorld"],
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
        self._action_probs = action_probs
        self._max_obs_distance = max_obs_distance
        self._normalize_reward = normalize_reward
        self._use_progress_reward = use_progress_reward

        self._max_sp_distance = self._grid.get_max_shortest_path_distance()
        self._max_raw_return = self.R_EVASION
        if self._use_progress_reward:
            self._max_raw_return += (self._max_sp_distance + 1) * self.R_PROGRESS
        self._min_raw_return = -self._max_raw_return

        def _coord_space():
            return spaces.Box(
                low=np.array([0, 0, -2 * math.pi], dtype=np.float32),
                high=np.array(
                    [self.grid.width, self.grid.height, 2 * math.pi], dtype=np.float32
                ),
            )

        self.possible_agents = tuple(str(i) for i in range(self.NUM_AGENTS))
        # s = Tuple[Coord, Direction, Coord, Direction, Coord, Coord, Coord, int]
        # e_coord, e_dir, p_coord, p_dir, e_start, p_start, e_goal, max_sp
        self.state_space = spaces.Tuple(
            (
                _coord_space(),
                _coord_space(),
                _coord_space(),
                _coord_space(),
                _coord_space(),
                spaces.Discrete(self._max_sp_distance + 1),
            )
        )

        self.use_holonomic = True
        factor = 1 if self.use_holonomic else 0.5 * math.pi
        self.action_spaces = {
            i: spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32) * factor,
                high=np.array([1.0, 1.0], dtype=np.float32) * factor,
            )
            for i in self.possible_agents
        }

        # o = Tuple[Tuple[WallObs, seen , heard], Coord, Coord, Coord]
        # Wall obs, seen, heard, e_start, p_start, e_goal/blank
        self.n_lines = 4
        self.obs_distance = 1
        self.observation_spaces = {
            i: spaces.Tuple(
                (
                    spaces.Box(
                        low=np.array([0] * self.n_lines, dtype=np.float32),
                        high=np.array(
                            [self.obs_distance] * self.n_lines, dtype=np.float32
                        ),
                    ),
                    spaces.Discrete(2),
                    spaces.Discrete(2),
                    _coord_space(),
                    _coord_space(),
                    _coord_space(),
                )
            )
            for i in self.possible_agents
        }
        self.is_symmetric = False

    @property
    def grid(self) -> "PEWorld":
        """The underlying grid for this model instance."""
        return self._grid

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[float, float]]:
        max_reward = self.R_CAPTURE
        if self._use_progress_reward:
            max_reward += self.R_PROGRESS
        if self._normalize_reward:
            max_reward = self._get_normalized_reward(max_reward)
        return {i: (-max_reward, max_reward) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: PEState) -> List[M.AgentID]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> PEState:
        return self._sample_initial_state(None, None, None)

    def sample_agent_initial_state(self, agent_id: M.AgentID, obs: PEObs) -> PEState:
        if agent_id == self.EVADER_IDX:
            return self._sample_initial_state(
                evader_coord=cast(np.ndarray, obs[1]),
                pursuer_coord=cast(np.ndarray, obs[2]),
                goal_coord=cast(np.ndarray, obs[3]),
            )
        return self._sample_initial_state(
            evader_coord=cast(np.ndarray, obs[1]),
            pursuer_coord=cast(np.ndarray, obs[2]),
            goal_coord=None,
        )

    def _sample_initial_state(
        self,
        evader_coord: Optional[np.ndarray],
        pursuer_coord: Optional[np.ndarray],
        goal_coord: Optional[np.ndarray],
    ) -> PEState:
        if evader_coord is None:
            evader_coord_pos = self.rng.choice(self.grid.evader_start_coords)
            evader_coord_pos = cast(List[Position], [evader_coord_pos])
            evader_coord = position_to_array(evader_coord_pos)
        if pursuer_coord is None:
            pursuer_coord_pos = self.rng.choice(self.grid.pursuer_start_coords)
            pursuer_coord_pos = cast(List[Position], [pursuer_coord_pos])
            pursuer_coord = position_to_array(pursuer_coord_pos)
        if goal_coord is None:
            evader_coord_pos = single_item_to_position(evader_coord)
            goal_coord_pos = self.rng.choice(
                self.grid.get_goal_coords(evader_coord_pos)
            )
            goal_coord_pos = cast(List[Position], [goal_coord_pos])
            goal_coord = position_to_array(goal_coord_pos)
        return PEState(
            evader_coord,
            pursuer_coord,
            evader_coord,
            pursuer_coord,
            goal_coord,
            self.grid.get_shortest_path_distance(
                single_item_to_position(evader_coord),
                single_item_to_position(goal_coord),
            ),
        )

    def sample_initial_obs(self, state: PEState) -> Dict[M.AgentID, PEObs]:
        return self._get_obs(state)[0]

    def step(
        self, state: PEState, actions: Dict[M.AgentID, PEAction]
    ) -> M.JointTimestep[PEState, PEObs]:
        # assert all(0 <= a_i < len(Direction) for a_i in actions.values())
        next_state = self._get_next_state(state, actions)
        obs, evader_detected = self._get_obs(next_state)
        rewards = self._get_reward(state, next_state, evader_detected)
        all_done = self._is_done(next_state)
        terminated = {i: all_done for i in self.possible_agents}
        truncated = {i: False for i in self.possible_agents}
        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}
        if all_done:
            for i, outcome in self._get_outcome(next_state).items():
                info[i]["outcome"] = outcome
        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, info
        )

    def _get_next_state(
        self, state: PEState, actions: Dict[M.AgentID, PEAction]
    ) -> PEState:
        evader_a = actions[str(self.EVADER_IDX)]
        pursuer_a = actions[str(self.PURSUER_IDX)]

        if (
            self._action_probs[self.EVADER_IDX] < 1.0
            and self.rng.random() > self._action_probs[self.EVADER_IDX]
        ):
            evader_a = self.action_spaces[str(self.EVADER_IDX)].sample()

        if (
            self._action_probs[self.PURSUER_IDX]
            and self.rng.random() > self._action_probs[self.PURSUER_IDX]
        ):
            pursuer_a = self.action_spaces[str(self.PURSUER_IDX)].sample()

        pursuer_coord = single_item_to_position(state.pursuer_coord)
        evader_coord = single_item_to_position(state.evader_coord)
        evader_goal_coord = single_item_to_position(state.evader_goal_coord)

        _pursuer_next_coord, succ = self.grid._get_next_coord(
            pursuer_coord, pursuer_a, ignore_blocks=False
        )
        pursuer_next_coord = _pursuer_next_coord if succ else pursuer_coord

        evader_next_coord = single_item_to_position(state.evader_coord)
        if not self.grid.agents_collide(pursuer_next_coord, evader_coord):
            _evader_next_coord, succ = self.grid._get_next_coord(
                evader_coord, evader_a, ignore_blocks=False
            )

            if succ:
                evader_next_coord = _evader_next_coord

        min_sp_distance = min(
            state.min_goal_dist,
            self.grid.get_shortest_path_distance(evader_next_coord, evader_goal_coord),
        )

        return PEState(
            position_to_array([evader_next_coord]),
            position_to_array([pursuer_next_coord]),
            state.evader_start_coord,
            state.pursuer_start_coord,
            state.evader_goal_coord,
            min_sp_distance,
        )

    def _get_obs(self, state: PEState) -> Tuple[Dict[M.AgentID, PEObs], bool]:
        evader_coord = single_item_to_position(state.evader_coord)
        pursuer_coord = single_item_to_position(state.pursuer_coord)
        walls, seen, heard = self._get_agent_obs(evader_coord, pursuer_coord)
        evader_obs: PEEvaderObs = (
            np.array(walls, dtype=np.float32),
            seen,
            heard,
            state.evader_start_coord,
            state.pursuer_start_coord,
            state.evader_goal_coord,
        )

        walls, seen, heard = self._get_agent_obs(pursuer_coord, evader_coord)
        pursuer_obs: PEPursuerObs = (
            np.array(walls, dtype=np.float32),
            seen,
            heard,
            state.evader_start_coord,
            state.pursuer_start_coord,
            # pursuer doesn't observe evader goal coord
            np.ndarray([0, 0, 0], dtype=np.float32),
        )

        return {
            str(self.EVADER_IDX): evader_obs,
            str(self.PURSUER_IDX): pursuer_obs,
        }, bool(seen)

    def _get_agent_obs(
        self, agent_coord: Position, opp_coord: Position
    ) -> Tuple[List[float], int, int]:
        # agent_dir = agent_coord[2]
        seen = False
        wall_obs: List[float] = []
        for i in range(self.n_lines):
            angle = 2 * math.pi * i / self.n_lines

            closest_index, closest_wall_dist = self.grid.check_collision_ray(
                agent_coord, self.obs_distance, angle, (), only_walls=True
            )
            if closest_index is None:
                wall_obs.append(self.obs_distance)
            elif closest_index == -1:
                wall_obs.append(closest_wall_dist)
            else:
                raise AssertionError("Should only see walls")

        seen = self._get_opponent_seen(agent_coord, opp_coord)
        dist = self.grid.manhattan_dist(agent_coord, opp_coord)
        heard = dist <= self.HEARING_DIST
        return wall_obs, int(seen), int(heard)

    def _get_opponent_seen(self, ego_coord: Position, opp_coord: Position) -> bool:
        angle, dist = self.relative_positioning(ego_coord, opp_coord)
        print(angle, dist)
        return abs(angle) < np.pi / 4 and dist < self._max_obs_distance

    def relative_positioning(
        self, agent_i: Position, agent_j: Position
    ) -> Tuple[float, float]:
        dist = self.grid.euclidean_dist(agent_i, agent_j)
        yaw = agent_i[2]

        # Rotation matrix of yaw
        R = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])

        T_p = np.array([agent_i[0] - agent_j[0], agent_i[1] - agent_j[1]])
        T_p = R.dot(T_p)
        alpha = math.atan2(T_p[1], T_p[0])

        return (alpha, dist)

    def _get_reward(
        self, state: PEState, next_state: PEState, evader_seen: bool
    ) -> Dict[M.AgentID, float]:
        evader_coord = single_item_to_position(next_state.evader_coord)
        pursuer_coord = single_item_to_position(next_state.pursuer_coord)
        evader_goal_coord = single_item_to_position(next_state.evader_goal_coord)

        evader_reward = 0.0
        if self._use_progress_reward and next_state.min_goal_dist < state.min_goal_dist:
            evader_reward += self.R_PROGRESS

        if self.grid.agents_collide(evader_coord, pursuer_coord) or evader_seen:
            evader_reward += self.R_CAPTURE
        elif self.grid.agents_collide(evader_coord, evader_goal_coord):
            evader_reward += self.R_EVASION

        if self._normalize_reward:
            evader_reward = self._get_normalized_reward(evader_reward)

        return {
            str(self.EVADER_IDX): evader_reward,
            str(self.PURSUER_IDX): -evader_reward,
        }

    def _is_done(self, state: PEState) -> bool:
        evader_coord = single_item_to_position(state.evader_coord)
        pursuer_coord = single_item_to_position(state.pursuer_coord)
        evader_goal_coord = single_item_to_position(state.evader_goal_coord)
        return (
            self.grid.agents_collide(evader_coord, pursuer_coord)
            or self.grid.agents_collide(evader_coord, evader_goal_coord)
            or self._get_opponent_seen(pursuer_coord, evader_coord)
        )

    def _get_outcome(self, state: PEState) -> Dict[M.AgentID, M.Outcome]:
        # Assuming this method is called on final timestep
        evader_coord, pursuer_coord = single_item_to_position(
            state.evader_coord
        ), single_item_to_position(state.pursuer_coord)
        evader_goal_coord = state.evader_goal_coord
        evader_id, pursuer_id = str(self.EVADER_IDX), str(self.PURSUER_IDX)
        # check this first before relatively expensive detection check
        if self.grid.agents_collide(evader_coord, pursuer_coord):
            return {evader_id: M.Outcome.LOSS, pursuer_id: M.Outcome.WIN}
        if self.grid.agents_collide(
            evader_coord, single_item_to_position(evader_goal_coord)
        ):
            return {evader_id: M.Outcome.WIN, pursuer_id: M.Outcome.LOSS}
        if self._get_opponent_seen(pursuer_coord, evader_coord):
            return {evader_id: M.Outcome.LOSS, pursuer_id: M.Outcome.WIN}
        return {evader_id: M.Outcome.DRAW, pursuer_id: M.Outcome.DRAW}

    def _get_normalized_reward(self, reward: float) -> float:
        """Normalize reward in [-1, 1] interval."""
        diff = self._max_raw_return - self._min_raw_return
        return 2 * (reward - self._min_raw_return) / diff - 1


class PEWorld(RectangularContinuousWorld):
    """A grid for the Pursuit Evasion Problem."""

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        block_coords: List[Object],
        goal_coords_map: Dict[Position, List[Position]],
        evader_start_coords: List[Position],
        pursuer_start_coords: List[Position],
    ):
        super().__init__(grid_width, grid_height, block_coords)

        self._goal_coords_map = goal_coords_map
        self.evader_start_coords = evader_start_coords
        self.pursuer_start_coords = pursuer_start_coords
        self.shortest_paths = self.get_all_shortest_paths(evader_start_coords)

    @property
    def all_goal_coords(self) -> List[Position]:
        """The list of all evader goal locations."""
        all_locs = set()
        for v in self._goal_coords_map.values():
            all_locs.update(v)
        return list(all_locs)

    def get_goal_coords(self, evader_start_coord: Position) -> List[Position]:
        """Get list of possible evader goal coords for given start coords."""
        return self._goal_coords_map[evader_start_coord]

    def get_shortest_path_distance(self, coord: Position, dest: Position) -> int:
        """Get the shortest path distance from coord to destination."""
        coord_c = self.convert_position_to_coordinate(coord)
        dest_c = self.convert_position_to_coordinate(dest)
        return int(self.shortest_paths[dest_c][coord_c])

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

    def get_max_fov_width(self, widening_increment: int, max_depth: int) -> float:
        """Get the maximum width of field of vision."""
        max_width = 1
        for d in range(max_depth):
            if d == 1 or d % widening_increment == 0:
                max_width += 2
        (_, max_x), (_, max_y) = self.get_bounds()
        return min(max_depth, min(max_x, max_y))


def get_8x8_grid() -> PEWorld:
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


def get_16x16_grid() -> PEWorld:
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


def get_32x32_grid() -> PEWorld:
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


def _loc_to_coord(loc: int, grid_width: int) -> Position:
    return (loc % grid_width + 0.5, loc // grid_width + 0.5, 0)


def _convert_map_to_grid(
    ascii_map: str,
    height: int,
    width: int,
    block_symbol: str = "#",
    pursuer_start_symbols: Optional[Set[str]] = None,
    evader_start_symbols: Optional[Set[str]] = None,
    evader_goal_symbol_map: Optional[Dict] = None,
) -> PEWorld:
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

    block_coords: List[Object] = []
    evader_start_coords = []
    pursuer_start_coords = []
    evader_symbol_coord_map = {}

    for loc, symbol in enumerate(ascii_map):
        coord = _loc_to_coord(loc, width)
        if symbol == block_symbol:
            # The radius of the block is slightly smaller, otherwise
            # agents get stuck
            block_coords.append(((coord[0], coord[1], 0), 0.4))
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

    return PEWorld(
        grid_width=width,
        grid_height=height,
        block_coords=block_coords,
        goal_coords_map=evader_goal_coords_map,
        evader_start_coords=evader_start_coords,
        pursuer_start_coords=pursuer_start_coords,
    )


# grid_name: (grid_make_fn, step_limit)
SUPPORTED_GRIDS: Dict[str, Tuple[Callable[[], PEWorld], int]] = {
    "8x8": (get_8x8_grid, 50),
    "16x16": (get_16x16_grid, 100),
    "32x32": (get_32x32_grid, 200),
}
