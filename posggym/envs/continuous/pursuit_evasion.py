"""The Pursuit-Evasion World World Environment."""
import math
from itertools import product
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pymunk
from gymnasium import spaces
from pymunk import Vec2d

import posggym.model as M
from posggym import logger
from posggym.core import DefaultEnv
from posggym.envs.continuous.core import (
    CircleEntity,
    CollisionType,
    Line,
    PMBodyState,
    FloatCoord,
    Position,
    SquareContinuousWorld,
    clip_actions,
    parse_world_str_interior_walls,
    array_to_position,
)
from posggym.utils import seeding


class PEState(NamedTuple):
    """Environment state in Pursuit Evastion problem."""

    evader_state: np.ndarray
    pursuer_state: np.ndarray
    evader_start_coord: np.ndarray
    pursuer_start_coord: np.ndarray
    evader_goal_coord: np.ndarray
    min_goal_dist: int  # for evader


PEAction = np.ndarray

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

    An adversarial continuous world problem involving two agents: an evader and a
    pursuer. The evader's goal is to reach a goal location, on the other side of
    the grid, while the goal of the pursuer is to spot the evader before it reaches
    it's goal. The evader is considered caught if it is observed by the pursuer, or
    occupies the same location. The evader and pursuer have knowledge of each others
    starting locations. However, only the evader has knowledge of it's goal location.
    The pursuer only knowns that the evader's goal location is somewhere on the opposite
    side of the world to the evaders start location.

    This environment requires each agent to reason about the which path the other agent
    will take through the environment.

    Possible Agents
    ---------------
    - Evader = `"0"`
    - Pursuer = `"1"`

    State Space
    -----------
    Each state is made up of:

    0. the `(x, y, yaw)` coordinate of the evader
    1. the `(x, y, yaw)` coordinate of the pursuer
    2. the `(x, y, yaw)` coordinate of the evader's start location
    3. the `(x, y, yaw)` coordinate of the pursuer's start location
    4. the `(x, y, yaw)` coordinate of the evader's goal
    5. the minimum distance to it's goal along the shortest discrete path achieved
       by the evader in the current episode (this is needed to correctly reward
       the agent for making progress.)

    Action Space
    ------------
    Each agent has 2 actions, which are the angular and linear velocity.

    Observation Space
    -----------------
    Each agent observes:

    1. Each agent observes a local sector around themselves as a vector. The angle of
    this sector is between -fov and fov. This observation is done by a series of
    n_sensors' lines starting at the agent which extend for a distance of 'obs_dist'.
    For each line the agent observes the closest entity (wall, pursuer, evader) along
    the line. This table enumerates the first part of the observation space:

    |        Index: [start, end)        | Description                       | Values |
    | :-------------------------------: | --------------------------------: | :----: |
    |           0 - n_sensors           | Wall distance for each sensor     | [0, d] |
    |    n_sensors - (2 * n_sensors)    | Other agent dist for each sensor  | [0, d] |

    Where `d = obs_dist`.

    If an entity is not observed (i.e. there is none along the sensor's line or it
    isn't the closest entity to the observing agent along the line), The distance will
    be 1.

    The sensor reading ordering is relative to the agent's direction. I.e. the values
    for the first sensor at indices `0`, `n_sensors`, `2*n_sensors` correspond to the
    distance reading to a wall/obstacle, predator, and prey, respectively, in the
    direction the agent is facing.

    2. whether they see the other agent in a cone in front of them (`1`) or not (`0`).
       The cone projects forward up to 'max_obs_distance' (default=`12`) cells in front
       of the agent.
    3. whether they hear the other agent (`1`) or not (`0`). The other agent is heard if
       they are within distance 2 from the agent in any direction.
    4. the `(x, y, yaw)` coordinate of the evader's start location,
    5. the `(x, y, yaw)` coordinate of the pursuer's start location,
    6. Evader: the `(x, y, 0)` coordinate of the evader's goal location.
       Pursuer: blank coordinate `(0, 0, 0)`.

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
    (it can be disabled by the `use_progress_reward` parameter.)+

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
    be adjusted when using larger worlds (this can be done by manually specifying a
    value for `max_episode_steps` when creating the environment with `posggym.make`).

    Arguments
    ---------

    - `world` - the world layout to use. This can either be a string specifying one of
         the supported worlds, or a custom :class:`PEWorld` object
         (default = `"16x16"`).
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
    - `fov` - the Field of View of the agent in radians. This will determine the
        sector of which the agent can see. This FOV will be relative to the angle
        of the agent.

    Available variants
    ------------------

    The PursuitEvasionContinuous environment comes with a number of pre-built world
    layouts which can be passed as an argument to `posggym.make`, to create
    different worlds.

    | World name         | World size |
    |-------------------|-----------|
    | `5x5`             | 8x8       |
    | `16x16`           | 16x16     |
    | `32x32`           | 32x32     |

    For example to use the PursuitEvasionContinuous environment with the `32x32` world
    layout, and episode step limit of 200, and the default values for the other
    parameters you would use:

    ```python
    import posggym
    env = posggym.make(
        'PursuitEvasionContinuous-v0',
        max_episode_steps=200,
        world="32x32",
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
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }

    def __init__(
        self,
        world: Union[str, "PEWorld"] = "16x16",
        action_probs: Union[float, Tuple[float, float]] = 1.0,
        max_obs_distance: int = 12,
        normalize_reward: bool = True,
        use_progress_reward: bool = True,
        fov: float = np.pi / 2,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        model = PursuitEvasionModel(
            world,
            action_probs=action_probs,
            max_obs_distance=max_obs_distance,
            normalize_reward=normalize_reward,
            use_progress_reward=use_progress_reward,
            fov=fov,
            **kwargs,
        )
        super().__init__(model, render_mode=render_mode)

        self._max_obs_distance = max_obs_distance
        self.renderer = None
        self._agent_imgs = None
        self.window_surface = None
        self.clock = None
        self.window_size = 600
        self.draw_options = None
        self.world = None
        self.fov = fov

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[M.AgentID, M.ObsType], Dict[M.AgentID, Dict]]:
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
        if self.render_mode in ("human", "rgb_array"):
            return self._render_img()
        else:
            logger.warn(
                "You are calling render method on an invalid render mode"
                'Continuous environments currently only support "human" or'
                '"rgb_array" render modes.'
                "You can specify the render_mode at initialization, "
                f'e.g. posggym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

    def _render_img(self):
        import pygame
        from pymunk import pygame_util

        model = cast(PursuitEvasionModel, self.model)
        state = cast(PEState, self.state)
        scale_factor = self.window_size / model.world.size

        if self.window_surface is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption(self.__class__.__name__)
                self.window_surface = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
            else:
                self.window_surface = pygame.Surface(
                    (self.window_size, self.window_size)
                )
            # Turn off alpha since we don't use it.
            self.window_surface.set_alpha(None)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.draw_options is None:
            pygame_util.positive_y_is_up = False
            self.draw_options = pygame_util.DrawOptions(self.window_surface)
            self.draw_options.transform = pymunk.Transform.scaling(scale_factor)
            # don't render collision lines
            self.draw_options.flags = (
                pygame_util.DrawOptions.DRAW_SHAPES
                | pygame_util.DrawOptions.DRAW_CONSTRAINTS
            )

        if self.world is None:
            # get copy of model world, so we can use it for rendering without
            # affecting the original
            self.world = model.world.copy()

        self.world.set_entity_state("pursuer", state.pursuer_state)
        self.world.set_entity_state("evader", state.evader_state)

        # Need to do this for space to update with changes
        self.world.space.step(0.0001)

        # reset screen
        self.window_surface.fill(pygame.Color("white"))

        # draw sensor lines
        n_sensors = model.n_sensors
        info = [
            (model.EVADER_IDX, state.evader_state),
            (model.PURSUER_IDX, state.pursuer_state),
        ]

        for idx, p_state in info:
            obs_i = self._last_obs[str(idx)][0]
            x, y, agent_angle = p_state[:3]

            angles = np.linspace(
                -self.fov / 2, self.fov / 2, n_sensors, endpoint=False, dtype=np.float32
            )
            for k in range(n_sensors):
                dist = min([obs_i[k], obs_i[n_sensors + k]])

                angle = angles[k] + agent_angle
                end_x = x + dist * math.cos(angle)
                end_y = y + dist * math.sin(angle)
                scaled_start = (int(x * scale_factor), int(y * scale_factor))
                scaled_end = int(end_x * scale_factor), (end_y * scale_factor)

                pygame.draw.line(
                    self.window_surface,
                    pygame.Color("red"),
                    scaled_start,
                    scaled_end,
                )

        x, y = state.evader_goal_coord
        scaled_center = (int(x * scale_factor), int(y * scale_factor))
        scaled_r = int(model.world.agent_radius * scale_factor)
        pygame.draw.circle(
            self.window_surface,
            pygame.Color(model.GOAL_COLOR),
            scaled_center,
            scaled_r,
        )

        self.world.space.debug_draw(self.draw_options)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
        )

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

    HEARING_DIST = 2

    EVADER_COLOR = (55, 155, 205, 255)  # Blueish
    PURSUER_COLOR = (255, 55, 55, 255)  # purpleish
    GOAL_COLOR = (55, 100, 155, 255)  # Blueish

    def __init__(
        self,
        world: Union[str, "PEWorld"],
        action_probs: Union[float, Tuple[float, float]] = 1.0,
        max_obs_distance: int = 12,
        normalize_reward: bool = True,
        use_progress_reward: bool = True,
        fov: float = np.pi / 2,
    ):
        if isinstance(world, str):
            assert world in SUPPORTED_WORLDS, (
                f"Unsupported world name '{world}'. If world is a string it must be "
                f"one of: {SUPPORTED_WORLDS.keys()}."
            )
            world = SUPPORTED_WORLDS[world][0]()

        if isinstance(action_probs, float):
            action_probs = (action_probs, action_probs)

        self.world = world
        self._action_probs = action_probs
        self._max_obs_distance = max_obs_distance
        self._normalize_reward = normalize_reward
        self._use_progress_reward = use_progress_reward
        self.fov = fov

        self._max_sp_distance = self.world.get_max_shortest_path_distance()
        self._max_raw_return = self.R_EVASION
        if self._use_progress_reward:
            self._max_raw_return += (self._max_sp_distance + 1) * self.R_PROGRESS
        self._min_raw_return = -self._max_raw_return

        def _coord_space():
            return spaces.Box(
                low=np.array([0, 0], dtype=np.float32),
                high=np.array([self.world.size, self.world.size], dtype=np.float32),
            )

        self.possible_agents = tuple(str(i) for i in range(self.NUM_AGENTS))
        # s = Tuple[Coord, Direction, Coord, Direction, Coord, Coord, Coord, int]
        # e_coord, e_dir, p_coord, p_dir, e_start, p_start, e_goal, max_sp
        self.state_space = spaces.Tuple(
            (
                PMBodyState.get_space(self.world.size),
                PMBodyState.get_space(self.world.size),
                _coord_space(),
                _coord_space(),
                _coord_space(),
                spaces.Discrete(self._max_sp_distance + 1),
            )
        )

        self.dyaw_limit = math.pi / 10
        self.action_spaces = {
            i: spaces.Box(
                low=np.array([-self.dyaw_limit, 0.0], dtype=np.float32),
                high=np.array([self.dyaw_limit, 1.0], dtype=np.float32),
            )
            for i in self.possible_agents
        }

        # Observes entity and distance to entity along a n_sensors rays from the agent
        # 0 to n_sensors = wall distance obs
        # n_sensors to (2 * n_sensors) = evaer dist
        self.n_sensors = 8
        self.obs_dist = self._max_obs_distance
        self.obs_dim = self.n_sensors * 2

        # o = Tuple[Tuple[WallObs, seen , heard], Coord, Coord, Coord]
        # Wall obs, seen, heard, e_start, p_start, e_goal/blank
        self.observation_spaces = {
            i: spaces.Tuple(
                (
                    spaces.Box(
                        low=0.0,
                        high=self.obs_dist,
                        shape=(self.obs_dim,),
                        dtype=np.float32,
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

        # Add physical entities to the world
        self.world.add_entity("pursuer", None, color=self.PURSUER_COLOR)
        self.world.add_entity("evader", None, color=self.EVADER_COLOR)

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
        evader_coord = self.rng.choice(self.world.evader_start_coords)
        evader_state = np.zeros(PMBodyState.num_features(), dtype=np.float32)
        evader_state[:2] = evader_coord

        pursuer_coord = self.rng.choice(self.world.pursuer_start_coords)
        pursuer_state = np.zeros(PMBodyState.num_features(), dtype=np.float32)
        pursuer_state[:2] = pursuer_coord

        goal_coord = self.rng.choice(self.world.get_goal_coords(evader_coord))
        return PEState(
            evader_state,
            pursuer_state,
            evader_state[:2],
            pursuer_state[:2],
            np.array(goal_coord, dtype=np.float32),
            self.world.get_shortest_path_distance(evader_coord, goal_coord),
        )

    def sample_initial_obs(self, state: PEState) -> Dict[M.AgentID, PEObs]:
        return self._get_obs(state)[0]

    def step(
        self, state: PEState, actions: Dict[M.AgentID, PEAction]
    ) -> M.JointTimestep[PEState, PEObs]:
        clipped_actions = clip_actions(actions, self.action_spaces)
        next_state = self._get_next_state(state, clipped_actions)
        obs, evader_seen = self._get_obs(next_state)
        rewards = self._get_reward(state, next_state, evader_seen)
        all_done = self._is_done(next_state, evader_seen)
        terminated = {i: all_done for i in self.possible_agents}
        truncated = {i: False for i in self.possible_agents}
        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}
        if all_done:
            for i, outcome in self._get_outcome(next_state, evader_seen).items():
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

        self.world.set_entity_state("pursuer", state.pursuer_state)
        self.world.set_entity_state("evader", state.evader_state)

        pursuer_angle = state.pursuer_state[2] + pursuer_a[0]
        pursuer_vel = pursuer_a[1] * Vec2d(1, 0).rotated(pursuer_angle)
        self.world.update_entity_state(
            "pursuer",
            angle=pursuer_angle,
            vel=pursuer_vel,
        )

        evader_angle = state.evader_state[2] + evader_a[0]
        evader_vel = evader_a[1] * Vec2d(1, 0).rotated(evader_angle)
        self.world.update_entity_state(
            "evader",
            angle=evader_angle,
            vel=evader_vel,
        )

        # simulate
        self.world.simulate(1.0 / 10, 10)

        pursuer_next_state = np.array(
            self.world.get_entity_state("pursuer"),
            dtype=np.float32,
        )
        evader_next_state = np.array(
            self.world.get_entity_state("evader"),
            dtype=np.float32,
        )

        evader_coord = (evader_next_state[0], evader_next_state[1])
        goal_coord = (state.evader_goal_coord[0], state.evader_goal_coord[1])

        min_sp_distance = min(
            state.min_goal_dist,
            self.world.get_shortest_path_distance(evader_coord, goal_coord),
        )

        return PEState(
            evader_next_state,
            pursuer_next_state,
            state.evader_start_coord,
            state.pursuer_start_coord,
            state.evader_goal_coord,
            min_sp_distance,
        )

    def _get_obs(self, state: PEState) -> Tuple[Dict[M.AgentID, PEObs], bool]:
        evader_pos = array_to_position(state.evader_state.squeeze())
        pursuer_pos = array_to_position(state.pursuer_state.squeeze())

        walls, seen, heard = self._get_agent_obs(evader_pos, pursuer_pos)
        evader_obs: PEEvaderObs = (
            np.array(walls, dtype=np.float32),
            seen,
            heard,
            state.evader_start_coord,
            state.pursuer_start_coord,
            state.evader_goal_coord,
        )

        walls, seen, heard = self._get_agent_obs(pursuer_pos, evader_pos)
        pursuer_obs: PEPursuerObs = (
            np.array(walls, dtype=np.float32),
            seen,
            heard,
            state.evader_start_coord,
            state.pursuer_start_coord,
            # pursuer doesn't observe evader goal coord
            np.array([0, 0], dtype=np.float32),
        )

        return {
            str(self.EVADER_IDX): evader_obs,
            str(self.PURSUER_IDX): pursuer_obs,
        }, bool(seen)

    def _get_agent_obs(
        self, agent_pos: Position, opp_pos: Position
    ) -> Tuple[np.ndarray, int, int]:
        ray_dists, ray_col_type = self.world.check_collision_circular_rays(
            agent_pos,
            self.obs_dist,
            self.n_sensors,
            np.array([opp_pos[:2]]),
            include_blocks=True,
            check_walls=True,
            use_relative_angle=True,
            angle_bounds=(-self.fov / 2, self.fov / 2),
        )

        obs = np.full((self.obs_dim,), self.obs_dist, dtype=np.float32)
        # Can bucket NONE type colisions with block/border/wall collisions, since these
        # will have dist of obs_dist and hence not affect final obs
        obs_type_idx = np.where(ray_col_type == CollisionType.AGENT.value, 0, 1)
        flat_obs_idx = np.ravel_multi_index(
            (obs_type_idx, np.arange(self.n_sensors)), dims=(2, self.n_sensors)
        )
        obs[flat_obs_idx] = ray_dists

        seen = any(c == CollisionType.AGENT.value for c in ray_col_type)
        dist = self.world.euclidean_dist(agent_pos, opp_pos)

        heard = dist <= self.HEARING_DIST
        return obs, int(seen), int(heard)

    def _get_reward(
        self, state: PEState, next_state: PEState, evader_seen: bool
    ) -> Dict[M.AgentID, float]:
        evader_pos = array_to_position(next_state.evader_state)
        pursuer_pos = array_to_position(next_state.pursuer_state)
        evader_goal_coord = next_state.evader_goal_coord

        evader_reward = 0.0
        if self._use_progress_reward and next_state.min_goal_dist < state.min_goal_dist:
            evader_reward += self.R_PROGRESS

        if self.world.agents_collide(evader_pos, pursuer_pos) or evader_seen:
            evader_reward += self.R_CAPTURE
        elif self.world.agents_collide(evader_pos, evader_goal_coord):
            evader_reward += self.R_EVASION

        if self._normalize_reward:
            evader_reward = self._get_normalized_reward(evader_reward)

        return {
            str(self.EVADER_IDX): evader_reward,
            str(self.PURSUER_IDX): -evader_reward,
        }

    def _is_done(self, state: PEState, evader_seen: bool) -> bool:
        if evader_seen:
            return True

        evader_pos = array_to_position(state.evader_state)
        pursuer_pos = array_to_position(state.pursuer_state)
        return self.world.agents_collide(
            evader_pos, pursuer_pos
        ) or self.world.agents_collide(evader_pos, state.evader_goal_coord)

    def _get_outcome(
        self, state: PEState, evader_seen: bool
    ) -> Dict[M.AgentID, M.Outcome]:
        # Assuming this method is called on final timestep
        evader_pos = array_to_position(state.evader_state)
        pursuer_pos = array_to_position(state.pursuer_state)
        evader_goal_coord = state.evader_goal_coord
        evader_id, pursuer_id = str(self.EVADER_IDX), str(self.PURSUER_IDX)

        if evader_seen or self.world.agents_collide(evader_pos, pursuer_pos):
            return {evader_id: M.Outcome.LOSS, pursuer_id: M.Outcome.WIN}
        if self.world.agents_collide(evader_pos, evader_goal_coord):
            return {evader_id: M.Outcome.WIN, pursuer_id: M.Outcome.LOSS}
        return {evader_id: M.Outcome.DRAW, pursuer_id: M.Outcome.DRAW}

    def _get_normalized_reward(self, reward: float) -> float:
        """Normalize reward in [-1, 1] interval."""
        diff = self._max_raw_return - self._min_raw_return
        return 2 * (reward - self._min_raw_return) / diff - 1

    def relative_positioning(
        self, agent_i: Position, agent_j: Position
    ) -> Tuple[float, float]:
        dist = self.world.euclidean_dist(agent_i, agent_j)
        yaw = agent_i[2]

        # Rotation matrix of yaw
        R = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])

        T_p = np.array([agent_i[0] - agent_j[0], agent_i[1] - agent_j[1]])
        T_p = R.dot(T_p)
        alpha = math.atan2(T_p[1], T_p[0]) + yaw

        return (alpha, dist)


class PEWorld(SquareContinuousWorld):
    """A world for the Pursuit Evasion Problem."""

    def __init__(
        self,
        size: int,
        blocks: Optional[List[CircleEntity]],
        interior_walls: List[Line],
        goal_coords_map: Dict[FloatCoord, List[FloatCoord]],
        evader_start_coords: List[FloatCoord],
        pursuer_start_coords: List[FloatCoord],
    ):
        super().__init__(
            size=size,
            blocks=blocks,
            interior_walls=interior_walls,
            agent_radius=0.4,
            border_thickness=0.01,
            enable_agent_collisions=True,
        )

        self._goal_coords_map = goal_coords_map
        self.evader_start_coords = evader_start_coords
        self.pursuer_start_coords = pursuer_start_coords
        self.shortest_paths = self.get_all_shortest_paths(evader_start_coords)

    @property
    def all_goal_coords(self) -> List[FloatCoord]:
        """The list of all evader goal locations."""
        all_locs = set()
        for v in self._goal_coords_map.values():
            all_locs.update(v)
        return list(all_locs)

    def get_goal_coords(self, evader_start_coord: FloatCoord) -> List[FloatCoord]:
        """Get list of possible evader goal coords for given start coords."""
        return self._goal_coords_map[evader_start_coord]

    def get_shortest_path_distance(self, coord: FloatCoord, dest: FloatCoord) -> int:
        """Get the shortest path distance from coord to destination."""
        coord_c = self.convert_to_coord(coord)
        dest_c = self.convert_to_coord(dest)
        return int(self.shortest_paths[dest_c][coord_c])

    def get_max_shortest_path_distance(self) -> int:
        """Get max shortest path distance between any start and goal coords."""
        max_dist = 0
        for start_coord, dest_coords in self._goal_coords_map.items():
            sp_dist = max(
                [
                    self.get_shortest_path_distance(start_coord, dest)
                    for dest in dest_coords
                ]
            )
            max_dist = max(max_dist, int(sp_dist))
        return max_dist


def get_8x8_world() -> PEWorld:
    """Generate the 8-by-8 PE world layout.

    This is an approximate discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    - 0, 1, 2, 7, 8, 9 are possible evader start and goal locations
    - 5, 6 are possible pursuer start locations

    The evader start and goal locations are always on opposite sides of the
    map.
    """
    ascii_map = (
        "  9 #8 #\n"
        "# #    #\n"
        "# ## # 7\n"
        "  6   # \n"
        "# #5# # \n"
        "   #    \n"
        "#    #2 \n"
        "0 #1### \n"
    )

    return _convert_map_to_world(ascii_map, 8, 8)


def get_16x16_world() -> PEWorld:
    """Generate the 16-by-16 PE world layout.

    This is an approximate discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    - 0, 1, 2, 7, 8, 9 are possible evader start and goal locations
    - 5, 6 are possible pursuer start locations

    The evader start and goal locations are always on opposite sides of the
    map.

    """
    ascii_map = (
        "  ## ####### ###\n"
        "    9##    8 ###\n"
        "##       # # ###\n"
        "## ## ##      ##\n"
        "## ## ##   ## ##\n"
        "#  #  ##   ## 7#\n"
        "# ## ###   #   #\n"
        "  #  6       # #\n"
        "#   ## ##  ##  #\n"
        "##   #5##  #   #\n"
        "## #   #     # #\n"
        "   # #   ##    #\n"
        " # # ##  ##   ##\n"
        "0#       #     #\n"
        "   ### #   ##2  \n"
        "###    1 #####  \n"
    )
    return _convert_map_to_world(ascii_map, 16, 16)


def get_32x32_world() -> PEWorld:
    """Generate the 32-by-32 PE world layout.

    This is an approximate discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    - 0, 1, 2, 7, 8, 9 are possible evader start and goal locations
    - 5, 6 are possible pursuer start locations

    The evader start and goal locations are always on opposite sides of the
    map.

    """
    ascii_map = (
        "       #  ########### #         \n"
        "   #####  ########### #  #######\n"
        "      #   ######      8  #######\n"
        " #       9#               ######\n"
        " ##              #### ##  ######\n"
        " ##   #####  ##  #### ##   #####\n"
        " ##   #####  ##  ##        #####\n"
        "      ##### ####      ###  #####\n"
        " ### ###### ####      ###   ####\n"
        " ### #####  ####      ####  ####\n"
        " ##  ##### #####      ##### 7###\n"
        " ##  ####  #####       ####  ###\n"
        " #  #####  #####       ####  ###\n"
        " #  ##### ######       #     ###\n"
        " #       6               ##  ###\n"
        "    #       #   ##      ####  ##\n"
        "     #   ####  ###     ####   ##\n"
        "#### #   ####  ###    ####    ##\n"
        "#### #   ### 5####   ####     ##\n"
        "####  #      ####     ##  #   ##\n"
        "##### #      ####        ##   ##\n"
        "#                          ##  #\n"
        "         ###        ##    2 #  #\n"
        "  ###### ###      #### ## #    #\n"
        "########  ##      ####    #  #  \n"
        "########  ##       ####     ####\n"
        "###          ##   1##        ###\n"
        "          #  ####       #       \n"
        "   0  ###### ######   ####      \n"
        "  ########## #        #####     \n"
        "##########      ############    \n"
        "#                               \n"
    )
    return _convert_map_to_world(ascii_map, 32, 32)


def _loc_to_coord(loc: int, world_width: int) -> Position:
    return (loc % world_width + 0.5, loc // world_width + 0.5, 0)


def _convert_map_to_world(
    ascii_map: str,
    height: int,
    width: int,
    block_symbol: str = "#",
    pursuer_start_symbols: Optional[Set[str]] = None,
    evader_start_symbols: Optional[Set[str]] = None,
    evader_goal_symbol_map: Optional[Dict] = None,
) -> PEWorld:
    row_strs = ascii_map.splitlines()
    assert len(row_strs) > 1
    assert all(len(row) == len(row_strs[0]) for row in row_strs)
    assert len(row_strs[0]) > 1

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

    interior_walls = parse_world_str_interior_walls(ascii_map)

    evader_start_coords = []
    pursuer_start_coords = []
    evader_symbol_coord_map = {}
    for r, c in product(range(height), range(width)):
        coord = (c + 0.5, r + 0.5)
        symbol = row_strs[r][c]
        # if symbol == block_symbol:
        # The radius of the block is slightly smaller, otherwise
        # agents get stuck
        # block_coords.append(((coord[0], coord[1], 0), 0.4))
        if symbol in pursuer_start_symbols:
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
        size=width,
        blocks=None,
        interior_walls=interior_walls,
        goal_coords_map=evader_goal_coords_map,
        evader_start_coords=evader_start_coords,
        pursuer_start_coords=pursuer_start_coords,
    )


# world_name: (world_make_fn, step_limit)
SUPPORTED_WORLDS: Dict[str, Tuple[Callable[[], PEWorld], int]] = {
    "8x8": (get_8x8_world, 50),
    "16x16": (get_16x16_world, 100),
    "32x32": (get_32x32_world, 200),
}
