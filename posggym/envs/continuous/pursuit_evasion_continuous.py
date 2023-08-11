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
from gymnasium import spaces

import posggym.model as M
from posggym import logger
from posggym.core import DefaultEnv
from posggym.envs.continuous.core import (
    CollisionType,
    Coord,
    FloatCoord,
    PMBodyState,
    SquareContinuousWorld,
    clip_actions,
    generate_interior_walls,
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
PEObs = np.ndarray


class PursuitEvasionContinuousEnv(DefaultEnv):
    """The Pursuit-Evasion Continuous World Environment.

    An adversarial continuous world problem involving two agents: an evader and a
    pursuer. The evader's goal is to reach a goal location, on the other side of
    the world, while the goal of the pursuer is to spot the evader before it reaches
    it's goal. The evader is considered caught if it is observed by the pursuer, or
    occupies the same location. The evader and pursuer have knowledge of each others
    starting locations, however only the evader has knowledge of it's goal location.
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

    0. the state of the evader
    1. the state of the pursuer
    2. the `(x, y)` coordinate of the evader's start location
    3. the `(x, y)` coordinate of the pursuer's start location
    4. the `(x, y)` coordinate of the evader's goal
    5. the minimum distance to it's goal along the shortest discrete path achieved
       by the evader in the current episode (this is needed to correctly reward
       the agent for making progress.)

    Both the evader and pursuer state consist of their:

    - `(x, y)` coordinate
    - angle/yaw (in radians)
    - velocity in x and y directions
    - angular velocity (in radians)

    Action Space
    ------------
    Each agent's actions is made up of two parts. The first action component specifies
    the angular velocity in `[-pi/4, pi/4]`, and the second component specifies the
    linear velocity in `[0, 1]`.

    Observation Space
    -----------------
    Each agent observes the local world in a `fov` radian cone in front of
    themselves. This is achieved by a series of `n_sensors` sensor lines starting
    at the agent which extend up to `max_obs_distance` away from the agent.
    For each sensor line the agent observes the closest entity along the line,
    specifically if there is a wall or the other agent. Along with the sensor
    readings each agent also observes whether they hear the other agent within
    a circle with radius ``HEARING_DIST = 4.0` around themselves, the `(x, y)`
    coordinate of the evader's start location and the `(x, y)` coordinate of the
    pursuer's start location. The evader also observes  the `(x, y)` coordinate of
    their goal location, while the pursuer receives a value of `(0, 0` for
    this feature.

    This table enumerates the first part of the observation space:

    | Index: start          | Description                          |  Values   |
    | :-------------------: | :----------------------------------: | :-------: |
    | 0                     | Wall distance                        | [0, d]    |
    | n_sensors             | Other agent distance                 | [0, d]    |
    | 2 * n_sensors         | Other agent heard                    | [0, 1]    |
    | 2 * n_sensors + 1     | Evader start coordinates             | [0, s]    |
    | 2 * n_sensors + 3     | Pursuer start coordinates            | [0, s]    |
    | 2 * n_sensors + 5     | Evader goal coordinates              | [0, s]    |

    Where `d = max_obs_distance` and `s = world.size`

    If an entity is not observed by a sensor (i.e. it's not within `max_obs_distance`
    or is not the closest entity to the observing agent along the line), The distance
    reading will be `max_obs_distance`.

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
    Actions are deterministic and will change the agents direction and velocity in the
    direction they are facing. The velocity range of an agent is [0, 1]; agent's cannot
    move backwards (have negative velocity).

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
    An episode ends when either the evader is seen or touched by the pursuer or the
    evader reaches it's goal. By default a `max_episode_steps` limit of `200` steps is
    also set. This may need to be adjusted when using larger worlds (this can be done
    by manually specifying a value for `max_episode_steps` when creating the environment
    with `posggym.make`).

    Arguments
    ---------

    - `world` - the world layout to use. This can either be a string specifying one of
         the supported worlds (see SUPPORTED_WORLDS), or a custom :class:`PEWorld`
         object (default = `"16x16"`).
    - `max_obs_distance` - the maximum distance from the agent that each agent's field
        of vision extends. If `None` then sets this to 1/3 of world size
        (default = `None`).
    - `fov` - the field of view of the agent in radians. This will determine the
        angle of the cone in front of the agent which it can see. The FOV will be
        relative to the angle of the agent (default = `pi / 3`).
    - `n_sensors` - the number of sensor lines eminating from the agent within their
        FOV. The agent will observe at `n_sensors` equidistance intervals over
        `[-fov / 2, fov / 2]` (default = `16`).
    - `normalize_reward` - whether to normalize both agents' rewards to be between `-1`
        and `1` (default = 'True`)
    - `use_progress_reward` - whether to reward the evader agent for making progress
        towards it's goal. If False the evader will only be rewarded when it reaches
        it's goal, making it a sparse reward problem (default = 'True`).

    Available variants
    ------------------

    The PursuitEvasionContinuous environment comes with a number of pre-built world
    layouts which can be passed as an argument to `posggym.make`, to create different
    worlds.

    | World name       | World size |
    |------------------|------------|
    | `8x8`            |  8x8       |
    | `16x16`          |  16x16     |
    | `32x32`          |  32x32     |

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
        max_obs_distance: Optional[float] = None,
        fov: float = np.pi / 3,
        n_sensors: int = 16,
        normalize_reward: bool = True,
        use_progress_reward: bool = True,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        model = PursuitEvasionContinuousModel(
            world,
            max_obs_distance=max_obs_distance,
            normalize_reward=normalize_reward,
            use_progress_reward=use_progress_reward,
            fov=fov,
            n_sensors=n_sensors,
            **kwargs,
        )
        super().__init__(model, render_mode=render_mode)
        self.window_surface = None
        self.blocked_surface = None
        self.clock = None
        self.window_size = 600
        self.draw_options = None
        self.world = None
        self.fov = fov

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
        return self._render_img()

    def _render_img(self):
        import pygame
        from pymunk import Transform, pygame_util

        model = cast(PursuitEvasionContinuousModel, self.model)
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
            self.draw_options.transform = Transform.scaling(scale_factor)
            # don't render collision lines
            self.draw_options.flags = (
                pygame_util.DrawOptions.DRAW_SHAPES
                | pygame_util.DrawOptions.DRAW_CONSTRAINTS
            )

        if self.world is None:
            # get copy of model world, so we can use it for rendering without
            # affecting the original
            self.world = model.world.copy()

        if self.blocked_surface is None:
            self.blocked_surface = pygame.Surface((self.window_size, self.window_size))
            self.blocked_surface.fill(pygame.Color("white"))
            cell_size = scale_factor
            for coord in self.world.blocked_coords:
                rect = (
                    round(coord[0] * cell_size),
                    round(coord[1] * cell_size),
                    round(cell_size),
                    round(cell_size),
                )
                self.blocked_surface.fill(self.world.BLOCK_COLOR, rect=rect)

        self.world.set_entity_state("pursuer", state.pursuer_state)
        self.world.set_entity_state("evader", state.evader_state)

        # Need to do this for space to update with changes
        self.world.space.step(0.0001)

        # reset screen
        self.window_surface.fill(pygame.Color("white"))

        # add blocks
        self.window_surface.blit(self.blocked_surface, (0, 0))

        # draw sensor lines
        n_sensors = model.n_sensors
        for idx, p_state in [
            (model.EVADER_IDX, state.evader_state),
            (model.PURSUER_IDX, state.pursuer_state),
        ]:
            obs_i = self._last_obs[str(idx)]
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
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


class PursuitEvasionContinuousModel(M.POSGModel[PEState, PEObs, PEAction]):
    """Continuous Pursuit-Evasion Model.

    Arguments
    ---------
    world : str, PEWorld
        the world layout to use. This can either be a string specifying one of
        the supported worlds (see SUPPORTED_WORLDS), or a custom :class:`PEWorld`
        object.
    max_obs_distance : float, optional
        the maximum distance from the agent that each agent's field of vision extends.
        If `None` the distance will be set to 1/3 of the world's size.
    fov : float
        the field of view of the agent in radians. This will determine the angle of the
        cone in front of the agent which it can see. The FOV will be relative to the
        angle of the agent.
    n_sensors : int
        the number of sensor lines eminating from the agent within their FOV. The agent
        will observe at `n_sensors` equidistance intervals over `[-fov / 2, fov / 2]`.
    normalize_reward : bool
        whether to normalize both agents' rewards to be between `-1` and `1`
    use_progress_reward : bool
        whether to reward the evader agent for making progress towards it's goal. If
        False the evader will only be rewarded when it reaches it's goal, making it a
        sparse reward problem.

    """

    NUM_AGENTS = 2

    EVADER_IDX = 0
    PURSUER_IDX = 1

    # Evader-centric rewards
    R_PROGRESS = 0.01  # Reward making progress toward goal
    R_EVASION = 1.0  # Reward for reaching goal (capture reward is -R_EVASION)

    HEARING_DIST = 4

    EVADER_COLOR = (55, 155, 205, 255)  # Blueish
    PURSUER_COLOR = (255, 55, 55, 255)  # Purpleish
    GOAL_COLOR = (55, 255, 55, 255)  # Greenish

    def __init__(
        self,
        world: Union[str, "PEWorld"],
        max_obs_distance: Optional[float] = None,
        fov: float = np.pi / 3,
        n_sensors: int = 16,
        normalize_reward: bool = True,
        use_progress_reward: bool = True,
    ):
        assert 0 < fov < 2 * np.pi, "fov must be in (0, 2 * pi)"
        assert n_sensors > 0, "n_sensors must be positive"

        if isinstance(world, str):
            assert world in SUPPORTED_WORLDS, (
                f"Unsupported world name '{world}'. If world is a string it must be "
                f"one of: {SUPPORTED_WORLDS.keys()}."
            )
            world = SUPPORTED_WORLDS[world]()

        self.world = world

        if max_obs_distance is None:
            max_obs_distance = self.world.size / 3
        assert max_obs_distance > 0, "max_obs_distance must be positive"
        self.max_obs_distance = max_obs_distance
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
        # e_state, p_state, e_start_coord, p_start_coord, e_goal_coord, min_sp
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

        # can turn by up to 45 degrees per timestep
        self.dyaw_limit = math.pi / 4
        self.action_spaces = {
            i: spaces.Box(
                low=np.array([-self.dyaw_limit, 0.0], dtype=np.float32),
                high=np.array([self.dyaw_limit, 1.0], dtype=np.float32),
            )
            for i in self.possible_agents
        }

        self.obs_dist = self.max_obs_distance
        self.n_sensors = n_sensors
        self.sensor_obs_dim = self.n_sensors * 2
        self.obs_dim = self.sensor_obs_dim + 7
        sensor_low = [0.0] * self.sensor_obs_dim
        sensor_high = [self.max_obs_distance] * self.sensor_obs_dim
        size = self.world.size
        self.observation_spaces = {
            i: spaces.Box(
                low=np.array([*sensor_low, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array(
                    [*sensor_high, 1, size, size, size, size, size, size],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )
            for i in self.possible_agents
        }
        self.is_symmetric = False

        # Add physical entities to the world
        self.world.add_entity("pursuer", None, color=self.PURSUER_COLOR)
        self.world.add_entity("evader", None, color=self.EVADER_COLOR)

    @property
    def reward_ranges(self) -> Dict[str, Tuple[float, float]]:
        max_reward = self.R_EVASION
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

    def get_agents(self, state: PEState) -> List[str]:
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

    def sample_initial_obs(self, state: PEState) -> Dict[str, PEObs]:
        return self._get_obs(state)[0]

    def step(
        self, state: PEState, actions: Dict[str, PEAction]
    ) -> M.JointTimestep[PEState, PEObs]:
        clipped_actions = clip_actions(actions, self.action_spaces)
        next_state = self._get_next_state(state, clipped_actions)
        obs, evader_seen = self._get_obs(next_state)

        rewards = self._get_reward(state, next_state, evader_seen)
        all_done = self._is_done(next_state, evader_seen)
        terminated = {i: all_done for i in self.possible_agents}
        truncated = {i: False for i in self.possible_agents}
        info: Dict[str, Dict] = {i: {} for i in self.possible_agents}
        if all_done:
            for i, outcome in self._get_outcome(next_state, evader_seen).items():
                info[i]["outcome"] = outcome

        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, info
        )

    def _get_next_state(self, state: PEState, actions: Dict[str, PEAction]) -> PEState:
        evader_a = actions[str(self.EVADER_IDX)]
        pursuer_a = actions[str(self.PURSUER_IDX)]

        self.world.set_entity_state("pursuer", state.pursuer_state)
        self.world.set_entity_state("evader", state.evader_state)

        pursuer_angle = state.pursuer_state[2] + pursuer_a[0]
        pursuer_vel = self.world.linear_to_xy_velocity(pursuer_a[1], pursuer_angle)
        self.world.update_entity_state("pursuer", angle=pursuer_angle, vel=pursuer_vel)

        evader_angle = state.evader_state[2] + evader_a[0]
        evader_vel = self.world.linear_to_xy_velocity(evader_a[1], evader_angle)
        self.world.update_entity_state("evader", angle=evader_angle, vel=evader_vel)

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

    def _get_obs(self, state: PEState) -> Tuple[Dict[str, PEObs], bool]:
        evader_obs, _ = self._get_agent_obs(state, evader=True)
        pursuer_obs, evader_seen = self._get_agent_obs(state, evader=False)

        return {
            str(self.EVADER_IDX): evader_obs,
            str(self.PURSUER_IDX): pursuer_obs,
        }, evader_seen

    def _get_agent_obs(
        self,
        state: PEState,
        evader: bool,
    ) -> Tuple[np.ndarray, bool]:
        if evader:
            agent_pos = (
                state.evader_state[0],
                state.evader_state[1],
                state.evader_state[2],
            )
            opp_coord = (state.pursuer_state[0], state.pursuer_state[1])
        else:
            agent_pos = (
                state.pursuer_state[0],
                state.pursuer_state[1],
                state.pursuer_state[2],
            )
            opp_coord = (state.evader_state[0], state.evader_state[1])

        ray_dists, ray_col_type = self.world.check_collision_circular_rays(
            agent_pos,
            self.max_obs_distance,
            self.n_sensors,
            np.array([opp_coord]),
            include_blocks=True,
            check_walls=True,
            use_relative_angle=True,
            angle_bounds=(-self.fov / 2, self.fov / 2),
        )

        obs = np.full((self.obs_dim,), self.max_obs_distance, dtype=np.float32)
        # Can bucket NONE type colisions with block/border/wall collisions, since these
        # will have dist of obs_dist and hence not affect final obs
        obs_type_idx = np.where(ray_col_type == CollisionType.AGENT.value, 1, 0)
        flat_obs_idx = np.ravel_multi_index(
            (obs_type_idx, np.arange(self.n_sensors)), dims=(2, self.n_sensors)
        )
        obs[flat_obs_idx] = ray_dists

        other_agent_seen = any(c == CollisionType.AGENT.value for c in ray_col_type)
        dist = self.world.euclidean_dist(agent_pos, opp_coord)

        aux_obs_idx = self.sensor_obs_dim
        obs[aux_obs_idx] = dist <= self.HEARING_DIST
        obs[aux_obs_idx + 1 : aux_obs_idx + 3] = state.evader_start_coord
        obs[aux_obs_idx + 3 : aux_obs_idx + 5] = state.pursuer_start_coord

        if evader:
            obs[aux_obs_idx + 5 : aux_obs_idx + 7] = state.evader_goal_coord
        else:
            obs[aux_obs_idx + 5 : aux_obs_idx + 7] = [0, 0]

        return obs, other_agent_seen

    def _get_reward(
        self, state: PEState, next_state: PEState, evader_seen: bool
    ) -> Dict[str, float]:
        evader_reward = 0.0
        if self._use_progress_reward and next_state.min_goal_dist < state.min_goal_dist:
            evader_reward += self.R_PROGRESS

        if evader_seen or self.world.agents_collide(
            next_state.evader_state, next_state.pursuer_state
        ):
            evader_reward -= self.R_EVASION
        elif self.world.agents_collide(
            next_state.evader_state, next_state.evader_goal_coord
        ):
            evader_reward += self.R_EVASION

        if self._normalize_reward:
            evader_reward = self._get_normalized_reward(evader_reward)

        return {
            str(self.EVADER_IDX): evader_reward,
            str(self.PURSUER_IDX): -evader_reward,
        }

    def _is_done(self, state: PEState, evader_seen: bool) -> bool:
        # need to do explicit bool conversion as conditions can return np.bool
        return bool(
            evader_seen
            or self.world.agents_collide(state.evader_state, state.pursuer_state)
            or self.world.agents_collide(state.evader_state, state.evader_goal_coord)
        )

    def _get_outcome(self, state: PEState, evader_seen: bool) -> Dict[str, M.Outcome]:
        evader_id, pursuer_id = str(self.EVADER_IDX), str(self.PURSUER_IDX)
        if evader_seen or self.world.agents_collide(
            state.evader_state, state.pursuer_state
        ):
            return {evader_id: M.Outcome.LOSS, pursuer_id: M.Outcome.WIN}
        if self.world.agents_collide(state.evader_state, state.evader_goal_coord):
            return {evader_id: M.Outcome.WIN, pursuer_id: M.Outcome.LOSS}
        return {evader_id: M.Outcome.DRAW, pursuer_id: M.Outcome.DRAW}

    def _get_normalized_reward(self, reward: float) -> float:
        """Normalize reward in [-1, 1] interval."""
        diff = self._max_raw_return - self._min_raw_return
        return 2 * (reward - self._min_raw_return) / diff - 1


class PEWorld(SquareContinuousWorld):
    """A world for the Pursuit Evasion Problem.

    Arguments
    ---------
    size : int
        height and width of the world.
    blocked_coords : Set[Coord]
        coordinates of blocked cell, i.e. 1x1 areas that contain walls.
    goal_coords_map : Dict[FloatCoord, List[FloatCoord]]
        map from evader start coordinates to a list of possible evader goal coordinates.
    evader_start_coords : List[FloatCoord]
        list of possible evader start coordinates.
    pursuer_start_coords : List[FloatCoord]
        list of possible pursuer start coordinates.

    """

    def __init__(
        self,
        size: int,
        blocked_coords: Set[Coord],
        goal_coords_map: Dict[FloatCoord, List[FloatCoord]],
        evader_start_coords: List[FloatCoord],
        pursuer_start_coords: List[FloatCoord],
    ):
        interior_walls = generate_interior_walls(size, size, blocked_coords)
        super().__init__(
            size=size,
            blocks=None,
            interior_walls=interior_walls,
            agent_radius=0.4,
            border_thickness=0.01,
            enable_agent_collisions=True,
        )
        self._blocked_coords = blocked_coords
        self._goal_coords_map = goal_coords_map
        self.evader_start_coords = evader_start_coords
        self.pursuer_start_coords = pursuer_start_coords
        self.shortest_paths = self.get_all_shortest_paths(self.all_goal_coords)

    def copy(self) -> "PEWorld":
        world = PEWorld(
            size=int(self.size),
            blocked_coords=self.blocked_coords,
            goal_coords_map=self._goal_coords_map,
            evader_start_coords=self.evader_start_coords,
            pursuer_start_coords=self.pursuer_start_coords,
        )
        for id, (body, shape) in self.entities.items():
            # make copies of each entity, and ensure the copies are linked correctly
            # and added to the new world and world space
            body = body.copy()
            shape = shape.copy()
            shape.body = body
            world.space.add(body, shape)
            world.entities[id] = (body, shape)
        return world

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
    """Generate the 8-by-8 PE world layout."""
    ascii_map = (
        "# 9 # 8#\n"
        "  #    #\n"
        " ### # 7\n"
        "  6   # \n"
        " ##5# # \n"
        "   #    \n"
        "#    #2 \n"
        " 0#1### \n"
    )
    return convert_map_to_world(ascii_map, 8, 8)


def get_16x16_world() -> PEWorld:
    """Generate the 16-by-16 PE world layout."""
    ascii_map = (
        "  ## ####### ###\n"
        "    9##    8 ###\n"
        "##       ### ###\n"
        "## ## ##      ##\n"
        "## ## ##   ## ##\n"
        "#  #  ##   #  7#\n"
        "# ## ###   # # #\n"
        "  #  6       # #\n"
        "#   ## ##  ##  #\n"
        "  #  #5##  #   #\n"
        " ###   #     # #\n"
        "     #   ###   #\n"
        " # # ### ##   ##\n"
        "0# #     #     #\n"
        "   ## ##   ##2  \n"
        "##     1 #####  \n"
    )
    return convert_map_to_world(ascii_map, 16, 16)


def get_32x32_world() -> PEWorld:
    """Generate the 32-by-32 PE world layout."""
    ascii_map = (
        "          #############  #######\n"
        "   #####           ####  #######\n"
        "      ##  #######     8  #######\n"
        " ###     9#               ######\n"
        " ###             #### ##  ######\n"
        " ##   #####  ##  #### ##   #####\n"
        " ##   #####  ##  ##        #####\n"
        "      ##### ####      ###  #####\n"
        " ### ###### ####      ###   ####\n"
        " ### #####  ####      ####  ####\n"
        " ##  ##### #####      ##### 7###\n"
        " ##  ####  #####       ####  ###\n"
        " #  #####  #####       #     ###\n"
        " #  ##### ######       # ##  ###\n"
        " #       6             # ##  ###\n"
        "    #       #   ##       ###  ##\n"
        "     #   ####  ###     ####   ##\n"
        "#### #   ####  ###    ####    ##\n"
        "#### #   ### 5####   ####     ##\n"
        "####  #      ####     ##  #   ##\n"
        "##### #      ####        ##   ##\n"
        "#                        #  #  #\n"
        "        ####        ##      #  #\n"
        "  #####  ####     #### ## #    #\n"
        "#   ####  ###     ####    # 2#  \n"
        "### ####  ##       ####     ####\n"
        "###       #  ##   1##        ###\n"
        "      #      ####       #       \n"
        " 0    ###### ######   ####      \n"
        "  ### ###### #        #####     \n"
        "# ### ####      ############    \n"
        "#                               \n"
    )
    return convert_map_to_world(ascii_map, 32, 32)


def convert_map_to_world(
    ascii_map: str,
    height: int,
    width: int,
    block_symbol: str = "#",
    pursuer_start_symbols: Optional[Set[str]] = None,
    evader_start_symbols: Optional[Set[str]] = None,
    evader_goal_symbol_map: Optional[Dict] = None,
) -> PEWorld:
    """Generate PE world layout from ascii map.

    By default
    - 0, 1, 2, 7, 8, 9 are possible evader start and goal locations
    - 5, 6 are possible pursuer start locations
    - # are blocks

    The default maps are approximate discrete versions of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    """
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

    blocked_coords: Set[Coord] = set()
    evader_start_coords = []
    pursuer_start_coords = []
    evader_symbol_coord_map = {}
    for r, c in product(range(height), range(width)):
        coord = (c + 0.5, r + 0.5)
        symbol = row_strs[r][c]
        if symbol == block_symbol:
            blocked_coords.add((c, r))
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
        blocked_coords=blocked_coords,
        goal_coords_map=evader_goal_coords_map,
        evader_start_coords=evader_start_coords,
        pursuer_start_coords=pursuer_start_coords,
    )


# world_name: world_make_fn
SUPPORTED_WORLDS: Dict[str, Callable[[], PEWorld]] = {
    "8x8": get_8x8_world,
    "16x16": get_16x16_world,
    "32x32": get_32x32_world,
}
