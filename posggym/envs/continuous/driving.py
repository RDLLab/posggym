"""The Driving Grid World Environment."""
import enum
from itertools import product
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
import math
import pymunk
from pymunk import Vec2d
from gymnasium import spaces
import numpy as np

import posggym.model as M
from posggym.core import DefaultEnv

from posggym.utils import seeding
from posggym import logger

from posggym.envs.continuous.core2 import (
    Position,
    CircleEntity,
    PMBodyState,
    SquareContinuousWorld,
    clip_actions,
)


class CollisionType(enum.IntEnum):
    """Type of collision for a vehicle."""

    NONE = 0
    OBSTACLE = 1
    VEHICLE = 2


class VehicleState(NamedTuple):
    """The state of a vehicle in the Driving Environment."""

    coord: np.ndarray
    dest_coord: np.ndarray
    status: np.ndarray
    min_dest_dist: np.ndarray


DState = Tuple[VehicleState, ...]

# Initial direction and speed of each vehicle
INIT_DIR = 0
INIT_SPEED = 0
MAX_SPEED = 1

# Obs = (adj_obs, speed, dest_coord, dest_reached, crashed)
DObs = np.ndarray
# Tuple[Tuple[Union[int, np.ndarray], ...], np.ndarray, np.ndarray, int, int]

# This time it is acceleration
DAction = np.ndarray

# Cell obs
VEHICLE = 0
WALL = 1
EMPTY = 2
DESTINATION = 3


class DrivingEnv(DefaultEnv[DState, DObs, DAction]):
    """The Driving Grid World Environment.

    A general-sum 2D continuous world problem involving multiple agents. Each agent
    controls a vehicle and is tasked with driving the vehicle from it's start
    location to a destination location while avoiding crashing into obstacles
    or other vehicles.

    This environment requires each agent to navigate in the world while also
    taking care to avoid crashing into other agents. The dynamics and
    observations of the environment are such that avoiding collisions requires
    some planning in order for the vehicle to brake in time or maintain a good
    speed. Depending on the grid layout, the environment will require agents to
    reason about and possibly coordinate with the other vehicles.

    Possible Agents
    ---------------
    The environment supports two or more agents, depending on the grid. It is possible
    for some agents to finish the episode before other agents by either crashing or
    reaching their destination, and so not all agents are guaranteed to be active at
    the same time. All agents will be active at the start of the episode however.

    State Space
    -----------
    Each state is made up of the state of each vehicle, which in turn is defined by:

    - the `(x, y)` coordinates (x=column, y=row, with origin at the top-left square of
      the grid) of the vehicle,
    - the direction the vehicle is facing [0, 2π]
    - the speed of the vehicle: [-1, 1],
    - the `(x, y)` coordinate of the vehicles destination
    - whether the vehicle has reached it's destination or not: `1` or `0`
    - whether the vehicle has crashed or not: `1` or `0`
    - the minimum distance to the destination achieved by the vehicle in the current
      episode, if the environment was discrete.

    Action Space
    ------------
    Each agent has 2 actions, which are the angular velocity and linear acceleration.


    Observation Space
    -----------------
    Each agent observes the cells in their local area, as well as their current speed,
    their destination location, whether they've reached their destination, and whether
    they've crashed.
    The local area is observed by a series of 'n_lines' lines starting at the agent
    which extend for a distance of `obs_dim`. For each cell in the observed area the
    agent observes whether the cell contains a `VEHICLE=0`, `WALL=1`, `EMPTY=2`, or it's
    `DESTINATION=3`.

    All together each agent's observation is tuple of the form:

        ((local obs), speed, destination coord, destination reached, crashed)

    Rewards
    -------
    All agents receive a penalty of `0.0` for each step. They receive a penalty of
    `-1.0` for crashing (i.e. hitting another vehicle), and `-0.05` for moving into a
    wall. A reward of `1.0` is given if the agent reaches it's destination and a reward
    of `0.05` is given if the agent makes progress towards it's destination (i.e. it
    reduces it's minimum distance achieved to the destination for the episode).

    If `obstacle_collision=True` then running into a wall is treated as crashing the
    vehicle, and so results in a penalty of `-1.0`.

    Dynamics
    --------
    Actions are deterministic and movement is determined by direction the vehicle is
    facing and it's speed:

    Accelerating increases speed, while deceleration decreased speed. If the
    vehicle will hit a wall or another vehicle when moving from one cell to another then
    it remains in it's current cell and it's crashed state variable is updated
    appropriately.

    Starting State
    --------------
    Each agent is randomly assigned to one of the possible starting locations on the
    grid and one of the possible destination locations, with no two agents starting in
    the same location or having the same destination location. The possible start and
    destination locations are determined by the grid layout being used.

    Episodes End
    ------------
    Episodes end when all agents have either reached their destination or crashed. By
    default a `max_episode_steps` is also set for each Driving environment. The default
    value is `50` steps, but this may need to be adjusted when using larger grids (this
    can be done by manually specifying a value for `max_episode_steps` when creating the
    environment with `posggym.make`).

    Arguments
    ---------

    - `grid` - the grid layout to use. This can either be a string specifying one of
         the supported grids, or a custom :class:`DrivingGrid` object
         (default = `"14x14RoundAbout"`).
    - `num_agents` - the number of agents in the environment (default = `2`).
    - `obs_dim` - the local observation distance, specifying how the distance which an
         agent can observe  (default = `3`).
    - `n_lines` - the number of lines eminating from the agent. The agent will observe
         at `n` equidistance intervals over `[0, 2*pi]` (default = `10`).
    - `obstacle_collisions` -  whether running into a wall results in the agent's
         vehicle crashing and thus the agent reaching a terminal state. This can make
         the problem significantly harder (default = "False").

    Available variants
    ------------------

    The Driving environment comes with a number of pre-built grid layouts which can be
    passed as an argument to `posggym.make`, to create different grids:

    | Grid name         | Max number of agents | Grid size |
    |-------------------|----------------------|---------- |
    | `3x3`             | 2                    | 3x3       |
    | `6x6`             | 6                    | 6x6       |
    | `7x7Blocks`       | 4                    | 7x7       |
    | `7x7CrissCross`   | 6                    | 7x7       |
    | `7x7RoundAbout`   | 4                    | 7x7       |
    | `14x14Blocks`     | 4                    | 14x14     |
    | `14x14CrissCross` | 8                    | 14x14     |
    | `14x14RoundAbout` | 4                    | 14x14     |


    For example to use the DrivingContinuous environment with the `7x7RoundAbout` grid
    and 2 agents, you would use:

    ```python
    import posggym
    env = posggym.make('DrivingContinuous-v0', grid="7x7RoundAbout", num_agents=2)
    ```

    Version History
    ---------------
    - `v0`: Initial version

    References
    ----------
    - Adam Lerer and Alexander Peysakhovich. 2019. Learning Existing Social Conventions
    via Observationally Augmented Self-Play. In Proceedings of the 2019 AAAI/ACM
    Conference on AI, Ethics, and Society. 107–114.
    - Kevin R. McKee, Joel Z. Leibo, Charlie Beattie, and Richard Everett. 2022.
    Quantifying the Effects of Environment and Population Diversity in Multi-Agent
    Reinforcement Learning. Autonomous Agents and Multi-Agent Systems 36, 1 (2022), 1–16

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }

    def __init__(
        self,
        grid: Union[str, "DrivingGrid"] = "7x7RoundAbout",
        num_agents: int = 2,
        obs_dim: float = 3.0,
        n_lines: int = 10,
        obstacle_collisions: bool = False,
        render_mode: Optional[str] = None,
    ):
        super().__init__(
            DrivingModel(grid, num_agents, obs_dim, n_lines, obstacle_collisions),
            render_mode=render_mode,
        )
        self._obs_dist = obs_dim
        self._renderer = None
        self._agent_imgs = None
        self.window_surface = None
        self.clock = None
        self.window_size = 600
        self.draw_options = None
        self.world = None

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
        # import posggym.envs.continuous.render as render_lib
        import pygame
        from pymunk import pygame_util

        model = cast(DrivingModel, self.model)
        state = cast(DState, self.state)
        scale_factor = self.window_size // model.world.size

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

        for i, p_state in enumerate(state):
            self.world.set_entity_state(f"vehicle_{i}", p_state.coord)

        # Need to do this for space to update with changes
        self.world.space.step(0.0001)

        # reset screen
        self.window_surface.fill(pygame.Color("white"))

        lines_colors = ["red", "green", "blue", "black"]
        # draw sensor lines
        n_sensors = model.n_sensors
        for i, obs_i in self._last_obs.items():
            p_state = state[int(i)].coord
            x, y, agent_angle = p_state[:3]
            angle_inc = 2 * math.pi / n_sensors
            for k in range(n_sensors):
                values = [obs_i[k], obs_i[n_sensors + k], obs_i[2 * n_sensors + k]]
                dist_idx = min(range(len(values)), key=values.__getitem__)
                dist = values[dist_idx]

                if all(x == self._obs_dist for x in values):
                    dist_idx = 3

                # import pdb; pdb.set_trace()

                angle = angle_inc * k + agent_angle
                end_x = x + dist * math.cos(angle)
                end_y = y + dist * math.sin(angle)
                scaled_start = (int(x * scale_factor), int(y * scale_factor))
                scaled_end = int(end_x * scale_factor), (end_y * scale_factor)

                pygame.draw.line(
                    self.window_surface,
                    pygame.Color(lines_colors[dist_idx]),
                    scaled_start,
                    scaled_end,
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
        if self._renderer is not None:
            # self._renderer.close()
            self._renderer = None


class DrivingModel(M.POSGModel[DState, DObs, DAction]):
    """Driving Problem Model.

    Parameters
    ----------
    grid : DrivingGrid
        the grid environment for the model scenario
    num_agents : int
        the number of agents in the model scenario
    obs_dims : float
        number of cells in front, behind, and to the side that each agent
        can observe
    obstacle_collisions : bool
        whether cars can crash into wall and other obstacles, on top of
        crashing into other vehicles

    """

    R_STEP_COST = 0.00
    R_CRASH_OBJECT = -0.05
    R_CRASH_VEHICLE = -1.0
    R_DESTINATION_REACHED = 1.0
    R_PROGRESS = 0.05
    PREDATOR_COLOR = (55, 155, 205, 255)  # Blueish
    GOAL_COLOR = (0, 100, 155, 255)  # Blueish
    COLLISION_DIST = 1.2

    def __init__(
        self,
        grid: Union[str, "DrivingGrid"],
        num_agents: int,
        obs_dim: float,
        n_lines: int,
        obstacle_collisions: bool,
    ):
        if isinstance(grid, str):
            assert grid in SUPPORTEDworldS, (
                f"Unsupported grid '{grid}'. If grid argument is a string it must be "
                f"one of: {SUPPORTEDworldS.keys()}."
            )
            grid_info = SUPPORTEDworldS[grid]
            supported_num_agents: int = grid_info["supported_num_agents"]
            assert 0 < num_agents <= supported_num_agents, (
                f"Driving grid `{grid}` does not support {num_agents} agents. The "
                f"supported number of agents is from 1 up to {supported_num_agents}."
            )
            grid = parseworld_str(
                grid_info["grid_str"], grid_info["supported_num_agents"]
            )
        else:
            assert 0 < num_agents <= grid.supported_num_agents, (
                f"Supplied DrivingGrid `{grid}` does not support {num_agents} agents. "
                "The supported number of agents is from 1 up to "
                f"{grid.supported_num_agents}."
            )

        self.world = grid
        self._obstacle_collisions = obstacle_collisions

        def _pos_space():
            # x, y, angle, vx, vy, vangle
            # shape = (1, 6)
            size, angle = self.world.size, 2 * math.pi
            low = np.array([-1, -1, -angle, -1, -1, -angle], dtype=np.float32)
            high = np.array(
                [size, size, angle, 1.0, 1.0, angle],
                dtype=np.float32,
            )
            return spaces.Box(low=low, high=high)

        def _coord_space():
            # x, y, unused
            # shape = (1, 3)
            size, angle = self.world.size, 2 * math.pi
            low = np.array([-1, -1, -angle], dtype=np.float32)
            high = np.array(
                [size, size, angle],
                dtype=np.float32,
            )
            return spaces.Box(low=low, high=high)

        self.possible_agents = tuple(str(i) for i in range(num_agents))
        self.state_space = spaces.Tuple(
            tuple(
                spaces.Tuple(
                    (
                        # State of each vehicle
                        _pos_space(),
                        _coord_space(),  # destination coord
                        # dest reached, crashed
                        spaces.MultiBinary(2),
                        # min distance to destination
                        # set this to upper bound of min shortest path distance, so
                        # state space works for generated grids as well
                        spaces.Box(
                            low=np.array([0], dtype=np.float32),
                            high=np.array([self.world.size**2], dtype=np.float32),
                        ),
                    )
                )
                for _ in range(len(self.possible_agents))
            )
        )

        self.dyaw_limit = math.pi / 10
        # dyaw, vel
        self.action_spaces = {
            i: spaces.Box(
                low=np.array([-self.dyaw_limit, 0.0], dtype=np.float32),
                high=np.array([self.dyaw_limit, 1.0], dtype=np.float32),
            )
            for i in self.possible_agents
        }

        self.n_lines = n_lines
        self.obs_distance = obs_dim

        # Observes entity and distance to entity along a n_sensors rays from the agent
        # 0 to n_sensors = wall distance obs
        # n_sensors to (2 * n_sensors) = other vehicle dist
        # (2 * n_sensors) to (3 * n_sensors) = destination coordinate
        self.n_sensors = 10
        self.obs_dist = 2
        self.obs_dim = self.n_sensors * 3
        self.observation_spaces = {
            i: spaces.Box(
                low=0.0, high=self.obs_dist, shape=(self.obs_dim,), dtype=np.float32
            )
            for i in self.possible_agents
        }

        # Add physical entities to the world
        for i in range(len(self.possible_agents)):
            self.world.add_entity(f"vehicle_{i}", None, color=self.PREDATOR_COLOR)

        self.is_symmetric = True

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[float, float]]:
        return {
            i: (self.R_CRASH_VEHICLE, self.R_DESTINATION_REACHED)
            for i in self.possible_agents
        }

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: DState) -> List[M.AgentID]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> DState:
        state = []
        chosen_start_coords: Set[Position] = set()
        chosen_dest_coords: Set[Position] = set()
        for i in range(len(self.possible_agents)):
            start_coords_i = self.world.start_coords[i]
            avail_start_coords = start_coords_i.difference(chosen_start_coords)
            start_coord = self.rng.choice(list(avail_start_coords))
            chosen_start_coords.add(start_coord)

            dest_coords_i = self.world.dest_coords[i]
            avail_dest_coords = dest_coords_i.difference(chosen_dest_coords)
            if start_coord in avail_dest_coords:
                avail_dest_coords.remove(start_coord)

            _dest_coord = self.rng.choice(list(avail_dest_coords))
            chosen_dest_coords.add(_dest_coord)

            vehicle_state = np.zeros((PMBodyState.num_features()), dtype=np.float32)
            vehicle_state[:3] = start_coord

            dest_coord = np.zeros(3, dtype=np.float32)
            dest_coord[:] = _dest_coord

            dest_dist = self.world.get_shortest_path_distance(start_coord, _dest_coord)

            state_i = VehicleState(
                coord=vehicle_state,
                dest_coord=dest_coord,
                status=np.array([int(False), int(False)], dtype=np.int8),
                min_dest_dist=np.array([dest_dist], dtype=np.float32),
            )

            self.world.add_entity(
                f"goal_{i}", self.world.agent_radius, self.GOAL_COLOR, is_static=True
            )
            self.world.update_entity_state(
                f"goal_{i}", coord=(dest_coord[0], dest_coord[1])
            )

            state.append(state_i)

        return tuple(state)

    def sample_initial_obs(self, state: DState) -> Dict[M.AgentID, DObs]:
        return self._get_obs(state)

    def step(
        self, state: DState, actions: Dict[M.AgentID, DAction]
    ) -> M.JointTimestep[DState, DObs]:
        clipped_actions = clip_actions(actions, self.action_spaces)

        next_state, collision_types = self._get_next_state(state, clipped_actions)
        obs = self._get_obs(next_state)
        rewards = self._get_rewards(state, next_state, collision_types)
        terminated = {i: any(next_state[int(i)].status) for i in self.possible_agents}
        truncated = {i: False for i in self.possible_agents}
        all_done = all(terminated.values())

        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}
        for idx in range(len(self.possible_agents)):
            if next_state[idx].status[0]:
                outcome_i = M.Outcome.WIN
            elif next_state[idx].status[1]:
                outcome_i = M.Outcome.LOSS
            else:
                outcome_i = M.Outcome.NA
            info[str(idx)]["outcome"] = outcome_i

        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, info
        )

    def _get_next_state(
        self, state: DState, actions: Dict[M.AgentID, DAction]
    ) -> Tuple[DState, List[CollisionType]]:
        for i in range(len(self.possible_agents)):
            self.world.set_entity_state(f"vehicle_{i}", state[i].coord)
            if state[i].status[1]:
                # do nothing
                continue
            action = actions[str(i)]
            angle = state[i].coord[2] + action[0]
            vel = action[1] * Vec2d(1, 0).rotated(angle)
            self.world.update_entity_state(f"vehicle_{i}", angle=angle, vel=vel)

        self.world.simulate(1.0 / 10, 10)

        exec_order = list(range(len(self.possible_agents)))
        self.rng.shuffle(exec_order)

        collision_types = [CollisionType.NONE] * len(self.possible_agents)

        new_state: List[Optional[VehicleState]] = [None] * len(exec_order)

        for idx in exec_order:
            coord = np.array(
                self.world.get_entity_state(f"vehicle_{idx}"), dtype=np.float32
            )

            state_i = state[idx]
            dest_distance = np.linalg.norm(state_i.dest_coord[:2] - coord[:2])
            crashed = False

            for other_state in new_state:
                if other_state is None:
                    continue
                dist = np.linalg.norm(other_state.coord[:2] - coord[:2])
                if dist <= self.COLLISION_DIST:
                    coord = state[idx].coord
                    crashed = True
                    collision_types[idx] = CollisionType.VEHICLE

            # for (b, radius) in self.world.blocks:
            #     dist = np.linalg.norm(b[:2] - coord[:2])

            #     if dist <= self.COLLISION_DIST:
            #         coord = state[idx].coord
            #         crashed = True
            #         collision_types[idx] = CollisionType.OBSTACLE
            crashed = crashed or bool(state_i.status[1])
            new_state[idx] = VehicleState(
                coord=coord,
                dest_coord=state_i.dest_coord,
                status=np.array(
                    [int(dest_distance <= self.COLLISION_DIST), int(crashed)],
                    dtype=np.int8,
                ),
                min_dest_dist=np.array([1], dtype=np.float32),
            )

        final_state = [ns for ns in new_state if ns is not None]
        assert len(final_state) == len(new_state)

        return tuple(final_state), collision_types

    def _reset_vehicle(
        self, v_idx: int, vs_i: VehicleState, vehicle_coords: Set[Position]
    ) -> VehicleState:
        if not any(vs_i.status):
            return vs_i

        start_coords_i = self.world.start_coords[v_idx]
        avail_start_coords = start_coords_i.difference(vehicle_coords)
        vs_i_pos: Position = (vs_i.coord[0], vs_i.coord[1], vs_i.coord[2])

        if vs_i_pos in start_coords_i:
            # add it back in since it will be remove during difference op
            avail_start_coords.add(vs_i_pos)

        new_coord = self.rng.choice(list(avail_start_coords))
        vehicle_state = np.zeros((PMBodyState.num_features()), dtype=np.float32)
        vehicle_state[:3] = new_coord

        min_dest_dist = self.world.euclidean_dist(new_coord, vs_i.dest_coord[:3])

        new_vs_i = VehicleState(
            coord=vehicle_state,
            dest_coord=vs_i.dest_coord,
            status=np.array([int(False), int(False)], dtype=np.int8),
            min_dest_dist=np.array([min_dest_dist], dtype=np.float32),
        )
        return new_vs_i

    def _get_obs(self, state: DState) -> Dict[M.AgentID, DObs]:
        return {i: self._get_local_obs(i, state) for i in self.possible_agents}

    def _get_local_obs(self, agent_id: M.AgentID, state: DState) -> np.ndarray:
        state_i = state[int(agent_id)]

        pos_i = (state_i.coord[0], state_i.coord[1], state_i.coord[2])

        vehicle_coords = np.array(
            [
                [s.coord[0], s.coord[1]]
                for i, s in enumerate(state)
                if i != int(agent_id)
            ]
        )
        vehicle_obs = self.world.check_collision_circular_rays(
            pos_i,
            self.obs_dist,
            self.n_sensors,
            vehicle_coords,
            include_blocks=False,
            check_border=False,
            use_relative_angle=True,
        )

        obstacle_obs = self.world.check_collision_circular_rays(
            pos_i,
            self.obs_dist,
            self.n_sensors,
            other_agents=None,
            include_blocks=True,
            check_border=True,
            use_relative_angle=True,
        )

        dest_obs = self.world.check_collision_circular_rays(
            pos_i,
            self.obs_dist,
            self.n_sensors,
            np.expand_dims(state_i.dest_coord[:2], axis=0),
            include_blocks=True,
            check_border=True,
            use_relative_angle=True,
        )

        obs = np.full((self.obs_dim,), self.obs_dist, dtype=np.float32)
        sensor_readings = np.vstack([obstacle_obs, vehicle_obs, dest_obs])
        min_idx = np.argmin(sensor_readings, axis=0)
        min_val = np.take_along_axis(
            sensor_readings, np.expand_dims(min_idx, axis=0), axis=0
        ).flatten()
        idx = np.ravel_multi_index(
            (min_idx, np.arange(self.n_sensors)), dims=sensor_readings.shape
        )
        obs[idx] = min_val

        return obs

    def _get_rewards(
        self, state: DState, next_state: DState, collision_types: List[CollisionType]
    ) -> Dict[M.AgentID, float]:
        rewards: Dict[M.AgentID, float] = {}
        for i in self.possible_agents:
            idx = int(i)
            if any(state[idx].status):
                # already in terminal/rewarded state
                r_i = 0.0
            elif (
                self._obstacle_collisions
                and collision_types[idx] == CollisionType.OBSTACLE
            ) or collision_types[idx] == CollisionType.VEHICLE:
                # Treat as if crashed into a vehicle
                r_i = self.R_CRASH_VEHICLE
            elif next_state[idx].status[0]:
                r_i = self.R_DESTINATION_REACHED
            else:
                r_i = self.R_STEP_COST

            progress = (state[idx].min_dest_dist - next_state[idx].min_dest_dist)[0]
            r_i += max(0, progress) * self.R_PROGRESS

            if (
                not self._obstacle_collisions
                and collision_types[idx] == CollisionType.OBSTACLE
            ):
                r_i += self.R_CRASH_OBJECT
            rewards[i] = r_i
        return rewards


class DrivingGrid(SquareContinuousWorld):
    """A grid for the Driving Problem."""

    def __init__(
        self,
        size: int,
        blocks: Optional[List[CircleEntity]],
        start_coords: List[Set[Position]],
        dest_coords: List[Set[Position]],
    ):
        super().__init__(size=size, blocks=blocks, agent_radius=0.5)
        assert len(start_coords) == len(dest_coords)
        self.start_coords = start_coords
        self.dest_coords = dest_coords
        self.shortest_paths = self.get_all_shortest_paths(set.union(*dest_coords))

    @property
    def supported_num_agents(self) -> int:
        """Get the number of agents supported by this grid."""
        return len(self.start_coords)

    def get_shortest_path_distance(self, coord: Position, dest: Position) -> int:
        """Get the shortest path distance from coord to destination."""
        coord_c = self.convert_position_to_coord(coord)
        dest_c = self.convert_position_to_coord(dest)
        return int(self.shortest_paths[dest_c][coord_c])

    def get_max_shortest_path_distance(self) -> int:
        """Get the longest shortest path distance to any destination."""
        return int(max([max(d.values()) for d in self.shortest_paths.values()]))


def parseworld_str(grid_str: str, supported_num_agents: int) -> DrivingGrid:
    """Parse a str representation of a grid.

    Notes on grid str representation:

    . = empty/unallocated cell
    # = a block
    0, 1, ..., 9 = starting location for agent with given index
    + = starting point for any agent
    a, b, ..., j = destination location for agent with given index
                   (a=0, b=1, ..., j=9)
    - = destination location for any agent

    Examples (" " quotes and newline chars omitted):

    1. A 3x3 grid with two agents, one block, and where each agent has a single
    starting location and a single destination location.

    a1.
    .#.
    .0.

    2. A 6x6 grid with 4 common start and destination locations and many
       blocks. This grid can support up to 4 agents.

    +.##+#
    ..+#.+
    #.###.
    #.....
    -..##.
    #-.-.-

    """
    row_strs = grid_str.splitlines()
    assert len(row_strs) > 1
    assert all(len(row) == len(row_strs[0]) for row in row_strs)
    assert len(row_strs[0]) > 1

    grid_height = len(row_strs)
    grid_width = len(row_strs[0])

    agent_start_chars = set(["+"] + [str(i) for i in range(10)])
    agent_dest_chars = set(["-"] + list("abcdefghij"))

    block_coords: List[CircleEntity] = []
    shared_start_coords: Set[Position] = set()
    agent_start_coords_map: Dict[int, Set[Position]] = {}
    shared_dest_coords: Set[Position] = set()
    agent_dest_coords_map: Dict[int, Set[Position]] = {}
    for r, c in product(range(grid_height), range(grid_width)):
        coord = (c + 0.5, r + 0.5, 0.0)
        char = row_strs[r][c]

        if char == "#":
            block_coords.append((coord, 0.25))
        elif char in agent_start_chars:
            if char != "+":
                agent_id = int(char)
                if agent_id not in agent_start_coords_map:
                    agent_start_coords_map[agent_id] = set()
                agent_start_coords_map[agent_id].add(coord)
            else:
                shared_start_coords.add(coord)
        elif char in agent_dest_chars:
            if char != "-":
                agent_id = ord(char) - ord("a")
                if agent_id not in agent_dest_coords_map:
                    agent_dest_coords_map[agent_id] = set()
                agent_dest_coords_map[agent_id].add(coord)
            else:
                shared_dest_coords.add(coord)

    assert (
        len(shared_start_coords) + len(agent_start_coords_map) >= supported_num_agents
    )
    assert len(shared_dest_coords) + len(agent_dest_coords_map) >= supported_num_agents

    included_agent_ids = list({*agent_start_coords_map, *agent_dest_coords_map})
    if len(included_agent_ids) > 0:
        assert max(included_agent_ids) < supported_num_agents

    start_coords: List[Set[Position]] = []
    dest_coords: List[Set[Position]] = []
    for i in range(supported_num_agents):
        agent_start_coords = set(shared_start_coords)
        agent_start_coords.update(agent_start_coords_map.get(i, {}))
        start_coords.append(agent_start_coords)

        agent_dest_coords = set(shared_dest_coords)
        agent_dest_coords.update(agent_dest_coords_map.get(i, {}))
        dest_coords.append(agent_dest_coords)

    return DrivingGrid(
        size=grid_width,
        blocks=block_coords,
        start_coords=start_coords,
        dest_coords=dest_coords,
    )


#  (grid_make_fn, max step_limit, )
SUPPORTEDworldS: Dict[str, Dict[str, Any]] = {
    "3x3": {
        "grid_str": ("a1.\n" ".#.\n" ".0b\n"),
        "supported_num_agents": 2,
        "max_episode_steps": 15,
    },
    "6x6Intersection": {
        "grid_str": ("##0b##\n" "##..##\n" "d....3\n" "2....c\n" "##..##\n" "##a1##\n"),
        "supperted_num_agets": 4,
        "max_episode_steps": 20,
    },
    "7x7Blocks": {
        "grid_str": (
            "#-...-#\n"
            "-##.##+\n"
            ".##.##.\n"
            ".......\n"
            ".##.##.\n"
            "-##.##+\n"
            "#+...+#\n"
        ),
        "supported_num_agents": 4,
        "max_episode_steps": 50,
    },
    "7x7CrissCross": {
        "grid_str": (
            "#-#-#-#\n"
            "-.....+\n"
            "#.#.#.#\n"
            "-.....+\n"
            "#.#.#.#\n"
            "-.....+\n"
            "#+#+#+#\n"
        ),
        "supported_num_agents": 6,
        "max_episode_steps": 50,
    },
    "7x7RoundAbout": {
        "grid_str": (
            "#-...-#\n"
            "-##.##+\n"
            ".#...#.\n"
            "...#...\n"
            ".#...#.\n"
            "-##.##+\n"
            "#+...+#\n"
        ),
        "supported_num_agents": 4,
        "max_episode_steps": 50,
    },
    "14x14Blocks": {
        "grid_str": (
            "#-..........-#\n"
            "-###.####.###+\n"
            ".###.####.###.\n"
            ".###.####.###.\n"
            "..............\n"
            ".###.####.###.\n"
            ".###.####.###.\n"
            ".###.####.###.\n"
            ".###.####.###.\n"
            "..............\n"
            ".###.####.###.\n"
            ".###.####.###.\n"
            "-###.####.###+\n"
            "#+..........+#\n"
        ),
        "supported_num_agents": 4,
        "max_episode_steps": 50,
    },
    "14x14CrissCross": {
        "grid_str": (
            "##-##-##-##-##\n"
            "##.##.##.##.##\n"
            "-............+\n"
            "##.##.##.##.##\n"
            "##.##.##.##.##\n"
            "-............+\n"
            "##.##.##.##.##\n"
            "##.##.##.##.##\n"
            "-............+\n"
            "##.##.##.##.##\n"
            "##.##.##.##.##\n"
            "-............+\n"
            "##.##.##.##.##\n"
            "##+##+##+##+##\n"
        ),
        "supported_num_agents": 8,
        "max_episode_steps": 50,
    },
    "14x14RoundAbout": {
        "grid_str": (
            "#-..........-#\n"
            "-#####..#####+\n"
            ".#####..#####.\n"
            ".#####..#####.\n"
            ".###......###.\n"
            ".###......###.\n"
            "......##......\n"
            "......##......\n"
            ".###......###.\n"
            ".###......###.\n"
            ".#####..#####.\n"
            ".#####..#####.\n"
            "-#####..#####+\n"
            "#+..........+#\n"
        ),
        "supported_num_agents": 4,
        "max_episode_steps": 50,
    },
}
