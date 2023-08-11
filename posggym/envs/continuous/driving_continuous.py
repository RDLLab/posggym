"""The Driving Continuous Environment."""
import math
from itertools import product
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union, cast

import numpy as np
from gymnasium import spaces
from pymunk import Vec2d

import posggym.model as M
from posggym import logger
from posggym.core import DefaultEnv
from posggym.envs.continuous.core import (
    AGENT_COLORS,
    CollisionType,
    Coord,
    FloatCoord,
    PMBodyState,
    SquareContinuousWorld,
    clip_actions,
    generate_interior_walls,
)
from posggym.utils import seeding


class VehicleState(NamedTuple):
    """The state of a vehicle in the Driving Environment."""

    body: np.ndarray
    dest_coord: np.ndarray
    status: np.ndarray
    min_dest_dist: np.ndarray


DState = Tuple[VehicleState, ...]

# Obs = (sensor obs, dir, vx, vy, dest_coord)
DObs = np.ndarray
# angular velocity, linear velocity
DAction = np.ndarray


class DrivingContinuousEnv(DefaultEnv[DState, DObs, DAction]):
    """The Driving Continuous World Environment.

    A general-sum 2D continuous world problem involving multiple agents. Each agent
    controls a vehicle and is tasked with driving the vehicle from it's start
    location to a destination location while avoiding crashing into other vehicles.
    This requires agents to coordinate to avoid collisions and can be used to explore
    conventions in multi-agent decision-making.

    Possible Agents
    ---------------
    The environment supports two or more agents, depending on the world layout. It is
    possible for some agents to finish the episode before other agents by either
    crashing or reaching their destination, and so not all agents are guaranteed to be
    active at the same time. All agents will be active at the start of the episode.

    State Space
    -----------
    Each state is made up of the state of each vehicle (see `VehicleState` class),
    which in turn is defined by the vehicle's:

    - `(x, y)` coordinates in [0, world_size]
    - direction in [-2π, 2π]
    - x, y velocity both in [-1, 1]
    - the angular velocity of the vehicle in [-2π, 2π]
    - the `(x, y)` coordinate of the vehicles destination
    - whether the vehicle has reached it's destination or not: `1` or `0`
    - whether the vehicle has crashed or not: `1` or `0`
    - the minimum distance to the destination achieved by the vehicle in the current
      episode, if the environment was discrete.

    Action Space
    ------------
    Each agent has 2 actions, which are the angular velocity and linear acceleration.
    Each agent's actions is made up of two parts. The first action component specifies
    the angular velocity in `[-pi/4, pi/4]`, and the second component specifies the
    linear acceleration in `[-0.25, 0.25]`.

    Observation Space
    -----------------
    Each agent observes a local circle around themselves as a vector. This is achieved
    by a series of 'n_sensors' lines starting at the agent which extend for a distance
    of 'obs_dist'. For each line the agent observes the closest entity along the line,
    specifically if there is a wall or another vehicle. Along with the sensor reading
    each agent also observes their vehicles angle, velocity (in x, y), and the distance
    to their destination.

    This table enumerates the observation space:

    | Index: start          | Description                          |  Values   |
    | :-------------------: | :----------------------------------: | :-------: |
    | 0                     | Wall distance                        | [0, d]    |
    | n_sensors             | Other vehicle distance               | [0, d]    |
    | 2 * n_sensors         | Vehicle angle                        | [-2π, 2π] |
    | 2 * n_sensors + 1     | Vehicle x velocity                   | [-1, 1]   |
    | 2 * n_sensors + 2     | Vehicle y velocity                   | [-1, 1]   |
    | 2 * n_sensors + 3     | distance to destination along x axis | [0, s]    |
    | 2 * n_sensors + 4     | distance to destination along y axis | [0, s]    |

    Where `d = obs_dist` and `s = world.size`

    If an entity is not observed by a sensor (i.e. it's not within `obs_dist` or is not
    the closest entity to the observing agent along the line), The distance reading will
    be `obs_dist`.

    The sensor reading ordering is relative to the agent's direction. I.e. the values
    for the first sensor at indices `0`, `n_sensors`, correspond to the distance
    reading to a wall, and other vehicle, respectively, in the direction the agent is
    facing.

    Rewards
    -------
    All agents receive a penalty of `0.0` for each step. They receive a penalty of
    `-1.0` for crashing (i.e. hitting another vehicle). A reward of `1.0` is given if
    the agent reaches it's destination and a reward of `0.05` is given to the agent at
    certain points as it makes progress towards it's destination (i.e. as it reduces
    it's minimum distance achieved along the shortest path to the destination for the
    episode).

    Dynamics
    --------
    Actions are deterministic and movement is determined by direction the vehicle is
    facing and it's speed. Vehicles are able to reverse, but cannot change direction
    while reversing.

    Max and min velocity are `1.0` and `-1.0`, and max linear acceleration is `0.25`,
    while max angular velocity is `π / 4`.

    Starting State
    --------------
    Each agent is randomly assigned to one of the possible starting locations in the
    world and one of the possible destination locations, with no two agents starting in
    the same location or having the same destination location. The possible start and
    destination locations are determined by the world layout being used.

    Episodes End
    ------------
    Episodes end when all agents have either reached their destination or crashed. By
    default a `max_episode_steps` is also set for each DrivingContinuous environment.
    The default value is `200` steps, but this may need to be adjusted when using
    larger worlds (this can be done by manually specifying a value for
    `max_episode_steps` when creating the environment with `posggym.make`).

    Arguments
    ---------

    - `world` - the world layout to use. This can either be a string specifying one of
         the supported worlds, or a custom :class:`DrivingWorld` object
         (default = `"14x14RoundAbout"`).
    - `num_agents` - the number of agents in the environment (default = `2`).
    - `obs_dist` - the sensor observation distance, specifying the distance away from
         itself which an agent can observe along each sensor (default = `5.0`).
    - `n_sensors` - the number of sensor lines eminating from the agent. The agent will
         observe at `n_sensors` equidistance intervals over `[0, 2*pi]`
         (default = `16`).

    Available variants
    ------------------
    The DrivingContinuous environment comes with a number of pre-built world layouts
    which can be passed as an argument to `posggym.make`, to create different worlds:

    | World name        | Max number of agents | World size |
    |-------------------|----------------------|----------- |
    | `6x6`             | 6                    | 6x6        |
    | `7x7Blocks`       | 4                    | 7x7        |
    | `7x7CrissCross`   | 6                    | 7x7        |
    | `7x7RoundAbout`   | 4                    | 7x7        |
    | `14x14Blocks`     | 4                    | 14x14      |
    | `14x14CrissCross` | 8                    | 14x14      |
    | `14x14RoundAbout` | 4                    | 14x14      |


    For example to use the DrivingContinuous environment with the `7x7RoundAbout`
    layout and 2 agents, you would use:

    ```python
    import posggym
    env = posggym.make('DrivingContinuous-v0', world="7x7RoundAbout", num_agents=2)
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
        world: Union[str, "DrivingWorld"] = "14x14RoundAbout",
        num_agents: int = 2,
        obs_dist: float = 5.0,
        n_sensors: int = 16,
        render_mode: Optional[str] = None,
    ):
        super().__init__(
            DrivingContinuousModel(world, num_agents, obs_dist, n_sensors),
            render_mode=render_mode,
        )
        self.window_surface = None
        self.clock = None
        self.window_size = 600
        self.draw_options = None
        self.world = None
        self.blocked_surface = None

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
        # import posggym.envs.continuous.render as render_lib
        import pygame
        from pymunk import Transform, pygame_util

        model = cast(DrivingContinuousModel, self.model)
        state = cast(DState, self.state)
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

        for i, v_state in enumerate(state):
            self.world.set_entity_state(f"vehicle_{i}", v_state.body)

        # Need to do this for space to update with changes
        self.world.space.step(0.0001)

        # reset screen
        self.window_surface.fill(pygame.Color("white"))

        # add blocks
        self.window_surface.blit(self.blocked_surface, (0, 0))

        lines_colors = ["red", "green", "black"]
        # draw sensor lines
        n_sensors = model.n_sensors
        for i, obs_i in self._last_obs.items():
            line_obs = obs_i[: model.sensor_obs_dim]
            x, y, agent_angle = state[int(i)].body[:3]
            angle_inc = 2 * math.pi / n_sensors
            for k in range(n_sensors):
                values = [
                    line_obs[k],
                    line_obs[n_sensors + k],
                ]
                dist_idx = min(range(len(values)), key=values.__getitem__)
                dist = values[dist_idx]
                dist_idx = len(values) if dist == model.obs_dist else dist_idx

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

        for i, v in enumerate(state):
            x, y = v.dest_coord[:2]
            scaled_center = (int(x * scale_factor), int(y * scale_factor))
            scaled_r = int(model.world.agent_radius * scale_factor)
            pygame.draw.circle(
                self.window_surface,
                AGENT_COLORS[i][1],
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


class DrivingContinuousModel(M.POSGModel[DState, DObs, DAction]):
    """Driving Problem Model.

    Parameters
    ----------
    world : str, DrivingWorld
        the world environment for the model scenario
    num_agents : int
        the number of agents in the model scenario
    obs_dists : float
        number of cells in front, behind, and to the side that each agent
        can observe
    n_sensors : the number of sensor lines eminating from the agent. The agent will
        observe at `n_sensors` equidistance intervals over `[0, 2*pi]`

    """

    R_STEP_COST = 0.00
    R_CRASH_VEHICLE = -1.0
    R_DESTINATION_REACHED = 1.0
    R_PROGRESS = 0.05

    def __init__(
        self,
        world: Union[str, "DrivingWorld"],
        num_agents: int,
        obs_dist: float,
        n_sensors: int,
    ):
        if isinstance(world, str):
            assert world in SUPPORTED_WORLDS, (
                f"Unsupported world '{world}'. If world argument is a string it must "
                f"be one of: {SUPPORTED_WORLDS.keys()}."
            )
            world_info = SUPPORTED_WORLDS[world]
            world = parseworld_str(
                world_info["world_str"], world_info["supported_num_agents"]
            )
        assert 0 < num_agents <= world.supported_num_agents, (
            f"Supplied DrivingWorld `{world}` does not support {num_agents} "
            "agents. The supported number of agents is from 1 up to "
            f"{world.supported_num_agents}."
        )

        self.world = world
        self.n_sensors = n_sensors
        self.obs_dist = obs_dist
        self.vehicle_collision_dist = 2.1 * self.world.agent_radius

        self.possible_agents = tuple(str(i) for i in range(num_agents))
        self.state_space = spaces.Tuple(
            tuple(
                spaces.Tuple(
                    (
                        # State of vehicle
                        PMBodyState.get_space(self.world.size),
                        # vehicle destination coord
                        spaces.Box(low=0.0, high=self.world.size, shape=(2,)),
                        # dest reached, crashed
                        spaces.MultiBinary(2),
                        # min distance to destination
                        # set this to upper bound of min shortest path distance, so
                        # state space works for generated worlds as well
                        spaces.Box(
                            low=np.array([0], dtype=np.float32),
                            high=np.array([self.world.size**2], dtype=np.float32),
                        ),
                    )
                )
                for _ in range(len(self.possible_agents))
            )
        )

        self.dyaw_limit = math.pi / 4
        self.dvel_limit = 0.25
        self.vel_limit_norm = 1.0
        # dyaw, dvel
        self.action_spaces = {
            i: spaces.Box(
                low=np.array([-self.dyaw_limit, -self.dvel_limit], dtype=np.float32),
                high=np.array([self.dyaw_limit, self.dvel_limit], dtype=np.float32),
            )
            for i in self.possible_agents
        }

        # Observes entity and distance to entity along a n_sensors rays from the agent
        # 0 to n_sensors = wall distance obs
        # n_sensors to (2 * n_sensors) = other vehicle dist
        # Also observs angle, vx, vy, dest dx, desy dy
        self.sensor_obs_dim = self.n_sensors * 2
        self.obs_dim = self.sensor_obs_dim + 5
        sensor_low = [0.0] * self.sensor_obs_dim
        sensor_high = [self.obs_dist] * self.sensor_obs_dim
        self.observation_spaces = {
            i: spaces.Box(
                low=np.array(
                    [*sensor_low, -2 * math.pi, -1, -1, 0, 0], dtype=np.float32
                ),
                high=np.array(
                    [*sensor_high, 2 * math.pi, 1, 1, self.world.size, self.world.size],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )
            for i in self.possible_agents
        }

        # Add physical entities to the world
        for i in range(len(self.possible_agents)):
            color_i, _ = AGENT_COLORS[i]
            self.world.add_entity(f"vehicle_{i}", None, color=color_i)

        self.is_symmetric = True

    @property
    def reward_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            i: (self.R_CRASH_VEHICLE, self.R_DESTINATION_REACHED)
            for i in self.possible_agents
        }

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: DState) -> List[str]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> DState:
        state = []
        chosen_start_coords: Set[FloatCoord] = set()
        chosen_dest_coords: Set[FloatCoord] = set()
        for i in range(len(self.possible_agents)):
            start_coords_i = self.world.start_coords[i]
            avail_start_coords = start_coords_i.difference(chosen_start_coords)
            start_coord = self.rng.choice(list(avail_start_coords))
            chosen_start_coords.add(start_coord)

            dest_coords_i = self.world.dest_coords[i]
            avail_dest_coords = dest_coords_i.difference(chosen_dest_coords)
            if start_coord in avail_dest_coords:
                avail_dest_coords.remove(start_coord)

            body_state = np.zeros((PMBodyState.num_features()), dtype=np.float32)
            body_state[:2] = start_coord

            _dest_coord = self.rng.choice(list(avail_dest_coords))
            chosen_dest_coords.add(_dest_coord)
            dest_coord = np.array(_dest_coord, dtype=np.float32)

            dest_dist = self.world.get_shortest_path_distance(start_coord, _dest_coord)

            state_i = VehicleState(
                body=body_state,
                dest_coord=dest_coord,
                status=np.array([int(False), int(False)], dtype=np.int8),
                min_dest_dist=np.array([dest_dist], dtype=np.float32),
            )
            state.append(state_i)

        return tuple(state)

    def sample_initial_obs(self, state: DState) -> Dict[str, DObs]:
        return self._get_obs(state)

    def step(
        self, state: DState, actions: Dict[str, DAction]
    ) -> M.JointTimestep[DState, DObs]:
        clipped_actions = clip_actions(actions, self.action_spaces)

        next_state, collision_types = self._get_next_state(state, clipped_actions)
        obs = self._get_obs(next_state)
        rewards = self._get_rewards(state, next_state, collision_types)
        terminated = {i: any(next_state[int(i)].status) for i in self.possible_agents}
        truncated = {i: False for i in self.possible_agents}
        all_done = all(terminated.values())

        info: Dict[str, Dict] = {i: {} for i in self.possible_agents}
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
        self, state: DState, actions: Dict[str, DAction]
    ) -> Tuple[DState, List[CollisionType]]:
        for i in range(len(self.possible_agents)):
            state_i = state[i]
            self.world.set_entity_state(f"vehicle_{i}", state_i.body)

            if state[i].status[0] or state[i].status[1]:
                self.world.update_entity_state(f"vehicle_{i}", vel=(0.0, 0.0))
                continue

            action_i = actions[str(i)]
            v_angle = state_i.body[2] + action_i[0]
            v_vel = Vec2d(*state_i.body[3:5]).rotated(action_i[0]) + (
                action_i[1] * Vec2d(1, 0).rotated(v_angle)
            )
            self.world.update_entity_state(
                f"vehicle_{i}",
                angle=v_angle,
                vel=self.world.clamp_norm(v_vel[0], v_vel[1], self.vel_limit_norm),
            )

        self.world.simulate(1.0 / 10, 10)

        collision_types = [CollisionType.NONE] * len(self.possible_agents)
        new_state: List[Optional[VehicleState]] = [None] * len(self.possible_agents)
        for idx in range(len(self.possible_agents)):
            next_v_body_state = np.array(
                self.world.get_entity_state(f"vehicle_{idx}"), dtype=np.float32
            )

            next_v_body_state[2] = self.world.convert_angle_to_0_2pi_interval(
                next_v_body_state[2]
            )

            # ensure vx, vy is in [-1, 1]
            # with collisions, etc pymunk can sometime push it over this limit
            next_v_body_state[3] = max(-1.0, min(1.0, next_v_body_state[3]))
            next_v_body_state[4] = max(-1.0, min(1.0, next_v_body_state[4]))

            state_i = state[idx]
            next_v_coords = next_v_body_state[:2]
            dest_distance = np.linalg.norm(state_i.dest_coord - next_v_coords)
            crashed = False

            for other_idx, other_v_state in enumerate(new_state):
                if other_v_state is None:
                    continue
                dist = np.linalg.norm(other_v_state.body[:2] - next_v_coords)
                if dist <= self.vehicle_collision_dist:
                    crashed = True
                    collision_types[idx] = CollisionType.AGENT
                    if not other_v_state.status[0]:
                        # must also update other vehicle but only update other vehicle
                        # if it has not already reached destination
                        other_v_state.status[1] = int(True)
                        collision_types[other_idx] = CollisionType.AGENT

            crashed = crashed or bool(state_i.status[1])

            min_dest_dist = min(
                state_i.min_dest_dist[0],
                self.world.get_shortest_path_distance(
                    (next_v_body_state[0], next_v_body_state[1]),
                    (state_i.dest_coord[0], state_i.dest_coord[1]),
                ),
            )

            new_state[idx] = VehicleState(
                body=next_v_body_state,
                dest_coord=state_i.dest_coord,
                status=np.array(
                    [int(dest_distance <= self.world.agent_radius), int(crashed)],
                    dtype=np.int8,
                ),
                min_dest_dist=np.array([min_dest_dist], dtype=np.float32),
            )

        final_state = [ns for ns in new_state if ns is not None]
        assert len(final_state) == len(new_state)

        return tuple(final_state), collision_types

    def _get_obs(self, state: DState) -> Dict[str, DObs]:
        return {i: self._get_agent_obs(i, state) for i in self.possible_agents}

    def _get_agent_obs(self, agent_id: str, state: DState) -> np.ndarray:
        state_i = state[int(agent_id)]
        if state_i.status[0] or state_i.status[1]:
            return np.zeros((self.obs_dim,), dtype=np.float32)

        pos_i = (state_i.body[0], state_i.body[1], state_i.body[2])
        vehicle_coords = np.array(
            [[s.body[0], s.body[1]] for i, s in enumerate(state) if i != int(agent_id)]
        )

        ray_dists, ray_col_type = self.world.check_collision_circular_rays(
            pos_i,
            ray_distance=self.obs_dist,
            n_rays=self.n_sensors,
            other_agents=vehicle_coords,
            include_blocks=True,
            check_walls=True,
            use_relative_angle=True,
        )

        obs = np.full((self.obs_dim,), self.obs_dist, dtype=np.float32)
        # Can bucket NONE type collisions with block/border/wall collisions, since these
        # will have distance of obs_dist and hence not affect final obs
        obs_type_idx = np.where(ray_col_type == CollisionType.AGENT.value, 1, 0)
        flat_obs_idx = np.ravel_multi_index(
            (obs_type_idx, np.arange(self.n_sensors)), dims=(2, self.n_sensors)
        )
        obs[flat_obs_idx] = ray_dists

        d = self.sensor_obs_dim
        obs[d] = self.world.convert_angle_to_0_2pi_interval(state_i.body[2])
        obs[d + 1] = max(-1.0, min(1.0, state_i.body[3]))
        obs[d + 2] = max(-1.0, min(1.0, state_i.body[4]))
        obs[d + 3] = abs(state_i.dest_coord[0] - pos_i[0])
        obs[d + 4] = abs(state_i.dest_coord[1] - pos_i[1])

        return obs

    def _get_rewards(
        self, state: DState, next_state: DState, collision_types: List[CollisionType]
    ) -> Dict[str, float]:
        rewards: Dict[str, float] = {}
        for i in self.possible_agents:
            idx = int(i)
            if any(state[idx].status):
                # already in terminal/rewarded state
                r_i = 0.0
            elif collision_types[idx] == CollisionType.AGENT:
                # Treat as if crashed into a vehicle
                r_i = self.R_CRASH_VEHICLE
            elif next_state[idx].status[0]:
                r_i = self.R_DESTINATION_REACHED
            else:
                r_i = self.R_STEP_COST

            progress = (state[idx].min_dest_dist - next_state[idx].min_dest_dist)[0]
            r_i += max(0, progress) * self.R_PROGRESS
            rewards[i] = r_i
        return rewards


class DrivingWorld(SquareContinuousWorld):
    """A world for the Driving Problem."""

    def __init__(
        self,
        size: int,
        blocked_coords: Set[Coord],
        start_coords: List[Set[FloatCoord]],
        dest_coords: List[Set[FloatCoord]],
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
        assert len(start_coords) == len(dest_coords)
        self._blocked_coords = blocked_coords
        self.start_coords = start_coords
        self.dest_coords = dest_coords
        self.shortest_paths = self.get_all_shortest_paths(
            set.union(*dest_coords)  # type: ignore
        )

    def copy(self) -> "DrivingWorld":
        assert self._blocked_coords is not None
        world = DrivingWorld(
            size=int(self.size),
            blocked_coords=self._blocked_coords,
            start_coords=self.start_coords,
            dest_coords=self.dest_coords,
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
    def supported_num_agents(self) -> int:
        """Get the number of agents supported by this world."""
        return len(self.start_coords)

    def get_shortest_path_distance(self, coord: FloatCoord, dest: FloatCoord) -> int:
        """Get the shortest path distance from coord to destination."""
        coord_c = self.convert_to_coord(coord)
        dest_c = self.convert_to_coord(dest)
        return int(self.shortest_paths[dest_c][coord_c])

    def get_max_shortest_path_distance(self) -> int:
        """Get the longest shortest path distance to any destination."""
        return int(max([max(d.values()) for d in self.shortest_paths.values()]))


def parseworld_str(world_str: str, supported_num_agents: int) -> DrivingWorld:
    """Parse a str representation of a world.

    Notes on world str representation:

    . = empty/unallocated cell
    # = a block
    0, 1, ..., 9 = starting location for agent with given index
    + = starting point for any agent
    a, b, ..., j = destination location for agent with given index
                   (a=0, b=1, ..., j=9)
    - = destination location for any agent

    Examples (" " quotes and newline chars omitted):

    1. A 3x3 world with two agents, one block, and where each agent has a single
    starting location and a single destination location.

    a1.
    .#.
    .0.

    2. A 6x6 world with 4 common start and destination locations and many
       blocks. This world can support up to 4 agents.

    +.##+#
    ..+#.+
    #.###.
    #.....
    -..##.
    #-.-.-

    """
    row_strs = world_str.splitlines()
    assert len(row_strs) > 1
    assert all(len(row) == len(row_strs[0]) for row in row_strs)
    assert len(row_strs[0]) > 1

    height = len(row_strs)
    width = len(row_strs[0])

    agent_start_chars = set(["+"] + [str(i) for i in range(10)])
    agent_dest_chars = set(["-"] + list("abcdefghij"))

    blocked_coords: Set[Coord] = set()
    shared_start_coords: Set[FloatCoord] = set()
    agent_start_coords_map: Dict[int, Set[FloatCoord]] = {}
    shared_dest_coords: Set[FloatCoord] = set()
    agent_dest_coords_map: Dict[int, Set[FloatCoord]] = {}
    for r, c in product(range(height), range(width)):
        coord = (c + 0.5, r + 0.5)
        char = row_strs[r][c]

        if char == "#":
            blocked_coords.add((c, r))
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

    start_coords: List[Set[FloatCoord]] = []
    dest_coords: List[Set[FloatCoord]] = []
    for i in range(supported_num_agents):
        agent_start_coords = set(shared_start_coords)
        agent_start_coords.update(agent_start_coords_map.get(i, {}))
        start_coords.append(agent_start_coords)

        agent_dest_coords = set(shared_dest_coords)
        agent_dest_coords.update(agent_dest_coords_map.get(i, {}))
        dest_coords.append(agent_dest_coords)

    return DrivingWorld(
        size=width,
        blocked_coords=blocked_coords,
        start_coords=start_coords,
        dest_coords=dest_coords,
    )


SUPPORTED_WORLDS: Dict[str, Dict[str, Any]] = {
    "6x6Intersection": {
        "world_str": (
            "##0b##\n" "##..##\n" "d....3\n" "2....c\n" "##..##\n" "##a1##\n"
        ),
        "supported_num_agents": 4,
        "max_episode_steps": 20,
    },
    "7x7Blocks": {
        "world_str": (
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
        "world_str": (
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
        "world_str": (
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
        "world_str": (
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
        "world_str": (
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
        "world_str": (
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
