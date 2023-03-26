"""The Driving Grid World Environment."""
import enum
from itertools import product
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import math
from gymnasium import spaces
import numpy as np

import posggym.model as M
from posggym.core import DefaultEnv

from posggym.envs.continuous.core import (
    Position,
    Object,
    RectangularContinuousWorld,
    position_to_array,
    single_item_to_position,
    clip_actions,
    array_to_position,
)
from posggym.utils import seeding


class CollisionType(enum.IntEnum):
    """Type of collision for a vehicle."""

    NONE = 0
    OBSTACLE = 1
    VEHICLE = 2


class VehicleState(NamedTuple):
    """The state of a vehicle in the Driving Environment."""

    coord: np.ndarray
    speed: float
    dest_coord: np.ndarray
    dest_reached: int
    crashed: int
    min_dest_dist: float


DState = Tuple[VehicleState, ...]

# Initial direction and speed of each vehicle
INIT_DIR = 0
INIT_SPEED = 0
MAX_SPEED = 1

# Obs = (adj_obs, speed, dest_coord, dest_reached, crashed)
DObs = Tuple[Tuple[Union[int, np.ndarray], ...], np.ndarray, np.ndarray, int, int]

# This time it is acceleration
DAction = List[float]

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
    - the direction the vehicle is facing `NORTH=0`, `EAST=1`, `SOUTH=2`, `WEST=3`,
    - the speed of the vehicle: `REVERSE=0`, `STOPPED=1`, `FORWARD_SLOW=2`,
      `FORWARD_FAST=2`,
    - the `(x, y)` coordinate of the vehicles destination
    - whether the vehicle has reached it's destination or not: `1` or `0`
    - whether the vehicle has crashed or not: `1` or `0`
    - the minimum distance to the destination achieved by the vehicle in the current
      episode.

    Action Space
    ------------
    Each agent has 5 actions: `DO_NOTHING=0`, `ACCELERATE=1`, `DECELERATE=2`,
    `TURN_RIGHT=3`, `TURN_LEFT=4`

    Observation Space
    -----------------
    Each agent observes the cells in their local area, as well as their current speed,
    their destination location, whether they've reached their destination, and whether
    they've crashed. The size of the local area observed is controlled by the `obs_dims`
    parameter (default = `(3, 1, 1)`, 3 cells in front, one cell behind, and 1 cell each
    side, giving a observation size of 5x3). For each cell in the observed area the
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

    - Speed=0 (REVERSE) - vehicle moves one cell in the opposite direction to which it
        is facing
    - Speed=1 (STOPPED) - vehicle remains in same cell
    - Speed=2 (FORWARD_SLOW) - vehicle move one cell in facing direction
    - Speed=3 (FORWARD_FAST) - vehicle moves two cells in facing direction

    Accelerating increases speed by 1, while deceleration decreased speed by 1. If the
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
    - `obs_dim` - the local observation dimensions, specifying how many cells in front,
         behind, and to each side the agent observes (default = `(3, 1, 1)`, resulting
         in the agent observing a 5x3 area: 3 in front, 1 behind, 1 to each side.)
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


    For example to use the Driving environment with the `7x7RoundAbout` grid and 2
    agents, you would use:

    ```python
    import posggym
    env = posggym.make('Driving-v0', grid="7x7RoundAbout", num_agents=2)
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
        "render_modes": ["human", "ansi", "rgb_array", "rgb_array_dict"],
        "render_fps": 15,
    }

    def __init__(
        self,
        grid: Union[str, "DrivingGrid"] = "7x7RoundAbout",
        num_agents: int = 2,
        obs_dim: Tuple[int, int, int] = (3, 1, 1),
        obstacle_collisions: bool = False,
        render_mode: Optional[str] = None,
    ):
        super().__init__(
            DrivingModel(grid, num_agents, obs_dim, obstacle_collisions),
            render_mode=render_mode,
        )
        self._obs_dim = obs_dim
        self._renderer = None
        self._agent_imgs = None

    def render(self):
        return self._render_img()

    def _render_img(self):
        if self.render_mode == "human":
            import posggym.envs.continuous.render as render_lib

            model: DrivingModel = self.model  # type: ignore

            if self._renderer is None:
                self._renderer = render_lib.GWContinuousRender(
                    self.render_mode,
                    env_name="PredatorPreyContinuous",
                    domain_max=model.grid.width,
                    render_fps=self.metadata["render_fps"],
                    num_colors=len(self._state) + 2,
                    arena_size=200,
                )

            color = [
                0
                if self._state[i].dest_reached
                else (1 if self._state[i].crashed else i + 2)
                for i in range(len(self._state))
            ]

            vehicles = tuple(
                [
                    (single_item_to_position(self._state[i].coord) + (color[i],))
                    for i in range(len(self._state))
                ]
            )

            destination_color = [i + 2 for i in range(len(self._state))]

            dest = tuple(
                [
                    (
                        single_item_to_position(self._state[i].dest_coord)
                        + (destination_color[i],)
                    )
                    for i in range(len(self._state))
                ]
            )
            num_agents = len(vehicles) * 2

            sizes = [model.grid.agent_size] * (num_agents)

            holonomic = [True] * len(vehicles) + [False] * len(vehicles)
            alpha = [128] * len(vehicles) + [255] * len(vehicles)

            self._renderer.clear_render()
            self._renderer.draw_arena()
            self._renderer.draw_agents(
                dest + vehicles,
                sizes=sizes,
                is_holonomic=holonomic,
                alpha=alpha,
            )
            self._renderer.draw_blocks(model.grid.block_coords)
            self._renderer.render()

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
    obs_dims : (int, int, int)
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

    def __init__(
        self,
        grid: Union[str, "DrivingGrid"],
        num_agents: int,
        obs_dim: Tuple[int, int, int],
        obstacle_collisions: bool,
    ):
        if isinstance(grid, str):
            assert grid in SUPPORTED_GRIDS, (
                f"Unsupported grid '{grid}'. If grid argument is a string it must be "
                f"one of: {SUPPORTED_GRIDS.keys()}."
            )
            grid_info = SUPPORTED_GRIDS[grid]
            supported_num_agents: int = grid_info["supported_num_agents"]
            assert 0 < num_agents <= supported_num_agents, (
                f"Driving grid `{grid}` does not support {num_agents} agents. The "
                f"supported number of agents is from 1 up to {supported_num_agents}."
            )
            grid = parse_grid_str(
                grid_info["grid_str"], grid_info["supported_num_agents"]
            )
        else:
            assert 0 < num_agents <= grid.supported_num_agents, (
                f"Supplied DrivingGrid `{grid}` does not support {num_agents} agents. "
                "The supported number of agents is from 1 up to "
                f"{grid.supported_num_agents}."
            )

        assert obs_dim[0] > 0 and obs_dim[1] >= 0 and obs_dim[2] >= 0
        self._grid = grid
        self._obs_front, self._obs_back, self._obs_side = obs_dim
        self._obstacle_collisions = obstacle_collisions

        def _coord_space():
            return spaces.Box(
                low=np.array([-1, -1, -2 * math.pi], dtype=np.float32),
                high=np.array(
                    [self.grid.width, self.grid.height, 2 * math.pi], dtype=np.float32
                ),
            )

        self.possible_agents = tuple(str(i) for i in range(num_agents))
        self.state_space = spaces.Tuple(
            tuple(
                spaces.Tuple(
                    (
                        _coord_space(),
                        spaces.Box(
                            low=np.array([-1], dtype=np.float32),
                            high=np.array([1], dtype=np.float32),
                        ),
                        _coord_space(),  # destination coord
                        spaces.Discrete(2),  # destination reached
                        spaces.Discrete(2),  # crashed
                        # min distance to destination
                        # set this to upper bound of min shortest path distance, so
                        # state space works for generated grids as well
                        spaces.Box(
                            low=np.array([0], dtype=np.float32),
                            high=np.array(
                                [self.grid.width * self.grid.height], dtype=np.float32
                            ),
                        ),
                    )
                )
                for _ in range(len(self.possible_agents))
            )
        )
        self.action_spaces = {
            i: spaces.Box(
                low=np.array([0, -self.grid.yaw_limit], dtype=np.float32),
                high=np.array([1.0, self.grid.yaw_limit], dtype=np.float32),
            )
            for i in self.possible_agents
        }
        self.n_lines = 4
        self.obs_distance = 1

        cell_obs_space = [spaces.Discrete(4), spaces.Box(0, self.obs_distance)]
        agent_obs = spaces.Tuple(sum([cell_obs_space for i in range(self.n_lines)], []))

        self.max_speed = 10

        # obs_depth = self._obs_front + self._obs_back + 1
        # obs_width = (2 * self._obs_side) + 1
        self.observation_spaces = {
            i: spaces.Tuple(
                (
                    agent_obs,
                    spaces.Box(
                        low=np.array([-self.max_speed], dtype=np.float32),
                        high=np.array([self.max_speed], dtype=np.float32),
                    ),
                    _coord_space(),  # dest coord,
                    spaces.Discrete(2),  # dest reached
                    spaces.Discrete(2),  # crashed
                )
            )
            for i in self.possible_agents
        }
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

    @property
    def grid(self) -> "DrivingGrid":
        """The underlying grid for this model instance."""
        return self._grid

    @grid.setter
    def grid(self, grid: "DrivingGrid"):
        assert (self._grid.height, self._grid.width) == (grid.height, grid.width)
        self._grid = grid

    def sample_initial_state(self) -> DState:
        state = []
        chosen_start_coords: Set[Position] = set()
        chosen_dest_coords: Set[Position] = set()
        for i in range(len(self.possible_agents)):
            start_coords_i = self.grid.start_coords[i]
            avail_start_coords = start_coords_i.difference(chosen_start_coords)
            start_coord = self.rng.choice(list(avail_start_coords))
            chosen_start_coords.add(start_coord)

            dest_coords_i = self.grid.dest_coords[i]
            avail_dest_coords = dest_coords_i.difference(chosen_dest_coords)
            if start_coord in avail_dest_coords:
                avail_dest_coords.remove(start_coord)
            dest_coord = self.rng.choice(list(avail_dest_coords))
            chosen_dest_coords.add(dest_coord)

            dest_dist = self.grid.get_shortest_path_distance(start_coord, dest_coord)

            state_i = VehicleState(
                coord=position_to_array([start_coord]),
                speed=INIT_SPEED,
                dest_coord=position_to_array([dest_coord]),
                dest_reached=int(False),
                crashed=int(False),
                min_dest_dist=dest_dist,
            )
            state.append(state_i)
        return tuple(state)

    def sample_agent_initial_state(self, agent_id: M.AgentID, obs: DObs) -> DState:
        agent_idx = int(agent_id)
        possible_agent_start_coords = set()
        agent_dest_coords = single_item_to_position(obs[2])

        # Need to get start states for agent that are valid given initial obs
        # Need to handle possible start states for other agents
        for all_agent_start_coords in product(
            *[list(s) for s in self.grid.start_coords[: len(self.possible_agents)]]
        ):
            if len(set(all_agent_start_coords)) != len(all_agent_start_coords):
                # skip any sets of start coord that contain duplicates
                continue
            local_obs = self._get_local_cell__obs(
                agent_idx, all_agent_start_coords, agent_dest_coords
            )
            if local_obs == obs[0]:
                possible_agent_start_coords.add(all_agent_start_coords[agent_idx])

        state = []
        chosen_start_coords: Set[Position] = set()
        chosen_dest_coords: Set[Position] = set()

        agent_start_coord = self.rng.choice(list(possible_agent_start_coords))
        chosen_start_coords.add(agent_start_coord)
        chosen_dest_coords.add(agent_dest_coords)

        for i in range(len(self.possible_agents)):
            if i == agent_idx:
                start_coord = agent_start_coord
            else:
                start_coords_i = self.grid.start_coords[i]
                avail_coords = start_coords_i.difference(chosen_start_coords)
                start_coord = self.rng.choice(list(avail_coords))
                chosen_start_coords.add(start_coord)

            if i == agent_idx:
                dest_coord = agent_dest_coords
            else:
                dest_coords_i = self.grid.dest_coords[i]
                avail_coords = dest_coords_i.difference(chosen_dest_coords)
                if start_coord in avail_coords:
                    avail_coords.remove(start_coord)
                dest_coord = self.rng.choice(list(avail_coords))
                chosen_dest_coords.add(dest_coord)

            dest_dist = self.grid.get_shortest_path_distance(start_coord, dest_coord)

            state_i = VehicleState(
                coord=position_to_array([start_coord]),
                speed=INIT_SPEED,
                dest_coord=position_to_array([dest_coord]),
                dest_reached=int(False),
                crashed=int(False),
                min_dest_dist=dest_dist,
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
        terminated = {
            i: bool(next_state[int(i)].dest_reached or next_state[int(i)].crashed)
            for i in self.possible_agents
        }
        truncated = {i: False for i in self.possible_agents}
        all_done = all(terminated.values())

        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}
        for idx in range(len(self.possible_agents)):
            if next_state[idx].dest_reached:
                outcome_i = M.Outcome.WIN
            elif next_state[idx].crashed:
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
        exec_order = list(range(len(self.possible_agents)))
        self.rng.shuffle(exec_order)

        next_state = list(state)
        vehicle_coords = {single_item_to_position(vs.coord) for vs in state}
        collision_types = [CollisionType.NONE] * len(self.possible_agents)

        for idx in exec_order:
            state_i = state[idx]
            agent_id = str(idx)
            if agent_id not in actions:
                assert state_i.crashed or state_i.dest_reached
                action_i = [0.0, 0.0]
            else:
                action_i = actions[agent_id]

            next_state_i = next_state[idx]
            if state_i.dest_reached or next_state_i.crashed:
                # already at destination or crashed, or was crashed into this
                # step
                continue

            vehicle_coords.remove(single_item_to_position(state_i.coord))
            next_speed = self._get_next_speed(action_i[1], state_i.speed)

            next_coord, collision_type, hit_vehicle = self._get_next_coord(
                single_item_to_position(state_i.coord),
                next_speed,
                action_i[0],
                list(vehicle_coords),
            )
            min_dest_dist = min(
                state_i.min_dest_dist,
                self.grid.get_shortest_path_distance(
                    next_coord, single_item_to_position(state_i.dest_coord)
                ),
            )

            crashed = False
            if collision_type == CollisionType.VEHICLE and hit_vehicle is not None:
                # update state of vehicle that was hit
                crashed = True
                collision_types[idx] = collision_type
                for jdx in range(len(self.possible_agents)):
                    next_state_j = next_state[jdx]
                    if self.grid.agents_collide(
                        single_item_to_position(next_state_j.coord), hit_vehicle
                    ):
                        collision_types[jdx] = CollisionType.VEHICLE
                        next_state[jdx] = VehicleState(
                            coord=next_state_j.coord,
                            speed=next_state_j.speed,
                            dest_coord=next_state_j.dest_coord,
                            dest_reached=next_state_j.dest_reached,
                            crashed=int(True),
                            min_dest_dist=next_state_j.min_dest_dist,
                        )
                        break
            elif collision_type == CollisionType.OBSTACLE:
                crashed = self._obstacle_collisions
                collision_types[idx] = collision_type

            next_state[idx] = VehicleState(
                coord=position_to_array([next_coord]),
                speed=next_speed,
                dest_coord=state_i.dest_coord,
                dest_reached=self.grid.agents_collide(
                    next_coord, single_item_to_position(state_i.dest_coord)
                ),
                crashed=int(crashed),
                min_dest_dist=min_dest_dist,
            )

            vehicle_coords.add(next_coord)

        return tuple(next_state), collision_types

    def _reset_vehicle(
        self, v_idx: int, vs_i: VehicleState, vehicle_coords: Set[Position]
    ) -> VehicleState:
        if not (vs_i.dest_reached or vs_i.crashed):
            return vs_i

        start_coords_i = self.grid.start_coords[v_idx]
        avail_start_coords = start_coords_i.difference(vehicle_coords)

        if vs_i.coord in start_coords_i:
            # add it back in since it will be remove during difference op
            avail_start_coords.add(single_item_to_position(vs_i.coord))

        new_coord = self.rng.choice(list(avail_start_coords))

        min_dest_dist = self.grid.euclidean_dist(
            new_coord, single_item_to_position((vs_i.dest_coord))
        )

        new_vs_i = VehicleState(
            coord=position_to_array([new_coord]),
            speed=INIT_SPEED,
            dest_coord=vs_i.dest_coord,
            dest_reached=int(False),
            crashed=int(False),
            min_dest_dist=min_dest_dist,
        )
        return new_vs_i

    @staticmethod
    def _get_next_speed(delta: float, curr_speed: float) -> float:
        return np.clip(delta + curr_speed, -MAX_SPEED, MAX_SPEED)

    def _get_next_coord(
        self,
        curr_coord: Position,
        speed: float,
        angular_dir: float,
        vehicle_coords: List[Position],
    ) -> Tuple[Position, CollisionType, Optional[Position]]:
        # assumes curr_coord isn't in vehicle coords
        next_coord = curr_coord
        collision = CollisionType.NONE
        hit_vehicle_coord = None

        next_coord = self.grid.get_next_coord(
            curr_coord, [angular_dir, speed], ignore_blocks=False
        )
        idx = self.grid.get_collision_index(next_coord, vehicle_coords)

        if self.grid.agents_collide(next_coord, curr_coord):
            collision = CollisionType.OBSTACLE

        elif idx is not None:
            collision = CollisionType.VEHICLE
            hit_vehicle_coord = vehicle_coords[idx]
            # next_coord = curr_coord

        curr_coord = next_coord

        return (next_coord, collision, hit_vehicle_coord)

    def _get_obs(self, state: DState) -> Dict[M.AgentID, DObs]:
        obs: Dict[M.AgentID, DObs] = {}
        for i in self.possible_agents:
            idx = int(i)
            local_cell_obs = self._get_local_cell__obs(
                idx,
                array_to_position(np.array([vs.coord for vs in state])),
                single_item_to_position(state[idx].dest_coord),
            )
            obs[i] = (
                local_cell_obs,
                np.array([state[idx].speed], dtype=np.float32),
                state[idx].dest_coord,
                state[idx].dest_reached,
                state[idx].crashed,
            )
        return obs

    def _get_local_cell__obs(
        self,
        agent_idx: int,
        vehicle_coords: Sequence[Position],
        dest_coord: Position,
    ) -> Tuple[Union[int, np.ndarray], ...]:
        agent_coord = vehicle_coords[agent_idx]

        cell_obs: List[Union[int, np.ndarray]] = []

        for i in range(self.n_lines):
            angle = 2 * math.pi * i / self.n_lines

            closest_agent_index, closest_agent_distance = self.grid.check_collision_ray(
                agent_coord,
                self.obs_distance,
                angle,
                other_agents=[dest_coord] + list(vehicle_coords),
                skip_id=int(agent_idx),
            )

            if closest_agent_index is None:
                agent_collision = EMPTY
            elif closest_agent_index == -1:
                agent_collision = WALL
            elif closest_agent_index == 0:
                agent_collision = DESTINATION
            else:
                agent_collision = VEHICLE

            cell_obs.append(agent_collision)
            cell_obs.append(np.array([closest_agent_distance], dtype=np.float32))

        return tuple(cell_obs)

    def _get_rewards(
        self, state: DState, next_state: DState, collision_types: List[CollisionType]
    ) -> Dict[M.AgentID, float]:
        rewards: Dict[M.AgentID, float] = {}
        for i in self.possible_agents:
            idx = int(i)
            if state[idx].crashed or state[idx].dest_reached:
                # already in terminal/rewarded state
                r_i = 0.0
            elif (
                self._obstacle_collisions
                and collision_types[idx] == CollisionType.OBSTACLE
            ) or collision_types[idx] == CollisionType.VEHICLE:
                # Treat as if crashed into a vehicle
                r_i = self.R_CRASH_VEHICLE
            elif next_state[idx].dest_reached:
                r_i = self.R_DESTINATION_REACHED
            else:
                r_i = self.R_STEP_COST

            progress = state[idx].min_dest_dist - next_state[idx].min_dest_dist
            r_i += max(0, progress) * self.R_PROGRESS

            if (
                not self._obstacle_collisions
                and collision_types[idx] == CollisionType.OBSTACLE
            ):
                r_i += self.R_CRASH_OBJECT
            rewards[i] = r_i
        return rewards


class DrivingGrid(RectangularContinuousWorld):
    """A grid for the Driving Problem."""

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        block_coords: List[Object],
        start_coords: List[Set[Position]],
        dest_coords: List[Set[Position]],
    ):
        super().__init__(grid_width, grid_height, block_coords)
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
        coord_c = self.convert_position_to_coordinate(coord)
        dest_c = self.convert_position_to_coordinate(dest)
        return int(self.shortest_paths[dest_c][coord_c])

    def get_max_shortest_path_distance(self) -> int:
        """Get the longest shortest path distance to any destination."""
        return int(max([max(d.values()) for d in self.shortest_paths.values()]))


def parse_grid_str(grid_str: str, supported_num_agents: int) -> DrivingGrid:
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

    block_coords: List[Object] = []
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
        grid_width=grid_width,
        grid_height=grid_height,
        block_coords=block_coords,
        start_coords=start_coords,
        dest_coords=dest_coords,
    )


#  (grid_make_fn, max step_limit, )
SUPPORTED_GRIDS: Dict[str, Dict[str, Any]] = {
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
