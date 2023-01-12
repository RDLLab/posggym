"""The Driving Grid World Environment.

A general-sum 2D grid world problem involving multiple agents. Each agent
controls a vehicle and is tasked with driving the vehicle from it's start
location to a destination location while avoiding crashing into obstacles
or other vehicles.

This environment requires each agent to navigate in the world while also
taking care to avoid crashing into other players. The dynamics and
observations of the environment are such that avoiding collisions requires
some planning in order for the vehicle to brake in time or maintain a good
speed. Depending on the grid layout, the environment will require agents to
reason about and possibly coordinate with the other vehicles.

References
----------
- Adam Lerer and Alexander Peysakhovich. 2019. Learning Existing Social Conventions via
Observationally Augmented Self-Play. In Proceedings of the 2019 AAAI/ACM Conference on
AI, Ethics, and Society. 107–114.
- Kevin R. McKee, Joel Z. Leibo, Charlie Beattie, and Richard Everett. 2022.
Quantifying the Effects of Environment and Population Diversity in Multi-Agent
Reinforcement Learning. Autonomous Agents and Multi-Agent Systems 36, 1 (2022), 1–16

"""
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
    SupportsFloat,
    Tuple,
    Union,
)

from gymnasium import spaces

import posggym.envs.grid_world.render as render_lib
import posggym.model as M
from posggym.core import DefaultEnv
from posggym.envs.grid_world.core import (
    DIRECTION_ASCII_REPR,
    Coord,
    Direction,
    Grid,
    GridCycler,
    GridGenerator,
)
from posggym.utils import seeding


class Speed(enum.IntEnum):
    """A speed setting for a vehicle."""

    REVERSE = 0
    STOPPED = 1
    FORWARD_SLOW = 2
    FORWARD_FAST = 3


class CollisionType(enum.IntEnum):
    """Type of collision for a vehicle."""

    NONE = 0
    OBSTACLE = 1
    VEHICLE = 2


class VehicleState(NamedTuple):
    """The state of a vehicle in the Driving Environment."""

    coord: Coord
    facing_dir: Direction
    speed: Speed
    dest_coord: Coord
    dest_reached: int
    crashed: int
    min_dest_dist: int


DState = Tuple[VehicleState, ...]

# Initial direction and speed of each vehicle
INIT_DIR = Direction.NORTH
INIT_SPEED = Speed.STOPPED

# The actions
DAction = int
DO_NOTHING = 0
ACCELERATE = 1
DECELERATE = 2
TURN_RIGHT = 3
TURN_LEFT = 4

ACTIONS = [DO_NOTHING, ACCELERATE, DECELERATE, TURN_RIGHT, TURN_LEFT]
ACTIONS_STR = ["0", "acc", "dec", "tr", "tl"]

# Obs = (adj_obs, speed, dest_coord, dest_reached, crashed)
DObs = Tuple[Tuple[int, ...], Speed, Coord, int, int]

# Cell obs
VEHICLE = 0
WALL = 1
EMPTY = 2
DESTINATION = 3

CELL_OBS = [VEHICLE, WALL, EMPTY, DESTINATION]
CELL_OBS_STR = ["V", "#", "0", "D"]


class DrivingEnv(DefaultEnv[DState, DObs, DAction]):
    """The Driving Grid World Environment.

    A general-sum 2D grid world problem involving multiple agents. Each agent
    controls a vehicle and is tasked with driving the vehicle from it's start
    location to a destination location while avoiding crashing into obstacles
    or other vehicles.

    This environment requires each agent to navigate in the world while also
    taking care to avoid crashing into other players. The dynamics and
    observations of the environment are such that avoiding collisions requires
    some planning in order for the vehicle to brake in time or maintain a good
    speed. Depending on the grid layout, the environment will require agents to
    reason about and possibly coordinate with the other vehicles.


    Agents
    ------
    Varied number

    State
    -----
    Each state is made up of the state of each vehicle, which in turn is
    defined by:

    - the (x, y) (x=column, y=row, with origin at the top-left square of the
      grid) of the vehicle,
    - the direction the vehicle is facing NORTH=0, SOUTH=1, EAST=2, WEST=3),
    - the speed of the vehicle (REVERSE=-1, STOPPED=0, FORWARD_SLOW=1,
      FORWARD_FAST=2),
    - the (x, y) of the vehicles destination
    - whether the vehicle has reached it's destination or not
    - whether the vehicle has crashed or not

    Actions
    -------
    Each agent has 5 actions: DO_NOTHING=0, ACCELERATE=1, DECELERATE=2,
    TURN_RIGHT=3, TURN_LEFT=4

    Observation
    -----------
    Each agent observes their current speed along with the cells in their local
    area. The size of the local area observed is controlled by the `obs_dims`
    parameter. For each cell in the observed are the agent observes whether
    they are one of four things: VEHICLE=0, WALL=1, EMPTY=2, DESTINATION=3.
    Each agent also observes the (x, y) coordinates of their destination,
    whether they have reached the destination, and whether they have crashed.

    Each observation is represented as a tuple:
        ((local obs), speed, destination coord, destination reached, crashed)

    Reward
    ------
    All agents receive a penalty of 0.00 for each step. They also recieve a
    penalty of -0.5 for hitting an obstacle (if ``obstacle_collision=True``),
    and -1.0 for hitting another vehicle. A reward of 1.0 is given if the agent
    reaches it's destination.

    Transition Dynamics
    -------------------
    Actions are deterministic and movement is determined by direction the
    vehicle is facing and it's speed:

    - Speed=-1 (REVERSE) - vehicle moves one cell in the opposite direction
    - Speed=0 (STOPPED) - vehicle remains in same cell
    - Speed=1 (FORWARD_SLOW) - vehicle move one cell in facing direction
    - Speed=1 (FORWARD_FAST) - vehicle moves two cells in facing direction

    Accelerating increases speed by 1, while deceleration decreased speed by 1.
    If the vehicle will hit a wall or an other vehicle when moving from one
    cell to another then it remains in it's current cell and its crashed state
    variable is updated. Once a vehicle reaches it's destination it is stuck.

    Episodes ends when all agents have either reached their destination or
    crashed, or the episode step limit is reached.

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
        "render_fps": 4,
    }

    def __init__(
        self,
        grid: Union[str, "DrivingGrid"],
        num_agents: int,
        obs_dim: Tuple[int, int, int],
        obstacle_collisions: bool,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            DrivingModel(grid, num_agents, obs_dim, obstacle_collisions, **kwargs),
            render_mode=render_mode,
        )
        self._obs_dim = obs_dim
        self._viewer = None
        self._renderer: Optional[render_lib.GWRenderer] = None
        # Whether to re-render blocks each new episode
        # This is needed when block positions change between episodes
        self._rerender_blocks = False

    def render(self):
        model: DrivingModel = self.model  # type: ignore
        if self.render_mode == "ansi":
            grid_str = model.grid.get_ascii_repr(
                [vs.coord for vs in self._state],
                [vs.facing_dir for vs in self._state],
                [vs.dest_coord for vs in self._state],
            )

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

        if self.render_mode == "human" and self._viewer is None:
            # pylint: disable=[import-outside-toplevel]
            from posggym.envs.grid_world import viewer

            self._viewer = viewer.GWViewer(  # type: ignore
                "Driving Env",
                (min(model.grid.width, 9), min(model.grid.height, 9)),
                num_agent_displays=len(self.possible_agents),
            )
            self._viewer.show(block=False)  # type: ignore

        blocks_added = False
        if self._renderer is None:
            blocks_added = True
            self._renderer = render_lib.GWRenderer(
                len(self.possible_agents), model.grid, [], render_blocks=True
            )

        agent_obs_coords = tuple(
            model.get_obs_coords(vs.coord, vs.facing_dir) for vs in self._state
        )
        agent_coords = tuple(vs.coord for vs in self._state)
        agent_dirs = tuple(vs.facing_dir for vs in self._state)

        # Add agent destination locations
        other_objs = [
            render_lib.GWObject(
                vs.dest_coord,
                render_lib.get_agent_color(i),
                render_lib.Shape.RECTANGLE,
                # make dest squares slightly different to vehicle color
                alpha=0.2,
            )
            for i, vs in enumerate(self._state)
        ]
        # Add blocks, as necessary
        if self._rerender_blocks and not blocks_added:
            for block_coord in model.grid.block_coords:
                other_objs.append(
                    render_lib.GWObject(block_coord, "grey", render_lib.Shape.RECTANGLE)
                )
        # Add visualization for crashed agents
        for i, vs in enumerate(self._state):
            if vs.crashed:
                other_objs.append(
                    render_lib.GWObject(vs.coord, "yellow", render_lib.Shape.CIRCLE)
                )

        env_img = self._renderer.render(
            agent_coords,
            agent_obs_coords,
            agent_dirs=agent_dirs,
            other_objs=other_objs,
            agent_colors=None,
        )
        agent_obs_imgs = self._renderer.render_all_agent_obs(
            env_img,
            agent_coords,
            agent_dirs,
            agent_obs_dims=self._obs_dim,
            out_of_bounds_obj=render_lib.GWObject(
                (0, 0), "grey", render_lib.Shape.RECTANGLE
            ),
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


class DrivingGenEnv(DrivingEnv):
    """The Generated Driving Grid World Environment.

    This is the same as the Driving Environment except that a new grid is
    generated at each reset.

    The generated Grids can be:

    1. set to be from some list of grids, in which case the grids in the list
       will be cycled through (in the same order or shuffled)
    2. generated a new each reset. Depending on grid size and generator
       parameters this will lead to possibly unique grids each episode.

    The seed parameter can be used to ensure the same grids are used on
    different runs.

    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: Tuple[int, int, int],
        obstacle_collisions: bool,
        n_grids: Optional[int],
        generator_params: Dict[str, Any],
        shuffle_grid_order: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ):
        if "seed" not in generator_params:
            generator_params["seed"] = seed

        self._n_grids = n_grids
        self._gen_params = generator_params
        self._shuffle_grid_order = shuffle_grid_order
        self._gen = DrivingGridGenerator(**generator_params)
        # Need to re-add blocks each time since these change each episode
        self._rerender_blocks = True

        if n_grids is not None:
            grids = self._gen.generate_n(n_grids)
            self._cycler = GridCycler(grids, shuffle_grid_order, seed=seed)
            grid: "DrivingGrid" = grids[0]  # type: ignore
        else:
            self._cycler = None  # type: ignore
            grid = self._gen.generate()

        self._model_kwargs = {
            "num_agents": num_agents,
            "obs_dim": obs_dim,
            "obstacle_collisions": obstacle_collisions,
            **kwargs,
        }

        super().__init__(
            grid,
            num_agents,
            obs_dim,
            obstacle_collisions=obstacle_collisions,
            seed=seed,
            **kwargs,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[Dict[M.AgentID, DObs]], Dict[M.AgentID, Dict]]:
        if seed is not None:
            self._gen_params["seed"] = seed
            self._gen = DrivingGridGenerator(**self._gen_params)

            if self._n_grids is not None:
                grids = self._gen.generate_n(self._n_grids)
                self._cycler = GridCycler(grids, self._shuffle_grid_order, seed=seed)

        if self._n_grids:
            grid = self._cycler.next()
        else:
            grid = self._gen.generate()

        self.model = DrivingModel(
            grid,  # type: ignore
            **self._model_kwargs,  # type: ignore
        )

        return super().reset(seed=seed)


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
        **kwargs,
    ):
        if isinstance(grid, str):
            grid = load_grid(grid)

        assert 0 < num_agents <= grid.supported_num_agents
        assert obs_dim[0] > 0 and obs_dim[1] >= 0 and obs_dim[2] >= 0
        self.grid = grid
        self._obs_front, self._obs_back, self._obs_side = obs_dim
        self._obstacle_collisions = obstacle_collisions

        def _coord_space():
            return spaces.Tuple(
                (spaces.Discrete(self.grid.width), spaces.Discrete(self.grid.height))
            )

        self.possible_agents = tuple(range(num_agents))
        self.state_space = spaces.Tuple(
            tuple(
                spaces.Tuple(
                    (
                        _coord_space(),
                        spaces.Discrete(len(Direction)),
                        spaces.Discrete(len(Speed)),
                        _coord_space(),  # destination coord
                        spaces.Discrete(2),  # destination reached
                        spaces.Discrete(2),  # crashed
                        # min distance to destination
                        # set this to upper bound of min shortest path distance, so
                        # state space works for generated grids as well
                        spaces.Discrete(self.grid.width * self.grid.height),
                    )
                )
                for _ in range(len(self.possible_agents))
            )
        )
        self.action_spaces = {
            i: spaces.Discrete(len(ACTIONS)) for i in self.possible_agents
        }
        obs_depth = self._obs_front + self._obs_back + 1
        obs_width = (2 * self._obs_side) + 1
        self.observation_spaces = {
            i: spaces.Tuple(
                (
                    spaces.Tuple(
                        tuple(
                            spaces.Discrete(len(CELL_OBS))
                            for _ in range(obs_depth * obs_width)
                        )
                    ),
                    spaces.Discrete(len(Speed)),
                    _coord_space(),  # dest coord,
                    spaces.Discrete(2),  # dest reached
                    spaces.Discrete(2),  # crashed
                )
            )
            for i in self.possible_agents
        }
        self.observation_first = True
        self.is_symmetric = True

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[SupportsFloat, SupportsFloat]]:
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
        chosen_start_coords: Set[Coord] = set()
        chosen_dest_coords: Set[Coord] = set()
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
                coord=start_coord,
                facing_dir=INIT_DIR,
                speed=INIT_SPEED,
                dest_coord=dest_coord,
                dest_reached=int(False),
                crashed=int(False),
                min_dest_dist=dest_dist,
            )
            state.append(state_i)
        return tuple(state)

    def sample_agent_initial_state(self, agent_id: M.AgentID, obs: DObs) -> DState:
        agent_id = int(agent_id)
        possible_agent_start_coords = set()
        agent_dest_coords = obs[2]

        # Need to get start states for agent that are valid given initial obs
        # Need to handle possible start states for other agents
        for all_agent_start_coords in product(
            *[list(s) for s in self.grid.start_coords[: len(self.possible_agents)]]
        ):
            if len(set(all_agent_start_coords)) != len(all_agent_start_coords):
                # skip any sets of start coord that contain duplicates
                continue
            local_obs = self._get_local_cell__obs(
                agent_id, all_agent_start_coords, INIT_DIR, agent_dest_coords
            )
            if local_obs == obs[0]:
                possible_agent_start_coords.add(all_agent_start_coords[agent_id])

        state = []
        chosen_start_coords: Set[Coord] = set()
        chosen_dest_coords: Set[Coord] = set()

        agent_start_coord = self.rng.choice(list(possible_agent_start_coords))
        chosen_start_coords.add(agent_start_coord)
        chosen_dest_coords.add(agent_dest_coords)

        for i in range(len(self.possible_agents)):
            if i == agent_id:
                start_coord = agent_start_coord
            else:
                start_coords_i = self.grid.start_coords[i]
                avail_coords = start_coords_i.difference(chosen_start_coords)
                start_coord = self.rng.choice(list(avail_coords))
                chosen_start_coords.add(start_coord)

            if i == agent_id:
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
                coord=start_coord,
                facing_dir=INIT_DIR,
                speed=INIT_SPEED,
                dest_coord=dest_coord,
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
        assert all(a_i in ACTIONS for a_i in actions.values())
        next_state, collision_types = self._get_next_state(state, actions)
        obs = self._get_obs(next_state)
        rewards = self._get_rewards(state, next_state, collision_types)
        terminated = {
            i: bool(next_state[int(i)].dest_reached or next_state[int(i)].crashed)
            for i in self.possible_agents
        }
        truncated = {i: False for i in self.possible_agents}
        all_done = all(terminated.values())

        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}
        for i in range(len(self.possible_agents)):
            if next_state[i].dest_reached:
                outcome_i = M.Outcome.WIN
            elif next_state[i].crashed:
                outcome_i = M.Outcome.LOSS
            else:
                outcome_i = M.Outcome.NA
            info[i]["outcome"] = outcome_i

        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, info
        )

    def _get_next_state(
        self, state: DState, actions: Dict[M.AgentID, DAction]
    ) -> Tuple[DState, List[CollisionType]]:
        exec_order = list(range(len(self.possible_agents)))
        self.rng.shuffle(exec_order)

        next_state = [vs for vs in state]
        vehicle_coords: Set[Coord] = set([vs.coord for vs in state])
        collision_types = [CollisionType.NONE] * len(self.possible_agents)

        for i in exec_order:
            action_i = actions[i]
            state_i = state[i]
            next_state_i = next_state[i]
            if state_i.dest_reached or next_state_i.crashed:
                # already at destination or crashed, or was crashed into this
                # step
                continue

            vehicle_coords.remove(state_i.coord)

            next_speed = self._get_next_speed(action_i, state_i.speed)
            move_dir = self._get_move_direction(
                action_i, next_speed, state_i.facing_dir
            )
            next_dir = self._get_next_direction(action_i, state_i.facing_dir)
            next_speed = self._get_next_speed(action_i, state_i.speed)
            next_coord, collision_type, hit_vehicle = self._get_next_coord(
                state_i.coord, next_speed, move_dir, vehicle_coords
            )

            min_dest_dist = min(
                state_i.min_dest_dist,
                self.grid.get_shortest_path_distance(next_coord, state_i.dest_coord),
            )

            crashed = False
            if collision_type == CollisionType.VEHICLE:
                # update state of vehicle that was hit
                crashed = True
                collision_types[i] = collision_type
                for j in range(len(self.possible_agents)):
                    next_state_j = next_state[j]
                    if next_state_j.coord == hit_vehicle:
                        collision_types[j] = CollisionType.VEHICLE
                        next_state[j] = VehicleState(
                            coord=next_state_j.coord,
                            facing_dir=next_state_j.facing_dir,
                            speed=next_state_j.speed,
                            dest_coord=next_state_j.dest_coord,
                            dest_reached=next_state_j.dest_reached,
                            crashed=int(True),
                            min_dest_dist=next_state_j.min_dest_dist,
                        )
                        break
            elif collision_type == CollisionType.OBSTACLE:
                crashed = self._obstacle_collisions
                collision_types[i] = collision_type

            next_state[i] = VehicleState(
                coord=next_coord,
                facing_dir=next_dir,
                speed=next_speed,
                dest_coord=state_i.dest_coord,
                dest_reached=int(next_coord == state_i.dest_coord),
                crashed=int(crashed),
                min_dest_dist=min_dest_dist,
            )

            vehicle_coords.add(next_coord)

        return tuple(next_state), collision_types

    def _reset_vehicle(
        self, v_idx: int, vs_i: VehicleState, vehicle_coords: Set[Coord]
    ) -> VehicleState:
        if not (vs_i.dest_reached or vs_i.crashed):
            return vs_i

        start_coords_i = self.grid.start_coords[v_idx]
        avail_start_coords = start_coords_i.difference(vehicle_coords)

        if vs_i.coord in start_coords_i:
            # add it back in since it will be remove during difference op
            avail_start_coords.add(vs_i.coord)

        new_coord = self.rng.choice(list(avail_start_coords))

        min_dest_dist = self.grid.manhattan_dist(new_coord, vs_i.dest_coord)

        new_vs_i = VehicleState(
            coord=new_coord,
            facing_dir=INIT_DIR,
            speed=INIT_SPEED,
            dest_coord=vs_i.dest_coord,
            dest_reached=int(False),
            crashed=int(False),
            min_dest_dist=min_dest_dist,
        )
        return new_vs_i

    def _get_move_direction(
        self, action: DAction, speed: Speed, curr_dir: Direction
    ) -> Direction:
        if speed == Speed.REVERSE:
            # No turning while in reverse,
            # so movement dir is just the opposite of current direction
            return Direction((curr_dir + 2) % len(Direction))
        return self._get_next_direction(action, curr_dir)

    @staticmethod
    def _get_next_direction(action: DAction, curr_dir: Direction) -> Direction:
        if action == TURN_RIGHT:
            return Direction((curr_dir + 1) % len(Direction))
        if action == TURN_LEFT:
            return Direction((curr_dir - 1) % len(Direction))
        return curr_dir

    @staticmethod
    def _get_next_speed(action: DAction, curr_speed: Speed) -> Speed:
        if action == DO_NOTHING:
            return curr_speed
        if action in (TURN_LEFT, TURN_RIGHT):
            if curr_speed == Speed.FORWARD_FAST:
                return Speed.FORWARD_SLOW
            return curr_speed
        if action == ACCELERATE:
            return Speed(min(curr_speed + 1, Speed.FORWARD_FAST))
        return Speed(max(curr_speed - 1, Speed.REVERSE))

    def _get_next_coord(
        self,
        curr_coord: Coord,
        speed: Speed,
        move_dir: Direction,
        vehicle_coords: Set[Coord],
    ) -> Tuple[Coord, CollisionType, Optional[Coord]]:
        # assumes curr_coord isn't in vehicle coords
        next_coord = curr_coord
        collision = CollisionType.NONE
        hit_vehicle_coord = None
        for i in range(abs(speed - Speed.STOPPED)):
            next_coord = self.grid.get_next_coord(
                curr_coord, move_dir, ignore_blocks=False
            )
            if next_coord == curr_coord:
                collision = CollisionType.OBSTACLE
                break
            elif next_coord in vehicle_coords:
                collision = CollisionType.VEHICLE
                hit_vehicle_coord = next_coord
                next_coord = curr_coord
                break
            curr_coord = next_coord

        return (next_coord, collision, hit_vehicle_coord)

    def _get_obs(self, state: DState) -> Dict[M.AgentID, DObs]:
        obs: Dict[M.AgentID, DObs] = {}
        for i in range(len(self.possible_agents)):
            local_cell_obs = self._get_local_cell__obs(
                i, [vs.coord for vs in state], state[i].facing_dir, state[i].dest_coord
            )
            obs[i] = (
                local_cell_obs,
                state[i].speed,
                state[i].dest_coord,
                state[i].dest_reached,
                state[i].crashed,
            )
        return obs

    def _get_local_cell__obs(
        self,
        agent_id: int,
        vehicle_coords: Sequence[Coord],
        facing_dir: Direction,
        dest_coord: Coord,
    ) -> Tuple[int, ...]:
        obs_depth = self._obs_front + self._obs_back + 1
        obs_width = (2 * self._obs_side) + 1
        agent_coord = vehicle_coords[agent_id]

        cell_obs = []
        for col, row in product(range(obs_width), range(obs_depth)):
            obs_grid_coord = self._map_obs_to_grid_coord(
                (col, row), agent_coord, facing_dir
            )
            if obs_grid_coord is None or obs_grid_coord in self.grid.block_coords:
                cell_obs.append(WALL)
            elif obs_grid_coord in vehicle_coords:
                cell_obs.append(VEHICLE)
            elif obs_grid_coord == dest_coord:
                cell_obs.append(DESTINATION)
            else:
                cell_obs.append(EMPTY)
        return tuple(cell_obs)

    def _map_obs_to_grid_coord(
        self, obs_coord: Coord, agent_coord: Coord, facing_dir: Direction
    ) -> Optional[Coord]:
        if facing_dir == Direction.NORTH:
            grid_row = agent_coord[1] + obs_coord[1] - self._obs_front
            grid_col = agent_coord[0] + obs_coord[0] - self._obs_side
        elif facing_dir == Direction.EAST:
            grid_row = agent_coord[1] + obs_coord[0] - self._obs_side
            grid_col = agent_coord[0] - obs_coord[1] + self._obs_front
        elif facing_dir == Direction.SOUTH:
            grid_row = agent_coord[1] - obs_coord[1] + self._obs_front
            grid_col = agent_coord[0] - obs_coord[0] + self._obs_side
        else:
            grid_row = agent_coord[1] - obs_coord[0] + self._obs_side
            grid_col = agent_coord[0] + obs_coord[1] - self._obs_front

        if 0 <= grid_row < self.grid.height and 0 <= grid_col < self.grid.width:
            return (grid_col, grid_row)
        return None

    def get_obs_coords(self, origin: Coord, facing_dir: Direction) -> List[Coord]:
        """Get the list of coords observed from agent at origin."""
        obs_depth = self._obs_front + self._obs_back + 1
        obs_width = (2 * self._obs_side) + 1
        obs_coords: List[Coord] = []
        for col, row in product(range(obs_width), range(obs_depth)):
            obs_grid_coord = self._map_obs_to_grid_coord((col, row), origin, facing_dir)
            if obs_grid_coord is not None:
                obs_coords.append(obs_grid_coord)
        return obs_coords

    def _get_rewards(
        self, state: DState, next_state: DState, collision_types: List[CollisionType]
    ) -> Dict[M.AgentID, SupportsFloat]:
        rewards: Dict[M.AgentID, SupportsFloat] = {}
        for i in range(len(self.possible_agents)):
            if state[i].crashed or state[i].dest_reached:
                # already in terminal/rewarded state
                r_i = 0.0
            elif (
                self._obstacle_collisions
                and collision_types[i] == CollisionType.OBSTACLE
            ):
                # Treat as if crashed into a vehicle
                r_i = self.R_CRASH_VEHICLE
            elif collision_types[i] == CollisionType.VEHICLE:
                r_i = self.R_CRASH_VEHICLE
            elif next_state[i].dest_reached:
                r_i = self.R_DESTINATION_REACHED
            else:
                r_i = self.R_STEP_COST

            progress = state[i].min_dest_dist - next_state[i].min_dest_dist
            r_i += max(0, progress) * self.R_PROGRESS

            if (
                not self._obstacle_collisions
                and collision_types[i] == CollisionType.OBSTACLE
            ):
                r_i += self.R_CRASH_OBJECT
            rewards[i] = r_i
        return rewards


class DrivingGrid(Grid):
    """A grid for the Driving Problem."""

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        block_coords: Set[Coord],
        start_coords: List[Set[Coord]],
        dest_coords: List[Set[Coord]],
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

    def get_shortest_path_distance(self, coord: Coord, dest: Coord) -> int:
        """Get the shortest path distance from coord to destination."""
        return int(self.shortest_paths[dest][coord])

    def get_max_shortest_path_distance(self) -> int:
        """Get the longest shortest path distance to any destination."""
        return int(max([max(d.values()) for d in self.shortest_paths.values()]))

    def get_ascii_repr(
        self,
        vehicle_coords: List[Coord],
        vehicle_dirs: List[Direction],
        vehicle_dests: List[Coord],
    ) -> str:
        """Get ascii repr of grid."""
        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                coord = (col, row)
                if coord in self.block_coords:
                    row_repr.append("#")
                elif coord in vehicle_dests:
                    row_repr.append("D")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)

        for coord, direction in zip(vehicle_coords, vehicle_dirs):
            grid_repr[coord[0]][coord[1]] = DIRECTION_ASCII_REPR[direction]

        return "\n".join(list(list((" ".join(r) for r in grid_repr))))


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

    Examples (" " quotes and newline chars ommited):

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
    agent_dest_chars = set(["-"] + [c for c in "abcdefghij"])

    block_coords: Set[Coord] = set()
    shared_start_coords: Set[Coord] = set()
    agent_start_coords_map: Dict[int, Set[Coord]] = {}
    shared_dest_coords: Set[Coord] = set()
    agent_dest_coords_map: Dict[int, Set[Coord]] = {}
    for r, c in product(range(grid_height), range(grid_width)):
        coord = (c, r)
        char = row_strs[r][c]

        if char == "#":
            block_coords.add(coord)
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

    included_agent_ids = list(set([*agent_start_coords_map, *agent_dest_coords_map]))
    if len(included_agent_ids) > 0:
        assert max(included_agent_ids) < supported_num_agents

    start_coords: List[Set[Coord]] = []
    dest_coords: List[Set[Coord]] = []
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


def get_3x3_grid() -> DrivingGrid:
    """Generate a simple Driving 3-by-3 grid layout."""
    grid_str = "a1.\n" ".#.\n" ".0b\n"
    return parse_grid_str(grid_str, 2)


def get_4x4_intersection_grid() -> DrivingGrid:
    """Generate a 4-by-4 intersection grid layout."""
    grid_str = "#0b#\n" "d..3\n" "2..c\n" "#a1#\n"
    return parse_grid_str(grid_str, 4)


def get_6x6_intersection_grid() -> DrivingGrid:
    """Generate a 6-by-6 intersection grid layout."""
    grid_str = "##0b##\n" "##..##\n" "d....3\n" "2....c\n" "##..##\n" "##a1##\n"
    return parse_grid_str(grid_str, 4)


def get_7x7_crisscross_grid() -> DrivingGrid:
    """Generate a 7-by-7 Criss-Cross grid layout."""
    grid_str = (
        "#-#-#-#\n"
        "-.....+\n"
        "#.#.#.#\n"
        "-.....+\n"
        "#.#.#.#\n"
        "-.....+\n"
        "#+#+#+#\n"
    )
    return parse_grid_str(grid_str, 6)


def get_7x7_blocks_grid() -> DrivingGrid:
    """Generate a 7-by-7 blocks grid layout."""
    grid_str = (
        "#-...-#\n"
        "-##.##+\n"
        ".##.##.\n"
        ".......\n"
        ".##.##.\n"
        "-##.##+\n"
        "#+...+#\n"
    )
    return parse_grid_str(grid_str, 4)


def get_7x7_roundabout_grid() -> DrivingGrid:
    """Generate a 7-by-7 round-about grid layout."""
    grid_str = (
        "#-...-#\n"
        "-##.##+\n"
        ".#...#.\n"
        "...#...\n"
        ".#...#.\n"
        "-##.##+\n"
        "#+...+#\n"
    )
    return parse_grid_str(grid_str, 4)


def get_14x14_crisscross_grid() -> DrivingGrid:
    """Generate a 14-by-14 Criss-Cross grid layout."""
    grid_str = (
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
    )
    return parse_grid_str(grid_str, 4)


def get_14x14_blocks_grid() -> DrivingGrid:
    """Generate a 14-by-14 Blocks grid layout."""
    grid_str = (
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
    )
    return parse_grid_str(grid_str, 4)


def get_14x14_roundabout_wide_grid() -> DrivingGrid:
    """Generate a 14-by-14 Round About grid layout with wide roads."""
    grid_str = (
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
    )
    return parse_grid_str(grid_str, 4)


class DrivingGridGenerator(GridGenerator):
    """Class for generating grid layouts for Driving Environment.

    Generates starting and destination coords in an alternating pattern along
    the outside of edge of the grid.
    """

    def __init__(
        self,
        width: int,
        height: int,
        max_obstacle_size: int,
        max_num_obstacles: int,
        seed: Optional[int] = None,
    ):
        super().__init__(
            width,
            height,
            self._generate_mask(width, height),
            max_obstacle_size,
            max_num_obstacles,
            ensure_grid_connected=True,
            seed=seed,
        )

        self._start_coords = [self.mask for _ in range(len(self.mask))]
        self._dest_coords = [self.mask for _ in range(len(self.mask))]

    def _generate_mask(self, width: int, height: int) -> Set[Coord]:
        start = 1
        mask = set()
        for x in range(start, width, 2):
            mask.add((x, 0))
            mask.add((x, height - 1))

        for y in range(start, height, 2):
            mask.add((0, y))
            mask.add((width - 1, y))

        return mask

    def generate(self) -> DrivingGrid:
        """Generate a new Driving grid."""
        base_grid = super().generate()
        driving_grid = DrivingGrid(
            base_grid.width,
            base_grid.height,
            base_grid.block_coords,
            self._start_coords,
            self._dest_coords,
        )
        return driving_grid


# (generator params, finite horizon step limit)
SUPPORTED_GEN_PARAMS = {
    "7x7": (
        {
            "width": 7,
            "height": 7,
            "max_obstacle_size": 2,
            "max_num_obstacles": 21,  # size * 3
        },
        40,
    ),
    "14x14": (
        {
            "width": 14,
            "height": 14,
            "max_obstacle_size": 3,
            "max_num_obstacles": 42,  # size * 3
        },
        100,
    ),
    "28x28": (
        {
            "width": 28,
            "height": 28,
            "max_obstacle_size": 4,
            "max_num_obstacles": 84,  # size * 3
        },
        200,
    ),
}


#  (grid_make_fn, max step_limit, )
SUPPORTED_GRIDS = {
    "3x3": (get_3x3_grid, 30),
    "4x4Intersection": (get_4x4_intersection_grid, 20),
    "6x6Intersection": (get_6x6_intersection_grid, 20),
    "7x7CrissCross": (get_7x7_crisscross_grid, 50),
    "7x7Blocks": (get_7x7_blocks_grid, 50),
    "7x7RoundAbout": (get_7x7_roundabout_grid, 50),
    "14x14Blocks": (get_14x14_blocks_grid, 100),
    "14x14CrissCross": (get_14x14_crisscross_grid, 100),
    "14x14WideRoundAbout": (get_14x14_roundabout_wide_grid, 50),
}


def load_grid(grid_name: str) -> DrivingGrid:
    """Load grid with given name."""
    grid_name = grid_name
    assert grid_name in SUPPORTED_GRIDS, (
        f"Unsupported grid name '{grid_name}'. Grid name must be one of: "
        f"{SUPPORTED_GRIDS.keys()}."
    )
    return SUPPORTED_GRIDS[grid_name][0]()
