"""The Drone Team Capture Environment.

A co-operative 2D circular world problem involving multiple agents working together
to catch a target agent in the environment.

This is an adaption of the original code base to allow for partially observable
environments.

Reference
---------
- C. de Souza, R. Newbury, A. Cosgun, P. Castillo, B. Vidolov and D. Kulić,
  "Decentralized Multi-Agent Pursuit Using Deep Reinforcement Learning,"
  in IEEE Robotics and Automation Letters, vol. 6, no. 3, pp. 4552-4559,
  July 2021, doi: 10.1109/LRA.2021.3068952.

"""
import math
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import numpy as np
from gymnasium import spaces

import posggym.model as M
from posggym.core import DefaultEnv
from posggym.envs.continuous.core import CircularContinuousWorld, Position, clip_actions
from posggym.utils import seeding


class DTCState(NamedTuple):
    """A state in the Drone Team Capture Environment."""

    pursuer_coords: Tuple[Position, ...]
    prev_pursuer_coords: Tuple[Position, ...]
    target_coords: Position
    prev_target_coords: Position
    target_vel: float


# Actions
DTCAction = List[float]
DTCObs = np.ndarray


class DroneTeamCaptureEnv(DefaultEnv[DTCState, DTCObs, DTCAction]):
    """The Drone Team Capture Environment.

    A co-operative 2D continuous world problem involving multiple pursuer drone agents
    working together to catch a target agent in the environment.

    Possible Agents
    ---------------
    Varied number (1-8)

    State Space
    -----------
    Each state consists of:

    1. tuple of the (x, y) position of all pursuers
    2. tuple of the (x, y) previous position of all pursuers
    3. tuple of the (x, y) position of the target
    4. tuple of the (x, y) previous position of the target
    5. The speed of the target relative to the pursuer (0.0 -> 2.0)

    For the coordinate x=column, y=row, with the origin (0, 0) at the
    top-left square of the world.

    Action Space
    ------------
    Each agent has either 1 or 2 actions. If 'velocity_control=False' then the agent
    can only control their angular velocity. If 'velocity_control=True' then the agent
    has two actions, which are the angular and linear velocity, in that order.

    Observation Space
    -----------------
    Each agent observes the other pursuers and target. For each pursuer they will
    observe the relative position and angle of the pursuer. They will observe
    their own heading and change in heading. They will also observe the distance
    to the target, and the angle to the target, as well as, their derivatives.
    The pursuers will be ordered by the relative distance to the target.

    However, there are two parameters to change this behaviour. Firstly,
    `observation_limit` which specifies the maximum distance another agent can be seen.
    If they are outside this radius, the observations of these agents will be `-1`.

    There is also the `n_communicating_purusers` which will limit the maximum amount of
    other pursuers an agent can observe.

    Rewards
    -------
    Each pursuer will receive a reward based on the Q parameter and the distance from
    the target. On successful capture, the capturing pursuer will receive a reward of
    `+130`, while other agents will receive `100`.

    Dynamics
    --------
    Actions of the puruser agents are deterministic and consist of moving based on the
    non-holonomic model.

    The target will move following a holonomic model TODO

    Starting State
    --------------
    Target will start near the outside of the circle, while pursuers will start in a
    line on the middle. The pursuers will start with a random yaw.

    Episodes End
    ------------
    Episodes ends when the target has been captured. By default a `max_episode_steps`
    limit of `50` steps is also set. This may need to be adjusted when using larger
    grids (this can be done by manually specifying a value for `max_episode_steps`when
    creating the environment with `posggym.make`).

    Arguments
    ---------

    - `num_agents` - The number of agents which exist in the environment
        Must be between 1 and 8 (default = `3`)
    - `n_communicating_puruser - The maximum number of agents which an
        agent can receive information from (default = `3`)
    - `velocity_control` - If the agents have control of their linear velocity
        (default = `False`)
    - `arena_size` - Size of the arena (default = `430`)
    - `observation_limit` - The limit of which agents can see other agents
        (default = `430`)
    - `use_curriculum` - If curriculum learning is used, a large capture radius is used
        the capture radius needs to be decreased using the 'decrease_cap_rad' function
        (default = `False`)

    Available variants
    ------------------

    For example to use the Drone Team Capture environment with 8 communicating pursuers
    and episode step limit of 100, and the default values for the other parameters
    (`velocity_control`, `arena_size`, `observation_limit`, `use_curriculum`) you would
    use:

    ```python
    import posggym
    env = posggym.make(
        'DroneTeamCapture-v0',
        max_episode_steps=100,
        num_agents=8,
        n_communicating_puruser=4,
    )
    ```

    Version History
    ---------------
    - `v0`: Initial version

    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 4,
    }

    def __init__(
        self,
        num_agents: int = 3,
        n_communicating_purusers: int = 3,
        arena_size: float = 430,
        observation_limit: float = 430,
        velocity_control: bool = False,
        use_curriculum: bool = False,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            DTCModel(
                num_agents,
                n_communicating_purusers,
                velocity_control,
                use_curriculum=use_curriculum,
                arena_size=arena_size,
                observation_limit=observation_limit,
                **kwargs,
            ),
            render_mode=render_mode,
        )

        self._viewer = None
        self._renderer = None
        self.render_mode = render_mode

    def render(self):
        if self.render_mode == "human":
            import posggym.envs.continuous.render as render_lib

            if self._renderer is None:
                self._renderer = render_lib.GWContinuousRender(
                    render_mode=self.render_mode,
                    domain_max=(self.model.r_arena),
                    domain_min=(-self.model.r_arena),
                    arena_size=200,
                    num_colors=3,
                    render_fps=self.metadata["render_fps"],
                    arena_type=render_lib.ArenaTypes.Circle,
                    env_name="Multi-agent pursuit evasion",
                )

            colored_pred = tuple(t + (0,) for t in self._state.pursuer_coords)
            colored_target = (self._state.target_coords + (1,),)

            is_holomic = [True] + ([False] * len(self._state.pursuer_coords))
            size = [self.model.cap_rad] + [None] * len(self._state.pursuer_coords)

            self._renderer.clear_render()
            self._renderer.draw_arena()
            self._renderer.draw_agents(colored_target + colored_pred, is_holomic, size)
            self._renderer.render()


class DTCModel(M.POSGModel[DTCState, DTCObs, DTCAction]):
    """The Drone Team Capture problem model.

    Parameters
    ----------
    num_agents: int,
        The number of agents which exist in the environment
        Must be between 1 and 8
    n_communicating_purusers: int
        The maximum number of agents which an agent can receive information from
    velocity_control: bool
        If the agents have control of their linear velocity
    arena_size : float
        Size of the arena
    observation_limit : float
        The limit of which agents can see other agents
    use_curriculum : bool
        If curriculum learning is used, a large capture radius is used
        The capture radius needs to be decreased using the 'decrease_cap_rad' function
    """

    R_MAX = 130

    def __init__(
        self,
        num_agents: int,
        n_communicating_purusers: int,
        velocity_control: bool = False,
        arena_size: float = 430,
        observation_limit: float = 430,
        use_curriculum=False,
        **kwargs,
    ):
        # The original paper used between 1 and 8 agents
        assert 1 < num_agents <= 8

        self.n_pursuers = num_agents

        # Linear Velocity of pursuer
        self.vel_pur = 10
        # Max linear velocity of target
        self.max_vel_tar = 2.0

        # The angular velocity of the pursuer / target
        self.omega_max_pur = math.pi / 10
        self.omega_max_tar = math.pi / 10

        # Agent can see distance + angle to other pursuers
        self.obs_each_pursuer_pursuer = 2

        # Agent can see more detailed info about target + self
        self.obs_self_pursuer = 6

        # Number of communicating pursuers in environment
        self.n_pursuers_com = n_communicating_purusers

        # TODO: remove this
        self.arena_dim_square = 1000

        # If the agents have constant or variable velocity
        self.velocity_control = velocity_control

        # Size of the world
        self.r_arena = arena_size
        # How larage of radius can agents communicate in
        self.observation_limit = observation_limit

        OBS_DIM = (
            self.obs_self_pursuer
            + (self.n_pursuers_com - 1) * self.obs_each_pursuer_pursuer
        )

        # The capture radius needs to be manually decreased if using curriculum
        self.cap_rad = 60 if use_curriculum else 30

        high = np.ones(OBS_DIM, dtype=np.float32) * np.inf

        # act[0] = angular velocity, act[1] = linear velocity
        acthigh = (
            np.array([1], dtype=np.float32)
            if not self.velocity_control
            else np.array([1, 1], dtype=np.float32)
        )
        actlow = (
            np.array([-1], dtype=np.float32)
            if not self.velocity_control
            else np.array([-1, 0], dtype=np.float32)
        )

        self.possible_agents = tuple(str(x) for x in range(self.n_pursuers))

        self.observation_spaces = {
            str(i): spaces.Box(-high, high, dtype=np.float32)
            for i in self.possible_agents
        }

        self.action_spaces = {
            str(i): spaces.Box(low=actlow, high=acthigh) for i in self.possible_agents
        }

        self.grid = CircularContinuousWorld(radius=self.r_arena, block_coords=None)

        # Not sure what this means??
        self.is_symmetric = True

    def get_agents(self, state: DTCState) -> List[M.AgentID]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> DTCState:
        pursuer_coords = []
        prev_pursuer_coords = []
        for i in range(self.n_pursuers):
            # distributes the agents based in their number
            x = 50.0 * (-math.floor(self.n_pursuers / 2) + i)
            pursuer_coords.append((x, 0.0, 0.0))
            prev_pursuer_coords.append((x, 0.0, 0.0))

        bounds = self.r_arena * 0.7
        x = self.rng.random() * (bounds * 2) - bounds
        y = self.rng.random() * (bounds * 2) - bounds

        # The velocity of the target varies
        relative_target_vel = float(self.rng.random() * self.max_vel_tar)

        target_coords = (x, y, 0.0)
        prev_target_coords = (x, y, 0.0)

        return DTCState(
            tuple(pursuer_coords),
            tuple(prev_pursuer_coords),
            target_coords,
            prev_target_coords,
            relative_target_vel,
        )

    def sample_initial_obs(self, state: DTCState) -> Dict[M.AgentID, DTCObs]:
        return self._get_obs(state)

    def step(
        self, state: DTCState, actions: Dict[M.AgentID, DTCAction]
    ) -> M.JointTimestep[DTCState, DTCObs]:
        clipped_actions = clip_actions(actions, self.action_spaces)

        next_state = self._get_next_state(state, clipped_actions)
        obs = self._get_obs(state)
        done, rewards = self._get_rewards(state)

        dones = {}

        for i in self.possible_agents:
            dones[i] = done

        truncated = {i: False for i in self.possible_agents}

        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}

        return M.JointTimestep(next_state, obs, rewards, dones, truncated, done, info)

    def abs_sum(self, vector: List[float]) -> float:
        return sum([abs(x) for x in vector])

    def decrease_cap_rad(self):
        self.cap_rad -= 5

    def check_capture(self, target_coords: Position, pursuer_coord: Position) -> bool:
        # Check distance between pursuer and target
        dist = CircularContinuousWorld.euclidean_dist(target_coords, pursuer_coord)
        return dist < self.cap_rad

    def target_distance(self, state: DTCState, index: int) -> float:
        return CircularContinuousWorld.euclidean_dist(
            state.pursuer_coords[index], state.target_coords
        )

    def _get_next_state(
        self, state: DTCState, actions: Dict[M.AgentID, List[float]]
    ) -> DTCState:
        prev_target = state.target_coords
        prev_pursuer = state.pursuer_coords

        new_target_coords = self.target_move_repulsive(state.target_coords, state)
        new_pursuer_coords = []
        for i, pred_pos in enumerate(state.pursuer_coords):
            # Force velocity to be 1, if only controlling angular speed
            velocity_factor = 1 if not self.velocity_control else actions[str(i)][1]

            new_coords, _ = self.grid._get_next_coord(
                pred_pos,
                [actions[str(i)][0], self.vel_pur * velocity_factor],
                True,
                use_holonomic_model=False,
            )

            new_pursuer_coords.append(new_coords)

        return DTCState(
            tuple(new_pursuer_coords),
            prev_pursuer,
            new_target_coords,
            prev_target,
            state.target_vel,
        )

    def _get_obs(self, state: DTCState) -> Dict[M.AgentID, DTCObs]:
        # Get observation for all pursuers
        observation = {}

        for i in range(self.n_pursuers):
            """getting the target engagement"""
            (alpha_t, _, dist_t, _, _), target_suc = self.engagmment(
                state.pursuer_coords[i],
                state.target_coords,
                dist_factor=self.arena_dim_square,
            )
            (alpha_t_prev, _, dist_t_prev, _, _), target_prev_suc = self.engagmment(
                state.prev_pursuer_coords[i],
                state.prev_target_coords,
                dist_factor=self.arena_dim_square,
            )

            engagement = []

            """ getting the relative engagement """
            for j in range(self.n_pursuers):
                if j != i:
                    eng, _ = self.engagmment(
                        state.pursuer_coords[i],
                        state.pursuer_coords[j],
                        dist_factor=self.arena_dim_square,
                    )
                    engagement.append(eng)

            # Put any invalid (-1) to the end
            def key_func(t):
                if t[1] == -1:
                    return float("inf")
                else:
                    return t[1]

            engagement = sorted(engagement, key=key_func)
            alphas, _, dists, _, _ = list(zip(*engagement))

            # change in alpha
            if not target_suc or not target_prev_suc:
                alpha_rate = -1.0
                dist_rate = -1.0
                turn_rate = -1.0
            else:
                alpha_rate = alpha_t - alpha_t_prev
                alpha_rate = math.asin(math.sin(alpha_rate)) / 4
                dist_rate = (dist_t - dist_t_prev) / 0.03

            north = -state.prev_pursuer_coords[i][2] / math.pi
            north_prev = -state.prev_pursuer_coords[i][2] / math.pi
            turn_rate = (north - north_prev) / 2

            # normalise the alpha
            if target_suc:
                alpha_t = alpha_t / math.pi

            temp_observation = [
                alpha_t,
                dist_t,
                north,
                alpha_rate,
                dist_rate,
                turn_rate,
            ]

            # alpha and distance to each pursuer
            for index in range(self.n_pursuers_com - 1):
                temp_observation.append(alphas[index] / math.pi)
                temp_observation.append(dists[index])

            observation[str(i)] = np.array(temp_observation, dtype=np.float32)

        return observation

    def _get_rewards(self, state: DTCState) -> Tuple[bool, Dict[M.AgentID, float]]:
        done = False

        reward: Dict[M.AgentID, float] = {}

        Q_formation = self.q_parameter(state)

        for id in range(len(state.pursuer_coords)):
            # Reward function definition
            reward[str(id)] = 0.0
            reward[str(id)] -= Q_formation * 0.1
            reward[str(id)] -= 0.002 * self.target_distance(state, id)

            if self.check_capture(state.target_coords, state.pursuer_coords[id]):
                done = True
                reward[str(id)] += 30.0  # 30% more than the others

        if done:
            # Large possible reward when done!
            for index in self.possible_agents:
                reward[index] += 100.0

        return done, reward

    def engagmment(
        self, agent_i: Position, agent_j: Position, dist_factor: float
    ) -> Tuple[Tuple[float, ...], bool]:
        dist = CircularContinuousWorld.euclidean_dist(agent_i, agent_j)

        yaw = agent_i[2]

        # Rotation matrix of yaw
        R = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])

        T_p = np.array([agent_i[0] - agent_j[0], agent_i[1] - agent_j[1]])
        T_p = R.dot(T_p)
        alpha = math.atan2(T_p[1], T_p[0])

        Los_angle = math.atan2(T_p[1], T_p[0])

        if dist > self.observation_limit:
            return tuple([-1] * 5), False

        return (
            alpha,
            Los_angle,
            dist / dist_factor,
            T_p[0] / dist_factor,
            T_p[1] / dist_factor,
        ), True

    def get_closest(self, state: DTCState) -> int:
        # Find closest agent to pursuer
        min_dist, min_index = None, None
        for idx, p in enumerate(state.pursuer_coords):
            dist = self.target_distance(state, idx)

            if min_dist is None or dist < min_dist:
                min_index = idx
                min_dist = dist

        if min_dist is None or min_index is None:
            raise Exception("No closest index found. Something has gone wrong.")

        return min_index

    def get_unit_vectors(self, state: DTCState) -> List[List[float]]:
        # Find unit vectors between target and pursuers
        unit = []
        for p in state.pursuer_coords:
            dist = self.grid.euclidean_dist(p, state.target_coords)
            unit.append([p[0] / dist, p[1] / dist])
        return unit

    def q_parameter(self, state: DTCState) -> float:
        # Q Parameter definition
        closest = self.get_closest(state)
        unit = self.get_unit_vectors(state)
        Qk = 0.0

        for i in range(self.n_pursuers):
            if i != closest:
                Qk = Qk + np.dot(unit[i], unit[closest]) + 1.0
        Qk = Qk / len(unit)

        return Qk

    def scale_vector(self, vector, scale, final_vector, factor=1.0) -> List[float]:
        div = self.abs_sum(vector)
        if div < 0.00001:
            div = 0.00001
        f = -factor * scale(self.abs_sum(vector)) / div
        vector = f * vector
        return [x + y for x, y in zip(final_vector, vector)]

    def target_move_repulsive(self, position: Position, state: DTCState) -> Position:
        # Target behaviour
        xy_pos = np.array(position[:2])
        x, y = xy_pos

        def scale(x):
            return 50000 / (abs(x) + 200) ** 2

        n_pursuer = len(state.pursuer_coords)
        final_vector = [0.0, 0.0]

        for i in state.pursuer_coords:
            vector = np.array(i[:2]) - xy_pos
            final_vector = self.scale_vector(vector, scale, final_vector)

        # Find closest point !!!! then put it in to the vectorial sum
        r_wall = self.r_arena
        gamma = math.atan2(y, x)

        virtual_wall = [r_wall * math.cos(gamma), r_wall * math.sin(gamma)]
        vector = virtual_wall - xy_pos
        final_vector = self.scale_vector(vector, scale, final_vector, 0.5 * n_pursuer)

        self.enemy_speed = 1
        scaled_move_dir = [
            self.enemy_speed * i / self.abs_sum(final_vector) for i in final_vector
        ]

        dx, dy = scaled_move_dir[0], scaled_move_dir[1]
        d = np.linalg.norm([dx, dy])
        x += (state.target_vel * self.vel_pur) * (dx / d)
        y += (state.target_vel * self.vel_pur) * (dy / d)

        return x, y, 0

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[float, float]]:
        return {i: (0.0, self.R_MAX) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng
