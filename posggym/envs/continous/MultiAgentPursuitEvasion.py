"""The Predator-Prey Grid World Environment.

A co-operative 2D grid world problem involving multiple predator agents working together
to catch prey agents in the environment.

Reference
---------
- Ming Tan. 1993. Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents
  In Proceedings of the Tenth International Conference on Machine Learning. 330–337.
- J. Z. Leibo, V. F. Zambaldi, M. Lanctot, J. Marecki, and T. Graepel. 2017. Multi-Agent
  Reinforcement Learning in Sequential Social Dilemmas. In AAMAS, Vol. 16. ACM, 464–473

"""
import math
from itertools import product
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    SupportsFloat,
    Tuple,
)
import numpy as np

from gymnasium import spaces

import posggym.envs.continous.render as render_lib
import posggym.model as M
from posggym.core import DefaultEnv
from posggym.envs.continous.core import CircularContinousWorld, Object, Position
from posggym.utils import seeding
import math
class PPState(NamedTuple):
    """A state in the Predator-Prey Environment."""
    predator_coords: Tuple[Position, ...]
    prev_predator_coords: Tuple[Position, ...]
    prey_coords: Position
    prev_prey_coords : Position
    pursuer_vel: float


# Actions
PPAction = float

PREY = 0
PREDATOR = 1
AGENT_TYPE = [PREDATOR, PREY] 


# Observations
# Obs = (adj_obs)
PPObs = Tuple[float, ...]
# Cell Obs

collision_distance = 1

class MAPEEnv(DefaultEnv[PPState, PPObs, PPAction]):


    metadata = {
        "render_modes": ["human"],
        "render_fps": 4,
    }

    def __init__(
        self,
        num_agents: int,
        n_communicating_purusers : int,
        velocity_control : bool = False,
        use_curriculum : bool = False,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            PPModel(
                num_agents,
                n_communicating_purusers,
                velocity_control,
                **kwargs,
            ),
            render_mode=render_mode,
        )
        # self._obs_dim = obs_dim
        self._viewer = None
        self._renderer: Optional[render_lib.GWContinousRender] = None
        self.render_mode = render_mode

    def render(self):
        if self.render_mode == "human":
            import posggym.envs.continous.render as render_lib

            
            if self._renderer is None:
                self._renderer = render_lib.GWContinousRender(
                    render_mode=self.render_mode,
                    domain_max=(self.model.r_arena),
                    domain_min=(-self.model.r_arena),
                    arena_size=200,
                    num_colors=3,
                    arena_type=render_lib.ArenaTypes.Circle,
                    env_name="Multi-agent pursuit evasion",
                )
            colored_pred =  tuple(t + (0,) for t in self._state.predator_coords)
            colored_prey =  tuple(((self._state.prey_coords + (1,),)))

            is_holomic = [True] + ([False] * len(self._state.predator_coords))
            size = [self.model.cap_rad] + [None] * len(self._state.predator_coords)
        
            self._renderer.render(colored_prey + colored_pred, is_holomic, size)



   
class PPModel(M.POSGModel[PPState, PPObs, PPAction]):
    """Predator-Prey Problem Model.

    Parameters
    ----------
    size : int
        the size of the grid (height and width)
    num_predators : int
        the number of predator (and thus controlled agents)
    num_prey : int
        the number of prey
    cooperative : bool
        whether environment rewards are fully shared (True) or only awarded to
        capturing predators (i.e. mixed) (False)
    prey_strenth : int
        the minimum number of predators needed to capture a prey
    obs_dims : int
        number of cells in each direction around the agent that the agent can
        observe

    """

    R_MAX = 1.0
    PREY_CAUGHT_COORD = (0, 0)

    def __init__(
        self,
        # grid: Union[str, "PPGrid"],
        num_agents: int,
        n_communicating_purusers : int,
        velocity_control : bool = False,
        **kwargs,
    ):
        assert 1 < num_agents <= 8


        self.n_pursuers = num_agents

        self.vel_pur = 10
        self.vel_tar = 20
        self.omega_max_pur = math.pi / 10
        self.omega_max_tar = math.pi / 10


        self.obs_each_pursuer_evader = 2
        self.obs_self_evader = 3

        self.obs_each_pursuer_pursuer = 2
        self.obs_self_pursuer = 6

        self.obs_self = self.obs_self_pursuer
        self.obs_pursuer = self.obs_each_pursuer_pursuer

        self.velocity_control = False
        self.n_pursuers_com = n_communicating_purusers
        self.arena_dim_square = 1000

        self.velocity_control = velocity_control

        self.r_arena = 430
        self.observation_limit = 300
        

        ACT_DIM = 2 if velocity_control else 1

        OBS_DIM = (
            self.obs_self_pursuer
            + (self.n_pursuers_com - 1) * self.obs_each_pursuer_pursuer
        )

        self.cap_rad = 60

        high = np.ones((self.n_pursuers, OBS_DIM), dtype=np.float64) * np.inf
        # Second number is velocity
        acthigh = np.array([1]) if not self.velocity_control else np.array([1, 1])
        actlow = np.array([-1]) if not self.velocity_control else np.array([-1, 0])

        self.possible_agents = tuple(str(x) for x in range(self.n_pursuers))


        self.observation_spaces = {
            str(i): spaces.Box(-high, high, dtype=np.float64) for i in self.possible_agents
        }

        self.action_spaces = {
            str(i): spaces.Box(low=actlow, high=acthigh) for i in self.possible_agents
        }

        self.grid = CircularContinousWorld(radius=430, block_coords=None)


    def get_agents(self, state: PPState) -> List[M.AgentID]:
        return list(self.possible_agents)


    def sample_initial_state(self) -> PPState:

        predator_coords = []
        prev_predator_coords = []
        for i in range(self.n_pursuers):
                x = 50 * (-math.floor(self.n_pursuers / 2) + i)  # distributes the agents based in their number
                predator_coords.append((x, 0, 0))
                prev_predator_coords.append((x,0,0))
        
        x=float(self.rng.random() * 600 - 300)
        
        y=float(self.rng.random() * 600 - 300)

        pursuer_vel=float(self.rng.random() * self.vel_tar)

        prey_coords = (x,y,0)
        prev_prey_coords = (x,y,0)

        return PPState(tuple(predator_coords), tuple(prev_predator_coords), prey_coords, tuple(prev_prey_coords), pursuer_vel)

    def sample_initial_obs(self, state: PPState) -> Dict[M.AgentID, PPObs]:
        # print(state, state)
        return self._get_obs(state)
    
    def step(
        self, state: PPState, actions: Dict[M.AgentID, PPAction]
    ) -> M.JointTimestep[PPState, PPObs]:
        next_state = self._get_next_state(state, actions)
        obs = self._get_obs(state)
        done, rewards = self._get_rewards(state)

        dones = {}

        for i in self.possible_agents:
            dones[i] = done
 
        truncated = {i: False for i in self.possible_agents}

        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}

        return M.JointTimestep(
            next_state, obs, rewards, dones, truncated, done, info
        )
    
    def abs_sum(self, vector : List[float]) -> float :
        return sum(map(abs, vector))
    
    def check_capture(self, prey_coords : Position, predator_coords: Position) -> bool:
        dist = CircularContinousWorld.euclidean_dist(prey_coords, predator_coords)
        return dist < self.cap_rad
    
    def target_distance(self, state: PPState, index : int):
        return CircularContinousWorld.euclidean_dist(state.predator_coords[index], state.prey_coords)

    def _get_next_state(self, state : PPState, actions):
        prev_prey = state.prey_coords
        prev_predator = state.predator_coords

        new_prey_coords = self.target_move_repulsive(state.prey_coords, state)
        new_predator_coords = []
        for i, pred_pos in enumerate(state.predator_coords):
            velocity_factor = 1 if not self.velocity_control else actions[str(i)][1]
            new_coords, _ = self.grid._get_next_coord(pred_pos, actions[str(i)][0], state.pursuer_vel * velocity_factor, True)
            new_predator_coords.append(new_coords)

        return PPState(tuple(new_predator_coords), prev_predator, new_prey_coords, prev_prey, state.pursuer_vel)

    def _get_obs(self, state : PPState) -> Dict[M.AgentID, PPObs]:
        # Get observation for all pursuers

        observation = {}
        
        for i in range(self.n_pursuers):

            """ getting the target engagement """
            (alpha_t, _, dist_t, _, _), target_suc = self.engagmment(state.predator_coords[i], state.prey_coords, dist_factor=self.arena_dim_square)
            (alpha_t_prev, _, dist_t_prev, _, _), target_prev_suc = self.engagmment(state.prev_predator_coords[i], state.prev_prey_coords, dist_factor=self.arena_dim_square)

            engagment = []

            """ getting the relative engagement """
            for j in range(self.n_pursuers):
                if j != i:
                    eng, _ = self.engagmment(state.predator_coords[i], state.predator_coords[j], 
                        dist_factor=self.arena_dim_square,
                    )
                    engagment.append(eng)
            # Put any invalid (-1) to the end
            def key_func(t):
                if t[1] == -1:
                    return float('inf')
                else:
                    return t[1]

            engagment = sorted(engagment, key=key_func)
            alphas, _, dists, _, _ = list(zip(*engagment))
           
            # change in alpha
            if not target_suc or not  target_prev_suc:
                alpha_rate = -1
                dist_rate = -1
                if target_suc:
                    north = -state.prev_predator_coords[i][2] / math.pi
                else:
                    north = -1
                north_prev = -1
                turn_rate = -1

            else:
                alpha_rate = alpha_t - alpha_t_prev
                alpha_rate = math.asin(math.sin(alpha_rate)) / 4
                north = -state.predator_coords[i][2] / math.pi
                north_prev = -state.prev_predator_coords[i][2] / math.pi
                turn_rate = (north - north_prev) / 2
                dist_rate = (dist_t - dist_t_prev) / 0.03

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

            observation[str(i)] = temp_observation

        return observation
        
    def _get_rewards(self, state : PPState):
        done = False

        reward : Dict[M.AgentID, float]  = {}

        Q_formation = self.q_parameter(state)

        for id in range(len(state.predator_coords)):
            reward[str(id)] = 0
            reward[str(id)] -= Q_formation * 0.1
            reward[str(id)] -= 0.002 * self.target_distance(state, id)

            # import pdb; pdb.set_trace()

            if self.check_capture(state.prey_coords, state.predator_coords[id]):
                done = True
                reward[str(id)] += 30  # 320% more than the others

        if done:
            for id in self.possible_agents:
                reward[id] += 100

        return done, reward           

    def engagmment(self, agent_i : Position, agent_j : Position, dist_factor : float) -> Tuple[Tuple[float, ...], bool]:
        dist = CircularContinousWorld.euclidean_dist(agent_i, agent_j)

        yaw = agent_i[2]

        R = np.array(
            [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
        )

        T_p =np.array([agent_i[0] - agent_j[0], agent_i[1] - agent_j[1]])
        Los_angle = math.atan2(T_p[1], T_p[0])
        T_p = R.dot(T_p)
        alpha = math.atan2(T_p[1], T_p[0])

        if dist > self.observation_limit:
            return tuple([-1] * 5), False

        return (
            alpha,
            Los_angle,
            dist / dist_factor,
            T_p[0] / dist_factor,
            T_p[1] / dist_factor,
        ), True

    def get_closest(self, state : PPState) -> int:
        min_dist, min_index = None, None
        for idx, p in enumerate(state.predator_coords):
            dist = np.linalg.norm([p[0] - state.prey_coords[0], p[1] - state.prey_coords[1]])
            if min_dist is None or dist < min_dist:
                min_index = idx
                min_dist = dist

        if min_dist is None or min_index is None:
            raise Exception("No closest index found. Something has gone wrong.")
        
        return min_index

    def get_unit_vectors(self, state : PPState) -> List[List[float]]:
        unit = []
        for p in state.predator_coords:
            dist = np.linalg.norm([p[0] - state.prey_coords[0], p[1] - state.prey_coords[1]])
            unit.append([p[0] / dist, p[1] / dist])
        return unit


    def q_parameter(self, state : PPState) -> float:
        closest = self.get_closest(state)
        unit = self.get_unit_vectors(state)
        Qk = 0

        for i in range(self.n_pursuers):
            if i != closest:
                Qk = Qk + np.dot(unit[i], unit[closest]) + 1
        Qk = Qk / len(unit)

        return Qk


    def scale_vector(self, vector, scale, final_vector, factor=1.0):
        div = self.abs_sum(vector)
        if div < 0.00001:
            div = 0.00001
        f = -factor * scale(self.abs_sum(vector)) / div
        vector = f * vector
        return list(map(lambda x, y: x + y, final_vector, vector))

    def target_move_repulsive(self, position : Position, state : PPState) -> Position:
        # import pdb; pdb.set_trace()
        xy_pos = np.array(position[:2])
        x,y = xy_pos

        scale = lambda x: 50000 / (abs(x) + 200) ** 2
        n_pursuer = len(state.predator_coords)
        final_vector = (0.0, 0.0)

        for i in state.predator_coords:
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
        x += self.vel_tar * (dx/d)
        y += self.vel_tar * (dy/d)

        return x,y,0


    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[SupportsFloat, SupportsFloat]]:
        return {i: (0.0, self.R_MAX) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng
    
    
