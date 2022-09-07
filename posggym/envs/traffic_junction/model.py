import random
from typing import Sequence, Dict, Tuple, Optional, NamedTuple

import numpy as np
from gym import spaces
from ma_gym.envs.traffic_junction import TrafficJunction

import posggym.model as M

Coord = Tuple[int, int]
Direction = Tuple[int, int]

# The actions
GAS = 0
BRAKE = 1

ACTIONS = [GAS, BRAKE]
ACTIONS_STR = ["G", "B"]

# Direction vectors
NORTH = (0, 1)
EAST = (1, 0)
SOUTH = (0, -1)
WEST = (-1, 0)

DIRECTIONS = [NORTH, EAST, SOUTH, WEST]

# Routes
GO_STRAIGHT = 1
TURN_RIGHT = 2
TURN_LEFT = 3

# Map from Route to map of direction to next direction
ROUTE_DIRECTION_MAP = {
    TURN_RIGHT: {
        NORTH: WEST, EAST: SOUTH
    }
}


class TJState(NamedTuple):
    """A state in the TrafficJunction ma-gym Environment."""
    step_count: int
    agent_pos: Tuple[Tuple[int, int], ...]
    agent_direction: Tuple[Tuple[int, int], ...]
    agent_routes: Tuple[int, ...]
    on_the_road: Tuple[bool, ...]
    agent_dones: Tuple[bool, ...]
    agent_collisions: Tuple[bool, ...]
    agent_turned: Tuple[bool, ...]
    agent_step_count: Tuple[int, ...]


class TrafficJunctionB0(M.Belief):
    """The initial belief for the TrafficJunction ma-gym environment."""

    def __init__(self,
                 n_agents: int,
                 route_vectors: Dict[Coord, Direction],
                 n_routes: int,
                 rng: random.Random,
                 dist_res: int = 1000):
        self._n_agents = n_agents
        self._route_vectors = route_vectors
        self._n_routes = n_routes
        self._rng = rng
        self._dist_res = dist_res

    def sample(self) -> M.State:
        shuffled_entry_gates = list(self._route_vectors.keys())
        self._rng.shuffle(shuffled_entry_gates)

        agent_pos = []
        agent_direction = []
        agent_routes = []
        agent_on_the_road = []
        curr_on_the_road = 0

        for i in range(self._n_agents):
            if curr_on_the_road >= len(self._route_vectors):
                # all entries taken so car not on road yet
                agent_pos.append((0, 0))
                agent_direction.append(NORTH)
                agent_on_the_road.append(False)
                agent_routes.append(0)
            else:
                pos = shuffled_entry_gates[i]
                agent_pos.append(pos)
                agent_direction.append(self._route_vectors[pos])
                agent_on_the_road.append(True)
                curr_on_the_road += 1
                agent_routes.append(self._rng.randint(GO_STRAIGHT, TURN_LEFT))

        return TJState(
            step_count=0,
            agent_pos=tuple(agent_pos),
            agent_direction=tuple(agent_direction),
            agent_routes=tuple(agent_routes),
            on_the_road=tuple(agent_on_the_road),
            agent_dones=tuple(False for _ in range(self._n_agents)),
            agent_collisions=tuple(False for _ in range(self._n_agents)),
            agent_turned=tuple(False for _ in range(self._n_agents)),
            agent_step_count=tuple(0 for _ in range(self._n_agents))
        )

    def sample_k(self, k: int) -> Sequence[M.State]:
        return [self.sample() for _ in range(k)]

    def get_dist(self) -> Dict[M.State, float]:
        return self.sample_belief_dist(self._dist_res)


class TrafficJunctionModel(M.POSGModel):
    """Model for TrafficJunction ma-gym environment.

    This code is copied and heavily adapted from the ma-gym library:
    https://github.com/koulanurag/ma-gym/blob/master/ma_gym/envs/traffic_junction/traffic_junction.py

    Changes to the code are primarily made so as to support the POSG model
    interface and use as a generative model.
    """

    GRID_SHAPE = (14, 14)
    AGENT_VIEW_MASK = (3, 3)
    STEP_COST = -0.01
    COLLISION_REWARD = -10
    ARRIVE_PROB = 0.5
    MAX_STEPS = 40
    N_AGENTS = 2
    N_ROUTES = 3

    def __init__(self, **kwargs):
        super().__init__(self.N_AGENTS, **kwargs)

        self._rng = random.Random(kwargs.get("seed", None))

        # entry gates where the cars spawn
        # Note: [(7, 0), (13, 7), (6, 13), (0, 6)] for (14 x 14) grid
        self._entry_gates = [
            (self.GRID_SHAPE[0] // 2, 0),
            (self.GRID_SHAPE[0] - 1, self.GRID_SHAPE[1] // 2),
            (self.GRID_SHAPE[0] // 2 - 1, self.GRID_SHAPE[1] - 1),
            (0, self.GRID_SHAPE[1] // 2 - 1)
        ]

        # destination places for the cars to reach
        # Note: [(7, 13), (0, 7), (6, 0), (13, 6)] for (14 x 14) grid
        self._destination = [
            (self.GRID_SHAPE[0] // 2, self.GRID_SHAPE[1] - 1),
            (0, self.GRID_SHAPE[1] // 2),
            (self.GRID_SHAPE[0] // 2 - 1, 0),
            (self.GRID_SHAPE[0] - 1, self.GRID_SHAPE[1] // 2 - 1)
        ]

        # dict{direction_vectors: (turn_right, turn_left)}
        # Note for (14 x14) grid:
        # [((7, 6), (7,7))), ((7, 7),(6,7)), ((6,6),(7, 6)), ((6, 7),(6,6))]
        self._turning_places = {
            NORTH: (
                (self.GRID_SHAPE[0] // 2, self.GRID_SHAPE[0] // 2 - 1),
                (self.GRID_SHAPE[0] // 2, self.GRID_SHAPE[0] // 2)
            ),
            WEST: (
                (self.GRID_SHAPE[0] // 2, self.GRID_SHAPE[0] // 2),
                (self.GRID_SHAPE[0] // 2 - 1, self.GRID_SHAPE[0] // 2)
            ),
            EAST: (
                (self.GRID_SHAPE[0] // 2 - 1, self.GRID_SHAPE[0] // 2 - 1),
                (self.GRID_SHAPE[0] // 2, self.GRID_SHAPE[0] // 2 - 1)
            ),
            SOUTH: (
                (self.GRID_SHAPE[0] // 2 - 1, self.GRID_SHAPE[0] // 2),
                (self.GRID_SHAPE[0] // 2 - 1, self.GRID_SHAPE[0] // 2 - 1)
            )
        }

        # dict{starting_place: direction_vector}
        self._route_vectors = {
            (self.GRID_SHAPE[0] // 2, 0): (0, 1),
            (self.GRID_SHAPE[0] - 1, self.GRID_SHAPE[0] // 2): (-1, 0),
            (0, self.GRID_SHAPE[0] // 2 - 1): (1, 0),
            (self.GRID_SHAPE[0] // 2 - 1, self.GRID_SHAPE[0] - 1): (0, -1)
        }

    @property
    def state_space(self) -> spaces.Space:
        # The global state is definte by:
        # 1. the current step
        # 2. the state of each vehicle in the environment

        # The State for each vehicle on the road is defined by:
        # 1. position (x, y) for x, y in interval [0, 1]
        # 2. direction (x, y) in [(0, 0), (0. 1), (1, 0), (1, 1)]
        # 3. route [1, 3]
        # 4. presence on the road [true, false]
        # 5. whether it's reached it's destination [true, false]
        # 6. whether the car has turned [true, false]
        # 5. steps since arriving on road [0, step_max]
        return None

    @property
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        return tuple(
            spaces.Discrete(len(ACTIONS)) for _ in range(self.n_agents)
        )

    @property
    def obs_spaces(self) -> Tuple[spaces.Space, ...]:
        # Each agent observes the (ID, position, route) for each agent/vehicle
        # (including itself) within a 3x3 grid around the agent (the agent is
        # at the center of the 3x3 grid.)

        # Note the original implementation encodes the positions in the [0, 1]
        # interval, here we keep it as coordinates in the grid.
        # This is to avoid doing float comparisons during search by BA-NRMCP.
        # This will lead to position being encoded as one-hot vectors for
        # deep RL policies
        vehicle_obs = spaces.Tuple((
            # agent ID
            spaces.Discrete(self.n_agents),
            # position
            spaces.Tuple((
                spaces.Discrete(self.GRID_SHAPE[0]),
                spaces.Discrete(self.GRID_SHAPE[1])
            )),
            # route
            spaces.Discrete(self.N_ROUTES)
        ))

        agent_obs = spaces.Tuple(tuple(
            vehicle_obs for _ in range(np.prod(self.AGENT_VIEW_MASK))
        ))

        return tuple(agent_obs for _ in range(self.n_agents))

    @property
    def reward_ranges(self) -> Tuple[Tuple[M.Reward, M.Reward], ...]:
        r_min = self.COLLISION_REWARD + self.MAX_STEPS * self.STEP_COST
        return tuple((r_min, 0.0) for _ in range(self.n_agents))

    @property
    def initial_belief(self) -> M.Belief:
        return TrafficJunctionB0(
            self.n_agents, self._route_vectors, self.N_ROUTES, self._rng
        )

    def get_agent_initial_belief(self,
                                 agent_id: M.AgentID,
                                 obs: M.Observation) -> M.Belief:
        # TODO
        return None

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        return self._get_obs(state)

    def step(self, state: M.State, actions: M.JointAction) -> M.JointTimestep:
        next_state = self._get_next_state(state, actions)
        obs = self._get_obs(next_state)
        rewards = self._get_rewards(next_state)
        outcome = self.get_outcome(next_state)
        return M.JointTimestep(
            next_state, obs, rewards, self.is_done(next_state), outcome
        )

    def is_done(self, state: M.State) -> bool:
        return all(state.agent_dones)

    def get_outcome(self, state: M.State) -> Tuple[M.Outcome, ...]:
        outcomes = []
        for i in range(self.n_agents):
            if not state.agent_dones[i]:
                outcomes.append(M.Outcome.NA)
            elif state.agent_pos[i] in [(0, 0), self._destination]:
                outcomes.append(M.Outcome.WIN)
            else:
                # time limit reached without reaching destination
                outcomes.append(M.Outcome.LOSS)
        return tuple(outcomes)

    def set_seed(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def is_absorbing(self, obs: M.Observation, agent_id: M.AgentID) -> bool:
        return False

    def _get_next_state(self,
                        state: TJState,
                        actions: M.JointAction) -> TJState:
        agent_pos = [*state.agent_pos]
        agent_direction = [*state.agent_direction]
        agent_routes = [*state.agent_routes]
        agent_on_the_road = [*state.on_the_road]
        agent_dones = [*state.agent_dones]
        # collisions reset for each step, unlike dones
        agent_collisions = list(False for _ in range(self.n_agents))
        agent_turned = [*state.agent_turned]
        agent_step_count = [*state.agent_step_count]

        for i, a_i in enumerate(actions):
            if state.agent_dones[i] or not state.on_the_road[i]:
                continue

            agent_step_count[i] += 1
            if a_i == GAS:
                next_pos, next_direction, next_turned = self._get_next_pos(
                    i, state
                )
                agent_direction[i] = next_direction
                agent_turned[i] = next_turned

                if next_pos in agent_pos:
                    # collision
                    agent_collisions[i] = True
                elif next_pos in self._destination:
                    agent_dones[i] = True
                    agent_pos[i] = (0, 0)
                else:
                    agent_pos[i] = next_pos

        if (
            not all(agent_on_the_road)
            and self._rng.random() < self.ARRIVE_PROB
        ):
            free_gates = [g for g in self._entry_gates if g not in agent_pos]
            if len(free_gates) > 0:
                # add next vehicle to the road
                next_i = agent_on_the_road.index(False)
                pos = self._rng.choice(free_gates)

                agent_pos[next_i] = pos
                agent_direction[next_i] = self._route_vectors[pos]
                agent_routes[next_i] = self._rng.randint(
                    GO_STRAIGHT, TURN_LEFT
                )
                agent_on_the_road[next_i] = True

        if state.step_count + 1 >= self.MAX_STEPS:
            agent_dones = list(True for _ in range(self.n_agents))

        return TJState(
            step_count=state.step_count + 1,
            agent_pos=tuple(agent_pos),
            agent_direction=tuple(agent_direction),
            agent_routes=tuple(agent_routes),
            on_the_road=tuple(agent_on_the_road),
            agent_dones=tuple(agent_dones),
            agent_collisions=tuple(agent_collisions),
            agent_turned=tuple(agent_turned),
            agent_step_count=tuple(agent_step_count)
        )

    def _get_next_pos(self,
                      agent_id: int,
                      state: TJState
                      ) -> Tuple[Coord, Direction, bool]:
        pos = state.agent_pos[agent_id]
        direction = state.agent_direction[agent_id]
        route = state.agent_routes[agent_id]
        turned = state.agent_turned[agent_id]
        turn_pos = self._turning_places[direction]

        next_direction = direction
        next_turned = turned
        if (
            route != GO_STRAIGHT
            and not turned
            and pos == turn_pos[route - GO_STRAIGHT - 1]
        ):
            # vehicle turning
            direction_idx = DIRECTIONS.index(direction)
            if route == TURN_RIGHT:
                next_direction_idx = (direction_idx + 1) % len(DIRECTIONS)
            else:
                next_direction_idx = (direction_idx - 1) % len(DIRECTIONS)

            next_direction = DIRECTIONS[next_direction_idx]
            next_turned = True

        next_pos = pos[0] + next_direction[0], pos[1] + next_direction[1]

        return next_pos, next_direction, next_turned

    def _get_obs(self, state: TJState) -> M.JointObservation:
        """Get joint obs.

        Ref: TrafficJunction.get_agent_obs()
        """
        # the observation of each agent's vehicle
        no_vehicle_obs = (0, (0, 0), 0)
        agent_vehicle_obs = []
        for i in range(self.n_agents):
            agent_vehicle_obs.append((
                i,
                state.agent_pos[i],
                state.agent_routes[i]
            ))

        joint_obs = []
        for i in range(self.n_agents):
            pos = state.agent_pos[i]
            agent_obs = []
            for row in range(
                    max(0, pos[0] - 1),
                    min(pos[0] + 1 + 1, self.GRID_SHAPE[0])
            ):
                for col in range(
                        max(0, pos[1] - 1),
                        min(pos[1] + 1 + 1, self.GRID_SHAPE[1])
                ):
                    if (row, col) in state.agent_pos:
                        j = state.agent_pos.index((row, col))
                        agent_obs.append(agent_vehicle_obs[j])
                    else:
                        agent_obs.append(no_vehicle_obs)
            joint_obs.append(tuple(agent_obs))
        return tuple(joint_obs)

    def _get_rewards(self, state: TJState) -> M.JointReward:
        rewards = []
        for i in range(self.n_agents):
            if state.agent_dones[i] or not state.on_the_road[i]:
                r_i = 0.0
            else:
                r_i = (
                    state.agent_collisions[i] * self.COLLISION_REWARD
                    + state.agent_step_count[i] * self.STEP_COST
                )
            rewards.append(r_i)
        return tuple(rewards)
