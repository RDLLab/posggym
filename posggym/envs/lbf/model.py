"""The model for Level-Based Foraging environment.

Ref:
https://github.com/semitable/lb-foraging

"""
from typing import Tuple, Optional, List

import numpy as np
from gym import spaces

import posggym.model as M
import posggym.envs.lbf.core as lbf

LBFState = Tuple[np.ndarray, Tuple[lbf.Player, ...], int]


class LBFState:
    """State in Level-Based Foraging environment."""

    def __init__(self,
                 field: np.ndarray,
                 players: Tuple[lbf.Player, ...],
                 step: int):
        self.field = field
        self.players = players
        self.step = step

    def __eq__(self, o):
        return (
            (self.field == o.field).all()
            and self.players == o.players
            and self.step == o.step
        )

    def __hash__(self):
        return hash((self.field.tostring(), self.players, self.step))


def get_env_state(env: lbf.ForagingEnv) -> LBFState:
    """Get state of ForagingEnv."""
    field = np.array(env.field)
    players = tuple(p.copy() for p in env.players)
    step = env.current_step
    return LBFState(field, players, step)


def set_env_state(env: lbf.ForagingEnv, state: LBFState):
    """Set state of ForagingEnv."""
    env.field = np.array(state.field)
    env.players = list(p.copy() for p in state.players)
    env.current_step = state.step


class LBFBelief(M.Belief):
    """The initial belief for Level-Based Foraging environment."""

    def __init__(self, env: lbf.ForagingEnv):
        self._env = env

    def sample(self) -> M.State:
        self._env.reset()
        return get_env_state(self._env)


class LBFModel(M.POSGModel):
    """Level-Based Foraging model.

    Parameters
    ----------
    num_agents : int
        the number of agents in the environment
    max_agent_level : int
        maximum foraging level of any agent
    field_size : Tuple[int, int]
        width and height of the playing field
    max_food : int
        the max number of food that can be spawned
    sight : int
        observation size of each agent
    force_coop : bool
        whether to force cooperation or not
    normalize_reward : bool, optional
        whether to normalize the reward or not (default=True)
    observation_mode : str, optional
        The observation mode for agent (default='tuple')
          - 'grid' - observations are multiple 2D grids (3D np.ndarray)
          - 'vector' - observations are vector (1D np.ndarray)
          - 'tuple' - observations are a tuple with same format as 'vector'
                      observations but as a hashable Python tuple object
                      containing integers instead of floats

    penalty : float, optional
        the penalty for failing to load food (default=0.0)

    """

    OBSERVATION_MODES = [
        "grid", "vector", "tuple"
    ]

    # pylint: disable=unused-argument
    def __init__(self,
                 num_agents: int,
                 max_agent_level: int,
                 field_size: Tuple[int, int],
                 max_food: int,
                 sight: int,
                 max_episode_steps: int,
                 force_coop: bool,
                 static_layout: bool,
                 normalize_reward: bool = True,
                 observation_mode: str = "tuple",
                 penalty: float = 0.0,
                 **kwargs):
        assert observation_mode in self.OBSERVATION_MODES

        super().__init__(num_agents, **kwargs)
        self._env = lbf.ForagingEnv(
            num_agents,
            max_agent_level,
            field_size,
            max_food,
            sight,
            max_episode_steps,
            force_coop,
            static_layout,
            normalize_reward,
            grid_observation=observation_mode not in ('vector', 'tuple'),
            penalty=penalty
        )
        self._env.seed(kwargs.get("seed", None))
        self.observation_mode = observation_mode
        self.field_size = field_size

    @property
    def observation_first(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def state_space(self) -> spaces.Space:
        raise NotImplementedError

    @property
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        return tuple(
            spaces.Discrete(len(lbf.LBFAction)) for _ in range(self.n_agents)
        )

    @property
    def observation_spaces(self) -> Tuple[spaces.Space, ...]:
        agent_obs_space = self._env._get_observation_space()
        return tuple(agent_obs_space for _ in range(self.n_agents))

    @property
    def reward_ranges(self) -> Tuple[Tuple[M.Reward, M.Reward], ...]:
        if self._env.normalize_reward:
            return ((self._env.penalty, 1.0), ) * self.n_agents
        return (
            (self._env.penalty, self._env.max_agent_level),
        ) * self.n_agents

    @property
    def initial_belief(self) -> LBFBelief:
        return LBFBelief(self._env)

    def step(self,
             state: M.State,
             actions: M.JointAction) -> M.JointTimestep:
        set_env_state(self._env, state)
        obs, rewards, dones, _ = self._env.step(actions)
        next_state = get_env_state(self._env)
        all_done = all(dones)
        outcomes = (M.Outcome.NA,) * self.n_agents

        if self.observation_mode == "tuple":
            # convert from vector obs to tuple obs
            obs = tuple(tuple(int(x) for x in o) for o in obs)

        return M.JointTimestep(
            next_state, obs, rewards, dones, all_done, outcomes
        )

    def set_seed(self, seed: Optional[int] = None):
        self._env.seed(seed)

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        set_env_state(self._env, state)
        obs, _, _, _ = self._env._make_gym_obs()

        if self.observation_mode == "tuple":
            # convert from vector obs to tuple obs
            obs = tuple(tuple(int(x) for x in o) for o in obs)

        return obs

    def get_agent_initial_belief(self,
                                 agent_id: M.AgentID,
                                 obs: M.Observation) -> M.Belief:
        raise NotImplementedError

    def parse_obs(self,
                  obs: np.ndarray
                  ) -> Tuple[
                      List[Tuple[int, int, int]], List[Tuple[int, int, int]]
                  ]:
        """Parse observation into (x, y, level) agent and food triplets.

        Agent obs are ordered so the observing agent is first, then the
        remaining observations are by agent order.

        On triplet of [-1, -1, 0] means no observation for the given agent or
        food.
        """
        if self.observation_mode == "grid":
            return self.parse_grid_obs(obs)
        elif self.observation_mode == "vector":
            return self.parse_vector_obs(obs)
        return self.parse_tuple_obs(obs)

    def parse_grid_obs(self,
                       obs: np.ndarray
                       ) -> Tuple[
                           List[Tuple[int, int, int]],
                           List[Tuple[int, int, int]]
                       ]:
        """Parse grid observation int (x, y, level) agent and food triplets.

        Agent obs are ordered so the observing agent is first, then the
        remaining observations are by agent order.

        On triplet of [-1, -1, 0] means no observation for the given agent or
        food.
        """
        raise NotImplementedError

    def parse_vector_obs(self,
                         obs: np.ndarray
                         ) -> Tuple[
                             List[Tuple[int, int, int]],
                             List[Tuple[int, int, int]]
                         ]:
        """Parse vector obs into (x, y, level) agent and food triplets.

        Agent obs are ordered so the observing agent is first, then the
        remaining observations are by agent order.

        On triplet of [-1, -1, 0] means no observation for the given agent or
        food.
        """
        assert obs.shape[0] == 3 * (self.n_agents + self._env.max_food)
        agent_obs = []
        food_obs = []
        for i in range(0, obs.shape[0], 3):
            triplet = tuple(int(x) for x in obs[i:i+3])
            if i < self._env.max_food * 3:
                food_obs.append(triplet)
            else:
                agent_obs.append(triplet)
        return agent_obs, food_obs

    def parse_tuple_obs(self,
                        obs: Tuple[int, ...]
                        ) -> Tuple[
                            List[Tuple[int, int, int]],
                            List[Tuple[int, int, int]]
                        ]:
        """Parse tuple obs into (x, y, level) agent and food triplets.

        Agent obs are ordered so the observing agent is first, then the
        remaining observations are by agent order.

        On triplet of [-1, -1, 0] means no observation for the given agent or
        food.
        """
        assert len(obs) == 3 * (self.n_agents + self._env.max_food)
        agent_obs = []
        food_obs = []
        for i in range(0, len(obs), 3):
            triplet = obs[i:i+3]
            if i < self._env.max_food * 3:
                food_obs.append(triplet)
            else:
                agent_obs.append(triplet)
        return agent_obs, food_obs
