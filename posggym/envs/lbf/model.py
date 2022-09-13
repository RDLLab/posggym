"""The model for Level-Based Foraging environment.

Ref:
https://github.com/semitable/lb-foraging

"""
import copy
from typing import Tuple, Optional, List

import numpy as np
from gym import spaces

import posggym.model as M
import posggym.envs.lbf.core as lbf

LBFState = Tuple[np.ndarray, Tuple[lbf.Player, ...], int]


def get_env_state(env: lbf.ForagingEnv) -> LBFState:
    """Get state of ForagingEnv."""
    field = np.array(env.field)
    players = tuple(copy.deepcopy(env.players))
    step = env.current_step
    return (field, players, step)


def set_env_state(env: lbf.ForagingEnv, state: LBFState):
    """Set state of ForagingEnv."""
    env.field = state[0]
    env.players = list(state[1])
    env.current_step = state[2]


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
    grid_observation : bool, optional
        whether agents observations are multiple 2D grids (True) or a vector
        (False) (default=False)
    penalty : float, optional
        the penalty for failing to load food (default=0.0)

    """
    # pylint: disable=unused-argument
    def __init__(self,
                 num_agents: int,
                 max_agent_level: int,
                 field_size: Tuple[int, int],
                 max_food: int,
                 sight: int,
                 max_episode_steps: int,
                 force_coop: bool,
                 normalize_reward: bool = True,
                 grid_observation: bool = False,
                 penalty: float = 0.0,
                 **kwargs):
        super().__init__(num_agents, **kwargs)
        self._env = lbf.ForagingEnv(
            num_agents,
            max_agent_level,
            field_size,
            max_food,
            sight,
            max_episode_steps,
            force_coop,
            normalize_reward,
            grid_observation,
            penalty
        )
        self._env.seed(kwargs.get("seed", None))
        self.grid_observation = grid_observation
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
        return M.JointTimestep(
            next_state, obs, rewards, dones, all_done, outcomes
        )

    def set_seed(self, seed: Optional[int] = None):
        self._env.seed(seed)

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        set_env_state(self._env, state)
        obs, _, _, _ = self._env._make_gym_obs()
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
        if self.grid_observation:
            return self.parse_grid_obs(obs)
        return self.parse_vector_obs(obs)

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
