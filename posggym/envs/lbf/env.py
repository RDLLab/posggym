from typing import Tuple

from posggym import core

import posggym.envs.lbf.core as lbf
import posggym.envs.lbf.model as lbfmodel


class LBFEnv(core.DefaultEnv):
    """The Level-Based Foraging Environment.

    This implementation uses and is based on the original implementation of
    the Level-Based Foraging environment:

    https://github.com/semitable/lb-foraging

    We provide a discription here for convinience.

    Agents
    ------
    Varied number

    State
    -----
    The state of the environment is defined by a (x, y, level) triplet for each
    agent and food object in the environment. The (x, y) components define the
    position, starting from the bottom left square, while the level is the
    level of the agent or food.

    Actions
    -------
    Each agent has six possible discrete actions [0-5] corresponding to:

        Noop, Move North, Move South, Move West, Move East, Pickup

    Observations
    ------------
    Each agent observes the (x, y, level) of food and other agent within their
    field of vision, which is the grid `sight` distance away from the agent in
    all directions.

    There are three observation modes:

    1. grid_observation
       The agent recieves a three 2D layers of size (1+2*`sight`, 1+2*`sight`).
       Each cell in each layer corresponds to a specific (x, y) position
       relative to the observing agent.
       The layers are:
         i. agent level
         ii. food level
         iii. whether cell is free or blocked (e.g. out of bounds)
    2. vector observation
       A vector of (x, y, level) triplets for each food and agent in the
       environment. If a given food or agent is not within the observing
       agent's field of vision triplets have a value of [-1, -1, 0].
       The size of the vector is (`num_agents` + `max_food`) * 3, with the
       first `max_food` triplets being for the food, the `max_food`+1 triplet
       being for the observing agent and the remaining `num_agents`-1
       triplets for the other agents.
       The ordering of the triplets for the other agents is consistent, while
       the food obs triplets can change based on how many are visible and their
       relative position to the observing agent.
    3. tuple observation
       This is the same as the vector observation except observations are
       Python tuples of integers instead of numpy arrays of floats.

    """

    metadata = lbf.ForagingEnv.metadata

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
        self._model = lbfmodel.LBFModel(
            num_agents,
            max_agent_level,
            field_size=field_size,
            max_food=max_food,
            sight=sight,
            max_episode_steps=max_episode_steps,
            force_coop=force_coop,
            static_layout=static_layout,
            normalize_reward=normalize_reward,
            observation_mode=observation_mode,
            penalty=penalty,
            **kwargs
        )
        self._env = lbf.ForagingEnv(
            num_agents,
            max_agent_level,
            field_size=field_size,
            max_food=max_food,
            sight=sight,
            max_episode_steps=max_episode_steps,
            force_coop=force_coop,
            static_layout=static_layout,
            normalize_reward=normalize_reward,
            grid_observation=observation_mode not in ('vector', 'tuple'),
            penalty=penalty
        )
        super().__init__()

    def render(self, mode: str = "human"):
        lbfmodel.set_env_state(self._env, self._state)
        return self._env.render(mode)

    @property
    def model(self) -> lbfmodel.LBFModel:
        return self._model

    def close(self) -> None:
        self._env.close()
