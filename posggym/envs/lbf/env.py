from typing import Tuple

from posggym import core

import posggym.envs.lbf.core as lbf
import posggym.envs.lbf.model as lbfmodel


class LBFEnv(core.DefaultEnv):
    """The Level-Based Foraging Environment.

    Ref: https://github.com/semitable/lb-foraging

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
                 normalize_reward: bool = True,
                 grid_observation: bool = False,
                 penalty: float = 0.0,
                 **kwargs):
        self._model = lbfmodel.LBFModel(
            num_agents,
            max_agent_level,
            field_size,
            max_food,
            sight,
            max_episode_steps,
            force_coop,
            normalize_reward,
            grid_observation,
            penalty,
            **kwargs
        )
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
        super().__init__()

    def render(self, mode: str = "human"):
        lbfmodel.set_env_state(self._env, self._state)
        return self._env.render(mode)

    @property
    def model(self) -> lbfmodel.LBFModel:
        return self._model

    def close(self) -> None:
        self._env.close()
