from highway_env.envs.common.abstract import AbstractEnv

from posggym import core

from posggym.envs.highway_env.model import HWModel, set_env_state


class HWEnv(core.DefaultEnv):
    """The HighwayEnv.

    Ref: https://github.com/eleurent/highway-env

    """

    metadata = AbstractEnv.metadata

    def __init__(self, n_agents: int, env: AbstractEnv, **kwargs):
        self._model = HWModel(n_agents, env, **kwargs)
        super().__init__()
        self._env = env

    def render(self, mode: str = "human"):
        set_env_state(self._env, self._state)
        return self._env.render(mode)

    @property
    def model(self) -> HWModel:
        return self._model

    def close(self) -> None:
        self._env.close()
