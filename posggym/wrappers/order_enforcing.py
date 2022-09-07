from typing import Tuple, Dict

import posggym.model as M
from posggym.core import Env, Wrapper


class OrderEnforcing(Wrapper):
    """Wraps environment to enforce environment is reset before stepped.

    Ref:
    https://github.com/openai/gym/blob/v0.21.0/gym/wrappers/order_enforcing.py
    """

    def __init__(self, env: Env, max_episode_steps=None):
        super().__init__(env)
        self._elapsed_steps = None

    def step(self,
             actions: M.JointAction,
             ) -> Tuple[M.JointObservation, M.JointReward, bool, Dict]:
        assert self._has_reset, "Cannot call env.step() before calling reset()"
        observations, rewards, done, info = self.env.step(actions)
        return observations, rewards, done, info

    def reset(self, **kwargs):
        self._has_reset = True
        return self.env.reset(**kwargs)
