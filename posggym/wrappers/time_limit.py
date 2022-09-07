from typing import Tuple, Dict

import posggym.model as M
from posggym.core import Env, Wrapper


class TimeLimit(Wrapper):
    """Wraps environment to enforce environment time limit.

    Ref:
    https://github.com/openai/gym/blob/v0.21.0/gym/wrappers/time_limit.py
    """

    def __init__(self, env: Env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps

        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self,
             actions: M.JointAction,
             ) -> Tuple[M.JointObservation, M.JointReward, bool, Dict]:
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"

        observations, rewards, done, info = self.env.step(actions)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observations, rewards, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
