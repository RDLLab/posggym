"""Wrapper for limiting the time steps of an environment."""
from typing import Set, Optional

import posggym
import posggym.model as M


class TimeLimit(posggym.Wrapper):
    """Wraps environment to enforce environment time limit.

    This wrapper will issue a `truncated` signal in the :meth:`step` method for any
    agents that have not already reached a terminal state by the time a maximum number
    of timesteps is exceeded. It will also signal that the episode is `done` for all
    agents.

    Ref:
    https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/wrappers/time_limit.py
    """

    def __init__(self, env: posggym.Env, max_episode_steps: Optional[int] = None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            assert env.spec is not None
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self._terminated_agents: Set[M.AgentID] = set()

    def step(self, actions):
        obs, rewards, terminated, truncated, done, info = self.env.step(actions)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            for i in self.agents:
                if i not in self._terminated_agents:
                    truncated[i] = True
            done = True
        else:
            for i, i_terminated in terminated.items():
                if i_terminated:
                    self._terminated_agents.add(i)

        return obs, rewards, terminated, truncated, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        obs, info = self.env.reset(**kwargs)
        self._terminated_agents = set()
        return obs, info
