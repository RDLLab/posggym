"""Wrapper for converting a posggym environment into pettingzoo environment."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import posggym

try:
    from pettingzoo.utils.env import ActionDict, ObsDict, ParallelEnv
except ImportError as e:
    raise posggym.error.DependencyNotInstalled(
        "pettingzoo is not installed, run `pip install pettingzoo` or visit "
        "'https://github.com/Farama-Foundation/PettingZoo#installation' for details on "
        "installing pettingzoo."
    ) from e


if TYPE_CHECKING:
    import numpy as np
    from gymnasium import spaces


class PettingZoo(ParallelEnv):
    """Converts POSGGym environment into a PettingZoo environment.

    Converts into a ``pettingzoo.ParallelEnv`` environment.

    This involves:

    - treating agent IDs as strings (instead of ints in some environments)
    - handling case where an individual agent is done in environment before the episode
      is over

    References
    ----------

    - parallel env docs: https://pettingzoo.farama.org/api/parallel/
    - parallel env code:
      https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/utils/env.py

    """

    def __init__(self, env: posggym.Env):
        self.env = env
        self._done_agents: Set[str] = set()

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.env.metadata

    @property
    def agents(self) -> List[str]:
        return [i for i in self.env.agents if i not in self._done_agents]

    @property
    def possible_agents(self) -> List[str]:
        return list(self.env.possible_agents)

    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        return self.env.action_spaces

    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        return self.env.observation_spaces

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> ObsDict:
        obs, info = self.env.reset(seed=seed, options=options)
        self._done_agents = set()

        if return_info:
            return obs, info
        return obs

    def step(
        self, actions: ActionDict
    ) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:
        obs, rewards, terminated, truncated, all_done, info = self.env.step(actions)

        # remove obs for already done agents, since pettingzoo expects no output for
        # agents that were terminated or truncated in a previous step
        for d in (obs, rewards, terminated, truncated, info):
            for i in self._done_agents:
                d.pop(i, None)

        # add any newly done agents to set of done agents so they are handled
        # appropriately next step
        for i, done in terminated.items():
            if done:
                self._done_agents.add(i)
        for i, done in truncated.items():
            if done:
                self._done_agents.add(i)

        return obs, rewards, terminated, truncated, info

    def render(self) -> None | np.ndarray | str | List:
        output = self.env.render()
        if isinstance(output, dict):
            return output.get("env", None)
        return output

    def close(self):
        self.env.close()

    def state(self):
        return self.env.state

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]
