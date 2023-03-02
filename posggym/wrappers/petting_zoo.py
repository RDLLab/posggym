"""Wrapper for converting a posggym environment into pettingzoo environment."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from gymnasium import spaces

import posggym
from posggym import error


try:
    from pettingzoo.utils.env import ActionDict, AgentID, ObsDict, ParallelEnv
except ImportError as e:
    raise error.DependencyNotInstalled(
        "pettingzoo is not installed, run `pip install pettingzoo` or visit "
        "'https://github.com/Farama-Foundation/PettingZoo#installation' for details on "
        "installing pettingzoo."
    ) from e


class PettingZoo(ParallelEnv):
    """Converts POSGGym environment into a PettingZoo environment.

    Converts into a pettingzoo.ParallelEnv environment.

    Ref:
    https://pettingzoo.farama.org/api/parallel/
    https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/utils/env.py

    """

    def __init__(self, env: posggym.Env):
        self.env = env
        self._int_ids = isinstance(self.env.possible_agents[0], int)
        self._done_agents: Set[AgentID] = set()

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.env.metadata

    @property
    def agents(self) -> List[AgentID]:
        return list(str(i) for i in self.env.agents if i not in self._done_agents)

    @property
    def possible_agents(self) -> List[AgentID]:
        return list(str(i) for i in self.env.possible_agents)

    @property
    def action_spaces(self) -> Dict[AgentID, spaces.Space]:
        return {
            str(i): action_space for i, action_space in self.env.action_spaces.items()
        }

    @property
    def observation_spaces(self) -> Dict[AgentID, spaces.Space]:
        return {
            str(i): obs_space for i, obs_space in self.env.observation_spaces.items()
        }

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
        if obs is None:
            obs = {str(i): None for i in self.env.agents}

        self._done_agents = set()

        if return_info:
            return (
                {str(i): v for i, v in obs.items()},
                {str(i): v for i, v in info.items()},
            )
        return {str(i): v for i, v in obs.items()}

    def step(
        self, actions: ActionDict
    ) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:
        if self._int_ids:
            actions = {int(i): a for i, a in actions.items()}

        obs, rewards, terminated, truncated, all_done, info = self.env.step(actions)

        # remove obs for already done agents, since pettingzoo expects no output for
        # agents that were terminated or truncated in a previous step
        for d in (obs, rewards, terminated, truncated, info):
            for i in self._done_agents:
                d.pop(i, None)

        # add any newly done agents to set of done agents so they are handled
        # appropriatly next step
        for i, done in terminated.items():
            if done:
                self._done_agents.add(i)
        for i, done in truncated.items():
            if done:
                self._done_agents.add(i)

        return (
            {str(i): v for i, v in obs.items()},
            {str(i): float(v) for i, v in rewards.items()},
            {str(i): v for i, v in terminated.items()},
            {str(i): v for i, v in truncated.items()},
            {str(i): v for i, v in info.items()},
        )

    def render(self) -> None | np.ndarray | str | List:
        output = self.env.render()
        if isinstance(output, dict):
            return output.get("env", None)
        return output

    def close(self):
        self.env.close()

    def state(self):
        return self.env.state

    def observation_space(self, agent: AgentID) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> spaces.Space:
        return self.action_spaces[agent]
