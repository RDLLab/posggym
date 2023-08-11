"""Wrapper for converting a posggym environment into rllib multi-agent environment."""
import warnings
from typing import Optional, Set, Tuple

from gymnasium import spaces

import posggym

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from ray.rllib.env.multi_agent_env import MultiAgentEnv
        from ray.rllib.utils.typing import AgentID, MultiAgentDict
except ImportError:
    raise posggym.error.DependencyNotInstalled(
        "The posggym.wrapper.rllib_multi_agent_env wrapper depends on the Ray RLlib "
        "library. run `pip install ray[rllib]>=2.3` or visit "
        "'https://docs.ray.io/en/latest/ray-overview/installation.html` for more "
        "details on installing rllib. "
    )


class RllibMultiAgentEnv(MultiAgentEnv):
    """An interface between a POSGGym env and a Rllib MultiAgentEnv.

    Converts a ``posggym.Env`` into a ``ray.rllib.MultiAgentEnv``.

    References
    ----------

    - https://github.com/ray-project/ray/blob/ray-2.3.0/rllib/env/multi_agent_env.py

    """

    def __init__(self, env: posggym.Env):
        self.env = env
        self._done_agents: Set[AgentID] = set()

        # must assign this first before calling super().__init__() so that
        # property functions are initialized before super().__init__() is
        # called
        self._agent_ids = set(self.env.possible_agents)
        # Do the action and observation spaces map from agent ids to spaces
        # for the individual agents?
        self._action_space_in_preferred_format = True
        self._obs_space_in_preferred_format = True

        super().__init__()

    @property
    def observation_space(self):
        """Get the environment observation space.

        This is a dictionary mapping agent IDs to the agent's observation space
        """
        return spaces.Dict(self.env.observation_spaces)

    @property
    def action_space(self):
        """Get the environment action space.

        This is a dictionary mapping agent IDs to the agent's action space
        """
        return spaces.Dict(self.env.action_spaces)

    def get_agent_ids(self) -> Set[AgentID]:
        """Return a set of agent ids in the environment."""
        return self._agent_ids

    def reset(  # type: ignore
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        self._done_agents = set()
        return self.env.reset(seed=seed, options=options)

    def step(  # type: ignore
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns
        -------
        observations: new observations for each ready agent
        rewards: reward values for each ready agent. If the episode is just started, the
            value will be None.
        terminateds: terminated values for each ready agent. The special key "__all__"
            (required) is used to indicate env termination.
        truncateds: truncated values for each ready agent.
        infos: info values for each agent id (may be empty dicts).

        """
        obs, rewards, terminated, truncated, all_done, info = self.env.step(action_dict)

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

        # need to add RLlib specific special key to signify episode termination
        terminated["__all__"] = all_done
        truncated["__all__"] = all(truncated.values())

        return obs, rewards, terminated, truncated, info

    def render(self):
        output = self.env.render()
        if isinstance(output, dict):
            return output.get("env", None)

        if self.env.render_mode == "human" and output is None:
            return True
        return output

    def close(self):
        self.env.close()

    @property
    def unwrapped(self):
        return self
