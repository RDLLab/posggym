"""Wrapper for converting a posggym environment into rllib multi-agent environment."""
from typing import Optional, Set, Tuple

from gymnasium import spaces

import posggym


try:
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

    This involves:
    - treating agent IDs as strings (instead of ints)
    - handling case where an individual agent is done in environment before the episode
      is over

    Ref:
    https://github.com/ray-project/ray/blob/ray-2.3.0/rllib/env/multi_agent_env.py

    """

    def __init__(self, env: posggym.Env):
        # must assign this first before calling super().__init__() so that
        # property functions are initialized before super().__init__() is
        # called
        self.env = env
        self._agent_ids = set(str(i) for i in self.env.possible_agents)

        self._int_ids = isinstance(self.env.possible_agents[0], int)
        self._done_agents: Set[AgentID] = set()

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
        return spaces.Dict(
            {str(i): self.env.observation_spaces[i] for i in self.env.possible_agents}
        )

    @property
    def action_space(self):
        """Get the environment action space.

        This is a dictionary mapping agent IDs to the agent's action space
        """
        return spaces.Dict(
            {str(i): self.env.action_spaces[i] for i in self.env.possible_agents}
        )

    def get_agent_ids(self) -> Set[AgentID]:
        """Return a set of agent ids in the environment."""
        return self._agent_ids

    def reset(  # type: ignore
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        obs, info = self.env.reset(seed=seed, options=options)
        if obs is None:
            obs = {str(i): None for i in self.env.agents}
        self._done_agents = set()
        return (
            {str(i): o for i, o in obs.items()},
            {str(i): o for i, o in info.items()}
        )

    def step(  # type: ignore
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns:
        observations: new observations for each ready agent
        rewards: reward values for each ready agent. If the episode is just started, the
            value will be None.
        terminateds: terminated values for each ready agent. The special key "__all__"
            (required) is used to indicate env termination.
        truncateds: truncated values for each ready agent.
        infos: info values for each agent id (may be empty dicts).

        """
        if self._int_ids:
            action_dict = {int(i): a for i, a in action_dict.items()}

        obs, rewards, terminated, truncated, all_done, info = self.env.step(action_dict)

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

        # need to add RLlib specific special key to signify episode termination
        terminated["__all__"] = all_done
        truncated["__all__"] = all(truncated.values())

        return (
            {str(i): v for i, v in obs.items()},
            {str(i): float(v) for i, v in rewards.items()},
            {str(i): v for i, v in terminated.items()},
            {str(i): v for i, v in truncated.items()},
            {str(i): v for i, v in info.items()},
        )

    def render(self):
        output = self.env.render()
        if isinstance(output, dict):
            return output.get("env", None)
        return output

    def close(self):
        self.env.close()

    @property
    def unwrapped(self):
        return self
