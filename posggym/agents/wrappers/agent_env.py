"""Wrapper for incorporating posggym.agents as part of the environment."""
from typing import Callable, Dict, List, Tuple

from gymnasium import spaces

import posggym
import posggym.agents as pga


class AgentEnvWrapper(posggym.Wrapper):
    """This wrapper incorporates posggym.agent policies as part of the environment.

    This wrapper makes it so that specified agents within the environment have their
    actions determined internally by a posggym.agent policy. The environment wrapper
    will only return observations, rewards, etc, for agents which are not controlled.

    Arguments
    ---------
    env : posggym.Env
        The environment to apply the wrapper
    agent_fn : Callable[[posggym.POSGModel], Dict[str, pga.Policy]]
        A function which returns a policy for each agent ID in the environment to be
        controlled as part of the environment. Should always return a policy for the
        same set of agent IDs.
    """

    def __init__(
        self,
        env: posggym.Env,
        agent_fn: Callable[[posggym.POSGModel], Dict[str, pga.Policy]],
    ):
        """Initializes the wrapper."""
        super().__init__(env)
        self.agent_fn = agent_fn
        self.policies = agent_fn(self.model)
        self.controlled_agents = list(self.policies)
        self.last_obs = {}
        self.last_terminateds = {}

    @property
    def possible_agents(self) -> Tuple[str, ...]:
        return tuple(
            i for i in super().possible_agents if i not in self.controlled_agents
        )

    @property
    def agents(self) -> List[str]:
        return [i for i in super().agents if i not in self.controlled_agents]

    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        return {
            i: act_space
            for i, act_space in super().action_spaces.items()
            if i not in self.controlled_agents
        }

    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        return {
            i: obs_space
            for i, obs_space in super().observation_spaces.items()
            if i not in self.controlled_agents
        }

    @property
    def reward_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            i: rew_range
            for i, rew_range in super().reward_ranges.items()
            if i not in self.controlled_agents
        }

    def reset(self, **kwargs):
        obs, infos = super().reset(**kwargs)
        self.policies = self.agent_fn(self.model)

        assert set(self.policies) == set(
            self.controlled_agents
        ), "Agent IDs must be consistent across resets."

        for policy in self.policies.values():
            policy.reset()

        self.last_obs = {i: obs[i] for i in self.controlled_agents if i in obs}
        self.last_terminateds = {i: False for i in self.controlled_agents}
        return {i: o for i, o in obs.items() if i not in self.controlled_agents}, {
            i: info for i, info in infos.items() if i not in self.controlled_agents
        }

    def step(self, actions):
        for i, policy in self.policies.items():
            assert i not in actions, f"Agent {i} is controlled by the environment."
            if not self.last_terminateds[i]:
                actions[i] = policy.step(self.last_obs[i])

        obs, rewards, terminated, truncated, dones, infos = super().step(actions)
        self.last_obs = {i: obs[i] for i in self.controlled_agents if i in obs}
        self.last_terminateds = {
            i: terminated[i] for i in self.controlled_agents if i in terminated
        }
        return (
            {i: obs[i] for i in obs if i not in self.controlled_agents},
            {i: rewards[i] for i in rewards if i not in self.controlled_agents},
            {i: terminated[i] for i in terminated if i not in self.controlled_agents},
            {i: truncated[i] for i in truncated if i not in self.controlled_agents},
            dones,
            {i: infos[i] for i in infos if i not in self.controlled_agents},
        )
