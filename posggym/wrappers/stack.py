"""Environment wrapper class that stacks agent observations into a single array."""
from typing import Any, Dict

import numpy as np
from gymnasium import spaces

import posggym


class StackEnv(posggym.Wrapper):
    """Converts environment to handle single array stacked agent inputs and outputs.

    This wrapper converts the :meth:`step` and :meth:`reset` functions handle actions,
    observations, rewards, and termination signals of all agents as a single stacked
    array, as opposed to a dictionary with one array per agent.

    For example, if the observation space of the unwrapped environment is such that::

        >>> env.observation_spaces
        {
            "agent-1": Box(0.0, 1.0, (4,), float32),
            "agent-2": Box(0.0, 1.0, (4,), float32),
        }
        >>> acts = {"agent-1": 1, "agent-2": 2}
        >>> obs, rews, terms, truncs, all_done, infos = env.step(acts)
        >>> obs
        {
            "agent-1": np.array([0.1, 0.2, 0.3, 0.4]),
            "agent-2": np.array([0.5, 0.6, 0.7, 0.8]),
        }
        >>> obs["agent-1"].shape
        (4,)

    Wrapping this environment with the :class:`StackEnv` wrapper will change the output
    and input to the form::

        >>> stacked_acts = np.array([1, 2])
        >>> obs, rews, terms, truncs, all_done, infos = wrapped_env.step(stacked_acts)
        >>> obs
        np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        >>> obs.shape
        (2, 4)
        >>> rews.shape
        (2,)
        >>> terms.shape
        (2,)
        >>> truncs.shape
        (2,)

    Note, if the environment is a vector environment, the output will be stacked
    in order of the each sub-environment, as opposed to in order of each agent. For
    example, if we have a vector environment with two agents and two sub-environments,
    the observation space of the vector environment will be something like::

        >>> vec_env.num_envs
        2
        >>> env.single_observation_spaces
        {
            "agent-1": Box(0.0, 1.0, (4,), float32),
            "agent-2": Box(0.0, 1.0, (4,), float32),
        }
        >>> env.observation_spaces
        {
            "agent-1": Box(0.0, 1.0, (2, 4), float32),
            "agent-2": Box(0.0, 1.0, (2, 4), float32),
        }

    The output from :meth:`step` and :meth:`reset` of the StackEnv wrapper around the
    vector env will be in the form of::

        >>> obs, rewards, terms, truncs, all_done, infos = wrapped_env.step(actions)
        >>> obs.shape
        (4, 4)

    With the first two rows corresponding to the outputs for `agent-1` and `agent-2` in
    the first sub-environment and the last two rows corresponding the respective agents
    outputs in the second sub-environment.

    Similarly, the actions passed to the step method of the StackEnv wrapper around the
    vector env are expected to be stacked in order of each environment. For example, for
    the vector environment above, assuming discrete actions, the action will look like::

        >>> actions
        np.array([0, 1, 0, 1])

    With the first two elements corresponding to the actions for `agent-1` and `agent-2`
    in the first sub-environment and the last two elements corresponding to the actions
    for the respective agents in the second sub-environment.

    """

    def __init__(self, env: posggym.Env):
        super().__init__(env)

        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)

        agent_0_obs_space = env.observation_spaces[env.possible_agents[0]]
        for obs_space in env.observation_spaces.values():
            assert isinstance(obs_space, spaces.Box), (
                f"StackEnv wrapper only supports environments with Box observation "
                f"spaces. Got {obs_space}. (Hint: try using the FlattenObservations "
                "wrapper.)"
            )
            assert obs_space == agent_0_obs_space, (
                f"StackEnv wrapper only supports environments where every agent "
                f"has the the same observation space. "
                f"Got {obs_space} and {agent_0_obs_space}."
            )

        agent_0_action_space = env.action_spaces[env.possible_agents[0]]
        for action_space in env.action_spaces.values():
            assert action_space == agent_0_action_space, (
                f"StackEnv wrapper only supports environments where every agent "
                f"has the the same action space. "
                f"Got {action_space} and {agent_0_action_space}."
            )

    def reset(self, **kwargs):
        obs, infos = super().reset(**kwargs)
        return self._stack_output(obs), infos

    def step(self, actions):
        # input shape (num_envs * num_agents, *single_action_space.shape)
        # convert to dict of actions, shape (num_envs, *single_action_space.shape)
        action_map = {
            i: actions[idx :: len(self.possible_agents)]
            for idx, i in enumerate(self.possible_agents)
        }
        obs, rewards, terminated, truncated, dones, infos = self.env.step(action_map)
        return (
            self._stack_output(obs),
            self._stack_output(rewards),
            self._stack_output(terminated),
            self._stack_output(truncated),
            dones if isinstance(dones, np.ndarray) else np.array([dones]),
            infos,
        )

    def _stack_output(self, output: Dict[str, Any]) -> np.ndarray:
        """Stacks the output of the environment into a single array."""
        x0 = list(output.values())[0]
        x0 = x0 if isinstance(x0, np.ndarray) else np.array([x0])
        num_agents = len(self.possible_agents)
        if self.is_vector_env:
            # agent outputs will have shape (num_envs, *single_env_output_shape)
            out_array = np.zeros((self.num_envs * num_agents, *x0.shape[1:]), x0.dtype)
            for i in output:
                idx = self.possible_agents.index(i)
                out_array[idx::num_agents] = output[i]
            return out_array

        # otherwise, agent output will have shape (single_env_output_shape, )
        x = [output.get(i, np.zeros(x0.shape, x0.dtype)) for i in self.possible_agents]
        return np.stack(x, axis=0)
