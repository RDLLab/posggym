import numpy as np

import posggym


class BatchTimeLimit(posggym.Wrapper):
    """Wraps environment batch to enforce environment time limit.

    This wrapper will issue a `truncated` signal in the :meth:`step` method for any
    agents that have not already reached a terminal state by the time a maximum number
    of timesteps is exceeded. It will also signal that the episode is `done` for all
    agents in all environments when their respective time limits are exceeded.

    Arguments:
    ---------
    env : posggym.Env
        The environment batch to apply the wrapper
    max_episode_steps : int, optional
        The maximum length of episode before it is truncated. If None then will not
        truncate episodes.
    """

    def __init__(self, env: posggym.Env, max_episode_steps: int | None = None) -> None:
        super().__init__(env)

        assert hasattr(self.env, "batch_size")
        self.num_envs = self.env.batch_size  # type: ignore

        if max_episode_steps is None and self.env.spec is not None:
            assert env.spec is not None
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = np.zeros(self.num_envs, dtype=int)
        self._terminated_agents = [set() for _ in range(self.num_envs)]

    def step(self, actions):
        """Take a step in all batched environments with time limit enforcement."""
        obs, rewards, terminated, truncated, done, info = self.env.step(actions)

        for env_idx in range(self.num_envs):
            self._elapsed_steps[env_idx] += 1

            # Check if max steps are reached for the environment
            if self._elapsed_steps[env_idx] >= self._max_episode_steps:
                for agent, agent_truncated in truncated.items():
                    if agent not in self._terminated_agents[env_idx]:
                        agent_truncated[env_idx] = True
                done[env_idx] = True
            else:
                for agent, agent_terminated in terminated.items():
                    # If the agent has terminated in this environment, mark it
                    if agent_terminated[env_idx]:
                        self._terminated_agents[env_idx].add(agent)

        return obs, rewards, terminated, truncated, done, info

    def reset(self, **kwargs):
        self._elapsed_steps.fill(0)
        obs, info = self.env.reset(**kwargs)
        self._terminated_agents = [set() for _ in range(self.num_envs)]
        return obs, info
