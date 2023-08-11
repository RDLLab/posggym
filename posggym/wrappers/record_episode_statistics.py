"""Wrapper for tracking cumulative rewards and episode lengths of an environment.

Adapted from gymnasium wrapper:
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/record_episode_statistics.py

"""
import time
from collections import deque
from typing import Dict

import numpy as np

import posggym


class RecordEpisodeStatistics(posggym.Wrapper):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.

    Attributes
    ----------
    episode_count : int
        The number of episodes that have been recorded.
    return_queue : deque
        A queue of the last ``deque_size`` episode returns. Each entry is a dictionary
        mapping agent ids to the cumulative reward of the respective agent for an
        episode.
    length_queue : deque
        A queue of the last ``deque_size`` episode lengths. Each entry is a dictionary
        mapping agent ids to the episode length of the respective agent for an episode.

    Arguments
    ---------
    env : posggym.Env
        The environment to apply the wrapper
    deque_size : int
        The size of the buffer for storing the previous episode statistics.

    Note
    ----
    This implementation is based on the similar Gymnasium wrapper:
    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/record_episode_statistics.py

    """

    def __init__(self, env: posggym.Env, deque_size: int = 100):
        super().__init__(env)
        self._deque_size = deque_size

        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_count = 0
        self.episode_start_times: np.ndarray = np.zeros(self.num_envs, np.float32)
        self.episode_returns: Dict[str, np.ndarray] = {
            i: np.zeros(self.num_envs, np.float32) for i in self.possible_agents
        }
        self.episode_lengths: Dict[str, np.ndarray] = {
            i: np.zeros(self.num_envs, np.int32) for i in self.possible_agents
        }
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_start_times = np.full(
            self.num_envs, time.perf_counter(), dtype=np.float32
        )
        self.episode_returns = {
            i: np.zeros(self.num_envs, np.float32) for i in self.possible_agents
        }
        self.episode_lengths = {
            i: np.zeros(self.num_envs, np.int32) for i in self.possible_agents
        }
        return obs, info

    def step(self, actions):
        obs, rewards, terminated, truncated, dones, infos = self.env.step(actions)

        for i in rewards:
            self.episode_returns[i] += rewards[i]
            self.episode_lengths[i] += 1

        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError(
                    "Attempted to add episode statistics when they already exist."
                )

            for i in self.possible_agents:
                infos[i]["episode"] = {
                    "r": np.where(dones, self.episode_returns[i], 0.0),
                    "l": np.where(dones, self.episode_lengths[i], 0.0),
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                    ),
                }
                if self.is_vector_env:
                    infos[i]["_episode"] = np.where(dones, True, False)

            self.episode_count += num_dones
            self.episode_start_times[dones] = time.perf_counter()
            for env_idx, d in enumerate([dones] if isinstance(dones, bool) else dones):
                if not d:
                    continue
                env_episode_returns = {
                    i: self.episode_returns[i][env_idx] for i in self.possible_agents
                }
                env_episode_lengths = {
                    i: self.episode_lengths[i][env_idx] for i in self.possible_agents
                }
                self.return_queue.append(env_episode_returns)
                self.length_queue.append(env_episode_lengths)

                for i in self.possible_agents:
                    self.episode_returns[i][env_idx] = 0.0
                    self.episode_lengths[i][env_idx] = 0

        return obs, rewards, terminated, truncated, dones, infos
