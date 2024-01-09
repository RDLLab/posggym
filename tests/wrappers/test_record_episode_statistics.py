"""Tests for RecordEpisodeStatistics wrapper.

Adapted from:
https://github.com/Farama-Foundation/Gymnasium/blob/main/tests/wrappers/test_record_episode_statistics.py

"""
import numpy as np
import posggym
import pytest
from posggym.vector.sync_vector_env import SyncVectorEnv
from posggym.wrappers import RecordEpisodeStatistics


@pytest.mark.parametrize("env_id", ["MultiAccessBroadcastChannel-v0", "Driving-v1"])
@pytest.mark.parametrize("deque_size", [2, 5])
def test_record_episode_statistics(env_id, deque_size):
    env = posggym.make(env_id, max_episode_steps=10, disable_env_checker=True)
    env = RecordEpisodeStatistics(env, deque_size)

    for _ in range(5):
        env.reset()

        assert env.episode_returns is not None and env.episode_lengths is not None
        assert len(env.episode_returns) == len(env.possible_agents)
        assert len(env.episode_lengths) == len(env.possible_agents)
        for i in env.agents:
            assert env.episode_returns[i][0] == 0.0
            assert env.episode_lengths[i][0] == 0

        assert env.spec is not None

        agent_returns = {i: 0.0 for i in env.possible_agents}
        for t in range(env.spec.max_episode_steps):
            _, rewards, _, _, done, infos = env.step(
                {i: env.action_spaces[i].sample() for i in env.agents}
            )

            for i in env.possible_agents:
                agent_returns[i] += rewards[i]

            if done:
                assert len(infos) == len(env.possible_agents)
                for i in env.possible_agents:
                    assert "episode" in infos[i]
                    assert all(
                        [item in infos[i]["episode"] for item in ["r", "l", "t"]]
                    )
                    assert np.isclose(infos[i]["episode"]["r"], agent_returns[i])
                    assert infos[i]["episode"]["l"] == t + 1
                break
            else:
                for i in env.possible_agents:
                    assert np.isclose(env.episode_returns[i][-1], agent_returns[i])
                    assert env.episode_lengths[i][-1] == t + 1

    assert len(env.return_queue) == deque_size
    assert len(env.length_queue) == deque_size

    while env.return_queue:
        episode_returns = env.return_queue.pop()
        episode_lengths = env.length_queue.pop()

        assert len(episode_returns) == len(env.possible_agents)
        assert len(episode_lengths) == len(env.possible_agents)
        for i in env.possible_agents:
            assert i in episode_returns
            assert i in episode_lengths


@pytest.mark.parametrize("num_envs", [1, 2, 4])
def test_record_episode_statistics_with_vectorenv(num_envs):
    env_id = "MultiAccessBroadcastChannel-v0"
    max_episode_steps = 10

    def _make_env():
        env = posggym.make(
            env_id,
            max_episode_steps=max_episode_steps,
            disable_env_checker=True,
        )
        env.reset(seed=num_envs)
        return env

    envs = SyncVectorEnv([_make_env for _ in range(num_envs)])
    envs = RecordEpisodeStatistics(envs)

    envs.reset()
    for _ in range(max_episode_steps + 1):
        _, _, _, _, dones, infos = envs.step(
            {i: envs.action_spaces[i].sample() for i in envs.possible_agents}
        )
        if any(dones):
            assert len(infos) == len(envs.possible_agents)
            for i in envs.possible_agents:
                assert "episode" in infos[i]
                assert "_episode" in infos[i]
                assert all(infos[i]["_episode"] == dones)
                assert all([item in infos[i]["episode"] for item in ["r", "l", "t"]])
            break
        else:
            for i in envs.possible_agents:
                if i in infos:
                    assert "episode" not in infos[i]
                    assert "_episode" not in infos[i]
