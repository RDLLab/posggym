"""Tests for synchronized vectorized environment.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/main/tests/vector/test_sync_vector_env.py
"""

import numpy as np
import posggym
import pytest
from gymnasium import spaces
from posggym.vector.sync_vector_env import SyncVectorEnv


def make_env(env_name, seed, **kwargs):
    def _make():
        env = posggym.make(env_name, disable_env_checker=True, **kwargs)
        for act_space in env.action_spaces.values():
            act_space.seed(seed)
        env.reset(seed=seed)
        return env

    return _make


def test_create_sync_vector_env():
    env_fns = [make_env("MultiAccessBroadcastChannel-v0", i) for i in range(8)]
    env = SyncVectorEnv(env_fns)
    env.close()
    assert env.num_envs == 8


def test_discrete_action_space_sync_vector_env():
    env_fns = [make_env("Driving-v1", i) for i in range(8)]
    env = SyncVectorEnv(env_fns)
    env.close()

    assert all(
        isinstance(act_space, spaces.Discrete)
        for act_space in env.single_action_spaces.values()
    )
    assert all(
        isinstance(act_space, spaces.MultiDiscrete)
        for act_space in env.action_spaces.values()
    )


def test_reset_sync_vector_env():
    env_fns = [make_env("DrivingContinuous-v0", i) for i in range(8)]
    env = SyncVectorEnv(env_fns)
    observations, infos = env.reset()
    env.close()

    assert len(observations) == len(env.possible_agents)
    assert len(infos) == len(env.possible_agents)

    for agent_id in env.possible_agents:
        assert agent_id in observations
        assert agent_id in infos
        obs_i = observations[agent_id]
        info_i = infos[agent_id]

        assert isinstance(env.observation_spaces[agent_id], spaces.Box)
        assert isinstance(obs_i, np.ndarray)
        assert obs_i.shape == env.observation_spaces[agent_id].shape
        assert obs_i.shape == (8,) + env.single_observation_spaces[agent_id].shape
        assert obs_i.dtype == env.observation_spaces[agent_id].dtype

        assert isinstance(info_i, dict)
        for v in info_i.values():
            assert len(v) == 8


@pytest.mark.parametrize("use_single_action_space", [True, False])
def test_step_sync_vector_env(use_single_action_space):
    env_fns = [make_env("DrivingContinuous-v0", i) for i in range(8)]
    env = SyncVectorEnv(env_fns)
    observations, infos = env.reset()

    assert all(
        isinstance(act_space, spaces.Box) for act_space in env.action_spaces.values()
    )
    assert all(
        isinstance(act_space, spaces.Box)
        for act_space in env.single_action_spaces.values()
    )

    if use_single_action_space:
        actions = {
            i: np.stack([act_space.sample() for _ in range(8)])
            for i, act_space in env.single_action_spaces.items()
        }
    else:
        actions = {i: act_space.sample() for i, act_space in env.action_spaces.items()}

    observations, rewards, terminations, truncations, all_done, infos = env.step(
        actions
    )

    env.close()

    assert len(observations) == len(env.possible_agents)
    assert len(rewards) == len(env.possible_agents)
    assert len(terminations) == len(env.possible_agents)
    assert len(truncations) == len(env.possible_agents)
    assert len(all_done) == 8
    assert len(infos) == len(env.possible_agents)

    for i in env.possible_agents:
        assert isinstance(env.observation_spaces[i], spaces.Box)
        assert isinstance(observations[i], np.ndarray)
        assert observations[i].shape == env.observation_spaces[i].shape
        assert observations[i].shape == (8,) + env.single_observation_spaces[i].shape
        assert observations[i].dtype == env.observation_spaces[i].dtype

        assert isinstance(rewards[i], np.ndarray)
        assert isinstance(rewards[i][0], (float, np.floating))
        assert rewards[i].shape == (8,)

        assert isinstance(terminations[i], np.ndarray)
        assert terminations[i].dtype == np.bool_
        assert terminations[i].shape == (8,)

        assert isinstance(truncations[i], np.ndarray)
        assert truncations[i].dtype == np.bool_
        assert truncations[i].shape == (8,)

        assert isinstance(infos[i], dict)
        for v in infos[i].values():
            assert len(v) == 8

    assert isinstance(all_done, np.ndarray)
    assert all_done.dtype == np.bool_
    assert all_done.shape == (8,)


def test_call_sync_vector_env():
    env_fns = [make_env("Driving-v1", i, render_mode="rgb_array") for i in range(4)]

    env = SyncVectorEnv(env_fns)
    env.reset()
    images = env.call("render")
    agents = env.agents
    states = env.state

    env.close()

    assert isinstance(images, tuple)
    assert len(images) == 4
    for i in range(4):
        assert isinstance(images[i][0], np.ndarray)

    assert isinstance(agents, tuple)
    assert len(agents) == 4
    for i in range(4):
        assert isinstance(agents[i], list)
        assert agents[i] == list(env.possible_agents)

    assert isinstance(states, tuple)
    assert len(states) == 4
    for i in range(4):
        assert isinstance(states[i], tuple)
