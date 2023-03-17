"""Tests for TimeLimit wrapper.

Adapted from:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/wrappers/test_time_limit.py

"""
import pytest

import posggym
from posggym.envs.classic.mabc import MABCEnv
from posggym.wrappers import TimeLimit


def test_time_limit_reset_info():
    env = posggym.make("MultiAccessBroadcastChannel-v0", disable_env_checker=True)
    env = TimeLimit(env)
    obs, info = env.reset()
    assert env.is_symmetric
    assert isinstance(obs, dict)
    assert isinstance(info, dict)


@pytest.mark.parametrize("double_wrap", [False, True])
def test_time_limit_wrapper(double_wrap):
    # The mabc env does not terminate by default
    # so we are sure termination is only due to timeout
    env = MABCEnv()
    max_episode_length = 20
    env = TimeLimit(env, max_episode_length)
    if double_wrap:
        env = TimeLimit(env, max_episode_length)
    env.reset()
    done = False
    terminated = {i: False for i in env.agents}
    truncated = {i: False for i in env.agents}
    n_steps = 0
    info = {}
    while not done:
        n_steps += 1
        _, _, terminated, truncated, done, info = env.step(
            {i: env.action_spaces[i].sample() for i in env.agents}
        )

    assert n_steps == max_episode_length
    assert all(list(truncated.values()))
    assert not any(list(terminated.values()))


@pytest.mark.parametrize("double_wrap", [False, True])
def test_termination_on_last_step(double_wrap):
    # Special case: termination at the last timestep
    # Truncation due to timeout also happens at the same step
    env = MABCEnv()

    def patched_step(_actions):
        return (
            {i: 0 for i in env.agents},
            {i: 0.0 for i in env.agents},
            {i: True for i in env.agents},
            {i: False for i in env.agents},
            True,
            {i: {} for i in env.agents},
        )

    env.step = patched_step

    max_episode_length = 1
    env = TimeLimit(env, max_episode_length)
    if double_wrap:
        env = TimeLimit(env, max_episode_length)
    env.reset()
    _, _, terminated, truncated, done, _ = env.step(
        {i: env.action_spaces[i].sample() for i in env.agents}
    )
    assert done is True
    assert all(terminated.values())
    assert all(truncated.values())
