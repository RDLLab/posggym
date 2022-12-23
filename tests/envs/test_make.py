"""Tests that `posggym.make` works as expected.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_make.py
"""
import re
import warnings
from copy import deepcopy

import numpy as np
import pytest

import posggym
from posggym.envs.classic import mabc
from posggym.wrappers import OrderEnforcing, TimeLimit
from tests.wrappers.utils import has_wrapper


@pytest.fixture(scope="function")
def register_make_testing_envs():
    """Registers testing envs for `posggym.make`"""
    posggym.register(
        "RegisterDuringMakeEnv-v0",
        entry_point="tests.envs.utils_envs:RegisterDuringMakeEnv",
    )

    posggym.register(
        id="test.ArgumentEnv-v0",
        entry_point="tests.envs.utils_envs:ArgumentEnv",
        kwargs={
            "arg1": "arg1",
            "arg2": "arg2",
        },
    )

    posggym.register(
        id="test/NoHuman-v0",
        entry_point="tests.envs.utils_envs:NoHuman",
    )

    posggym.register(
        id="test/NoHumanNoRGB-v0",
        entry_point="tests.envs.utils_envs:NoHumanNoRGB",
    )

    yield

    del posggym.envs.registration.registry["RegisterDuringMakeEnv-v0"]
    del posggym.envs.registration.registry["test.ArgumentEnv-v0"]
    del posggym.envs.registration.registry["test/NoHuman-v0"]
    del posggym.envs.registration.registry["test/NoHumanNoRGB-v0"]


def test_make():
    env = posggym.make("MABC-v0")
    assert env.spec is not None
    assert env.spec.id == "MABC-v0"
    assert isinstance(env.unwrapped, mabc.MABCEnv)
    env.close()


# TODO Need env with v1
# def test_make_deprecated():
#     with warnings.catch_warnings(record=True):
#         with pytest.raises(
#             posggym.error.Error,
#             match=re.escape(
#                 "Environment version v0 for `Humanoid` is deprecated. Please use "
#                 "`Humanoid-v4` instead."
#             ),
#         ):
#             posggym.make("Humanoid-v0")


def test_make_max_episode_steps(register_make_testing_envs):
    # Default, uses the spec's
    env = posggym.make("CartPole-v1", disable_env_checker=True)
    assert has_wrapper(env, TimeLimit)
    assert env.spec is not None
    assert (
        env.spec.max_episode_steps
        == posggym.envs.registry["CartPole-v1"].max_episode_steps
    )
    env.close()

    # Custom max episode steps
    env = posggym.make("CartPole-v1", max_episode_steps=100, disable_env_checker=True)
    assert has_wrapper(env, TimeLimit)
    assert env.spec is not None
    assert env.spec.max_episode_steps == 100
    env.close()

    # Env spec has no max episode steps
    assert posggym.spec("test.ArgumentEnv-v0").max_episode_steps is None
    env = posggym.make(
        "test.ArgumentEnv-v0", arg1=None, arg2=None, arg3=None, disable_env_checker=True
    )
    assert has_wrapper(env, TimeLimit) is False
    env.close()
