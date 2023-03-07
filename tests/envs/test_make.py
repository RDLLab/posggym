"""Tests that `posggym.make` works as expected.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_make.py
"""
import re
import warnings
from copy import deepcopy

import pytest

import posggym
from posggym.envs.classic import mabc
from posggym.wrappers import OrderEnforcing, PassiveEnvChecker, TimeLimit
from tests.envs.test_envs import PASSIVE_CHECK_IGNORE_WARNING
from tests.envs.utils import all_testing_env_specs
from tests.envs.utils_envs import ArgumentEnv, RegisterDuringMakeEnv
from tests.wrappers.utils import has_wrapper


@pytest.fixture(scope="function")
def register_make_testing_envs():
    """Registers testing envs for `posggym.make`"""
    posggym.register(
        "DummyEnv-v0",
        entry_point="tests.envs.utils_envs:DummyEnv",
    )

    posggym.register(
        "DummyStepLimitEnv-v0",
        entry_point="tests.envs.utils_envs:DummyEnv",
        max_episode_steps=100,
    )

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

    del posggym.envs.registration.registry["DummyEnv-v0"]
    del posggym.envs.registration.registry["DummyStepLimitEnv-v0"]
    del posggym.envs.registration.registry["RegisterDuringMakeEnv-v0"]
    del posggym.envs.registration.registry["test.ArgumentEnv-v0"]
    del posggym.envs.registration.registry["test/NoHuman-v0"]
    del posggym.envs.registration.registry["test/NoHumanNoRGB-v0"]


def test_make():
    env = posggym.make("MultiAccessBroadcastChannel-v0", disable_env_checker=True)
    assert env.spec is not None
    assert env.spec.id == "MultiAccessBroadcastChannel-v0"
    assert isinstance(env.unwrapped, mabc.MABCEnv)
    env.close()


def test_make_deprecated():
    # Making environment version that is no longer supported will raise an error
    # Note: making an older version (i.e. not the latest version) will only raise a
    #       warning if the older version is still supported (i.e. is in the registry)
    posggym.register(
        "DummyEnv-v1",
        entry_point="tests.envs.utils_envs:DummyEnv",
    )

    with warnings.catch_warnings(record=True), pytest.raises(
        posggym.error.Error,
        match=re.escape(
            "Environment version v0 for `DummyEnv` is deprecated. Please use "
            "`DummyEnv-v1` instead."
        ),
    ):
        posggym.make("DummyEnv-v0", disable_env_checker=True)

    del posggym.envs.registration.registry["DummyEnv-v1"]


def test_make_max_episode_steps(register_make_testing_envs):
    # Default, uses the spec's step limit
    env = posggym.make("DummyStepLimitEnv-v0", disable_env_checker=True)
    assert has_wrapper(env, TimeLimit)
    assert env.spec is not None
    assert (
        env.spec.max_episode_steps
        == posggym.envs.registry["DummyStepLimitEnv-v0"].max_episode_steps
    )
    env.close()

    # Custom max episode steps
    env = posggym.make(
        "MultiAccessBroadcastChannel-v0",
        max_episode_steps=100,
        disable_env_checker=True,
    )
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


def test_make_disable_env_checker():
    """Tests that `posggym.make` disables env checker correctly.

    Specifically only when `posggym.make(..., disable_env_checker=False)`.
    """
    spec = deepcopy(posggym.spec("MultiAccessBroadcastChannel-v0"))

    # Test with spec disable env checker
    spec.disable_env_checker = False
    env = posggym.make(spec)
    assert has_wrapper(env, PassiveEnvChecker)
    env.close()

    # Test with overwritten spec using make disable env checker
    assert spec.disable_env_checker is False
    env = posggym.make(spec, disable_env_checker=True)
    assert has_wrapper(env, PassiveEnvChecker) is False
    env.close()

    # Test with spec enabled disable env checker
    spec.disable_env_checker = True
    env = posggym.make(spec)
    assert has_wrapper(env, PassiveEnvChecker) is False
    env.close()

    # Test with overwritten spec using make disable env checker
    assert spec.disable_env_checker is True
    env = posggym.make(spec, disable_env_checker=False)
    assert has_wrapper(env, PassiveEnvChecker)
    env.close()


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_passive_checker_wrapper_warnings(spec):
    with warnings.catch_warnings(record=True) as caught_warnings:
        env = posggym.make(spec)  # disable_env_checker=False
        env.reset()
        env.step({i: env.action_spaces[i].sample() for i in env.agents})
        env.close()

    for warning in caught_warnings:
        if not any(
            warning.message.args[0].startswith(msg)
            for msg in PASSIVE_CHECK_IGNORE_WARNING
        ):
            raise posggym.error.Error(f"Unexpected warning: {warning.message}")


def test_make_order_enforcing():
    """Checks that gym.make wrappers the environment with the OrderEnforcing wrapper."""
    assert all(spec.order_enforce is True for spec in all_testing_env_specs)

    env = posggym.make("MultiAccessBroadcastChannel-v0", disable_env_checker=True)
    assert has_wrapper(env, OrderEnforcing)
    # We can assume that there all other specs will also have the order enforcing
    env.close()

    posggym.register(
        id="test.OrderlessArgumentEnv-v0",
        entry_point="tests.envs.utils_envs:ArgumentEnv",
        order_enforce=False,
        kwargs={"arg1": None, "arg2": None, "arg3": None},
    )

    env = posggym.make("test.OrderlessArgumentEnv-v0", disable_env_checker=True)
    assert has_wrapper(env, OrderEnforcing) is False
    env.close()


def test_make_render_mode(register_make_testing_envs):
    env = posggym.make("MultiAccessBroadcastChannel-v0", disable_env_checker=True)
    assert env.render_mode is None
    env.close()

    # Make sure that render_mode is applied correctly
    env = posggym.make(
        "MultiAccessBroadcastChannel-v0", render_mode="ansi", disable_env_checker=True
    )
    assert env.render_mode == "ansi"
    env.reset()
    # Since MultiAccessBroadcastChannel env is action first
    env.step({i: act_space.sample() for i, act_space in env.action_spaces.items()})
    render = env.render()
    # Make sure that the `render` method does what is supposed to
    assert isinstance(render, str)
    env.close()

    env = posggym.make(
        "MultiAccessBroadcastChannel-v0", render_mode=None, disable_env_checker=True
    )
    assert env.render_mode is None
    valid_render_modes = env.metadata["render_modes"]
    env.close()

    assert len(valid_render_modes) > 0
    with warnings.catch_warnings(record=True) as caught_warnings:
        env = posggym.make(
            "MultiAccessBroadcastChannel-v0",
            render_mode=valid_render_modes[0],
            disable_env_checker=True,
        )
        assert env.render_mode == valid_render_modes[0]
        env.close()

    for warning in caught_warnings:
        raise posggym.error.Error(f"Unexpected warning: {warning.message}")


# HumanRendering wrapper not currently supported
# Add this test when it is
# def test_make_human_rendering(register_make_testing_envs):
#     # Make sure that native rendering is used when possible
#     env = posggym.make("MultiAccessBroadcastChannel-v0", render_mode="human")
#     assert not has_wrapper(env, HumanRendering)  # Should use native human-rendering
#     assert env.render_mode == "human"
#     env.close()

#     with pytest.warns(
#         UserWarning,
#         match=re.escape(
#             "You are trying to use 'human' rendering for an environment that doesn't "
#             "natively support it. The HumanRendering wrapper is being applied to "
#             " your environment."
#         ),
#     ):
#         # Make sure that `HumanRendering` is applied here
#         env = posggym.make(
#             "test/NoHuman-v0", render_mode="human"
#         )  # This environment doesn't use native rendering
#         assert has_wrapper(env, HumanRendering)
#         assert env.render_mode == "human"
#         env.close()


def test_make_kwargs(register_make_testing_envs):
    env = posggym.make(
        "test.ArgumentEnv-v0",
        arg2="override_arg2",
        arg3="override_arg3",
        disable_env_checker=True,
    )
    assert env.spec is not None
    assert env.spec.id == "test.ArgumentEnv-v0"
    assert isinstance(env.unwrapped, ArgumentEnv)
    assert env.arg1 == "arg1"
    assert env.arg2 == "override_arg2"
    assert env.arg3 == "override_arg3"
    env.close()


def test_import_module_during_make(register_make_testing_envs):
    # Tests that the module of a custom environment is imported during registration
    # This ensures the custom env is registered before attempting to be made (assuming
    # custom environment module registers the custom env in the posggym registry during
    # module initialization.)
    env = posggym.make(
        "tests.envs.utils:RegisterDuringMakeEnv-v0", disable_env_checker=True
    )
    assert isinstance(env.unwrapped, RegisterDuringMakeEnv)
    env.close()
