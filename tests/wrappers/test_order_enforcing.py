"""Tests for OrderEnforcing wrapper.

Adapted from:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/wrappers/test_order_enforcing.py

"""
import pytest

import posggym
from posggym.envs.classic.mabc import MABCEnv
from posggym.error import ResetNeeded
from posggym.wrappers import OrderEnforcing
from tests.envs.utils import all_testing_env_specs
from tests.wrappers.utils import has_wrapper


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_gym_make_order_enforcing(spec):
    """Checks that posggym.make wraps environments with OrderEnforcing wrapper."""
    env = posggym.make(spec.id, disable_env_checker=True)
    assert has_wrapper(env, OrderEnforcing)


def test_order_enforcing():
    """Checks that the order enforcing wrapper works as expected.

    Raising an error before reset is called and not after.
    """
    # The reason for not using gym.make is that all environments are by default
    # wrapped in the order enforcing wrapper
    env = MABCEnv(render_mode="ansi")
    assert not has_wrapper(env, OrderEnforcing)

    # Assert that the order enforcing works for step and render before reset
    order_enforced_env = OrderEnforcing(env)
    assert order_enforced_env.has_reset is False
    with pytest.raises(ResetNeeded):
        order_enforced_env.step({i: 0 for i in env.possible_agents})
    with pytest.raises(ResetNeeded):
        order_enforced_env.render()
    assert order_enforced_env.has_reset is False

    # Assert that the Assertion errors are not raised after reset
    order_enforced_env.reset()
    assert order_enforced_env.has_reset is True
    order_enforced_env.step({i: 0 for i in env.possible_agents})
    order_enforced_env.render()
