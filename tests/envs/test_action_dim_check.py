"""Tests that environments handle out-of-bounds actions correctly.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_action_dim_check.py
"""
from typing import Dict, Tuple, Union

import numpy as np
import pytest
from gymnasium import spaces
from tests.envs.utils import all_testing_initialised_envs

import posggym

DISCRETE_ENVS = list(
    filter(
        lambda env: all(
            isinstance(act_space, spaces.Discrete)
            for act_space in env.action_spaces.values()
        ),
        all_testing_initialised_envs,
    )
)


@pytest.mark.parametrize(
    "env",
    DISCRETE_ENVS,
    ids=[env.spec.id for env in DISCRETE_ENVS if env.spec is not None],
)
def test_discrete_actions_out_of_bound(env: posggym.Env):
    """Test out of bound actions in Discrete action_space.

    In discrete action_space environments, `out-of-bound` actions are not allowed and
    should raise an exception.
    """
    assert all(
        isinstance(act_space, spaces.Discrete)
        for act_space in env.action_spaces.values()
    )
    upper_bounds = {
        i: env.action_spaces[i].start + env.action_spaces[i].n  # type: ignore
        for i in env.agents
    }

    env.reset()
    with pytest.raises(Exception):
        env.step(upper_bounds)

    env.close()


BOX_ENVS = list(
    filter(
        lambda env: all(
            isinstance(act_space, spaces.Box)
            for act_space in env.action_spaces.values()
        ),
        all_testing_initialised_envs,
    )
)
OOB_VALUE = 100


def tuple_equal(
    a: Tuple[Union[int, np.ndarray, float]], b: Tuple[Union[int, np.ndarray, float]]
) -> bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if isinstance(a[i], np.ndarray) and isinstance(b[i], np.ndarray):
            if not np.array_equal(a[i], b[i]):
                return False
        elif a[i] != b[i]:
            return False
    return True


@pytest.mark.parametrize(
    "env", BOX_ENVS, ids=[env.spec.id for env in BOX_ENVS if env.spec is not None]
)
def test_box_actions_out_of_bound(env: posggym.Env):
    """Test out of bound actions in Box action_space.

    Environments with Box actions spaces perform clipping inside `step`.
    The expected behaviour is that an action `out-of-bound` has the same effect of an
    action with value exactly at the upper (or lower) bound.
    """
    env.reset(seed=42)

    assert env.spec is not None
    oob_env = posggym.make(env.spec.id, disable_env_checker=True)
    oob_env.reset(seed=42)

    action_spaces: Dict[str, spaces.Box] = env.action_spaces  # type: ignore
    assert all(
        isinstance(act_space, spaces.Box) for act_space in action_spaces.values()
    )

    dtypes = {i: action_spaces[i].dtype for i in env.agents}
    upper_bounds = {i: action_spaces[i].high for i in env.agents}
    lower_bounds = {i: action_spaces[i].low for i in env.agents}

    if all(np.all(action_spaces[i].bounded_above) for i in env.agents):
        obs, _, _, _, _, _ = env.step(upper_bounds)
        oob_actions = {
            i: np.cast[dtypes[i]](upper_bounds[i] + OOB_VALUE) for i in upper_bounds
        }

        assert all(np.all(oob_actions[i] > upper_bounds[i]) for i in upper_bounds)
        oob_obs, _, _, _, _, _ = oob_env.step(oob_actions)

        assert all(tuple_equal(obs[i], oob_obs[i]) for i in upper_bounds)

    if all(np.all(action_spaces[i].bounded_below) for i in env.agents):
        obs, _, _, _, _, _ = env.step(lower_bounds)

        oob_actions = {
            i: np.cast[dtypes[i]](lower_bounds[i] - OOB_VALUE) for i in lower_bounds
        }
        assert all(np.all(oob_actions[i] < lower_bounds[i]) for i in lower_bounds)
        oob_obs, _, _, _, _, _ = oob_env.step(oob_actions)

        assert all(tuple_equal(obs[i], oob_obs[i]) for i in lower_bounds)

    env.close()
