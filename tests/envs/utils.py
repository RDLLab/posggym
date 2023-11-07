"""Finds all the specs that we can test with.

Reference:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/utils.py
"""
from typing import List, Optional

import numpy as np
import posggym
from posggym import logger
from posggym.envs.registration import EnvSpec

from tests.conftest import env_id_prefix


def try_make_env(env_spec: EnvSpec) -> Optional[posggym.Env]:
    """Tries to make the environment showing if it is possible.

    Warning the environments have no wrappers, including time limit and order enforcing.
    """
    # To avoid issues with registered environments during testing, we check that the
    # spec entry points are from posggym.envs.
    if (
        isinstance(env_spec.entry_point, str)
        and "posggym.envs." in env_spec.entry_point
    ):
        try:
            return env_spec.make(disable_env_checker=True).unwrapped
        except (
            ImportError,
            posggym.error.DependencyNotInstalled,
            posggym.error.MissingArgument,
        ) as e:
            logger.warn(f"Not testing {env_spec.id} due to error: {e}")
    return None


# Tries to make all environment to test with
_all_testing_initialised_envs: List[Optional[posggym.Env]] = [
    try_make_env(env_spec)
    for env_spec in posggym.envs.registry.values()
    if env_id_prefix is None or env_spec.id.startswith(env_id_prefix)
]
all_testing_initialised_envs: List[posggym.Env] = [
    env for env in _all_testing_initialised_envs if env is not None
]

# All testing posggym environment specs
all_testing_env_specs: List[EnvSpec] = [
    env.spec for env in all_testing_initialised_envs if env.spec is not None
]


def assert_equals(a, b, prefix=None):
    """Assert equality of data structures `a` and `b`.

    Arguments
    ---------
    a:
        first data structure
    b:
        second data structure
    prefix:
        prefix for failed assertion message for types and dicts

    """
    assert type(a) == type(b), f"{prefix}Differing types: {a} and {b}"
    if isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), f"{prefix}Key sets differ: {a} and {b}"
        for k in a:
            v_a = a[k]
            v_b = b[k]
            assert_equals(v_a, v_b, prefix)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, tuple):
        for elem_from_a, elem_from_b in zip(a, b):
            assert_equals(elem_from_a, elem_from_b, prefix)
    else:
        assert a == b
