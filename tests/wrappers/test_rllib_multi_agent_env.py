"""Tests for the Rllib MultiAgentEnv wrapper."""
import warnings

import posggym
import pytest

from tests.envs.utils import all_testing_env_specs


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_make_rllib_multi_agent_env(spec):
    """Checks that posggym -> rllib MultiAgentEnv conversion works correctly.

    These tests are pretty basic, and just check envs can be loaded, wrapped in
    RllibMultiAgentEnv, reset, stepped, and closed.

    Currently it doesn't perform any rllib API checking.

    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from posggym.posggym.wrappers.rllib_env import RllibMultiAgentEnv
            from ray.rllib.utils.pre_checks.env import check_env
    except ImportError as e:
        pytest.skip(f"ray[rllib] not installed.: {str(e)}")

    env = posggym.make(spec.id, disable_env_checker=True)
    rllib_env = RllibMultiAgentEnv(env)

    check_env(rllib_env)
