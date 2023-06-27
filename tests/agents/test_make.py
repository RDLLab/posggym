"""Tests that `posggym_agents.make` works as expected.

Adapted from:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_make.py

"""
import re
import warnings

import pytest
from tests.agents.helpers import assert_equals

import posggym
import posggym.agents as pga
from posggym import error
from posggym.agents.random_policies import DiscreteFixedDistributionPolicy, RandomPolicy
from posggym.agents.registration import get_env_args_id
from posggym.agents.utils.action_distributions import DiscreteActionDistribution

TEST_ENV_ID = "MultiAccessBroadcastChannel-v0"
TEST_ENV_ID_UNV = "MultiAccessBroadcastChannel"
TEST_ENV_ARGS = {
    "num_nodes": 3,
    "fill_probs": (0.9, 0.2, 0.3),
    "init_buffer_dist": (1.0, 1.0, 1.0),
}
TEST_ENV_ARGS_ID = get_env_args_id(TEST_ENV_ARGS)


@pytest.fixture(scope="function")
def register_make_testing_policies():
    """Registers testing policies for `posggym.agents.make`"""
    pga.register(policy_name="GenericTestPolicy", entry_point=RandomPolicy, version=0)
    pga.register(
        policy_name="EnvTestPolicy",
        entry_point=RandomPolicy,
        version=1,
        env_id=TEST_ENV_ID,
    )
    pga.register(
        policy_name="EnvTestPolicy",
        entry_point=RandomPolicy,
        version=3,
        env_id=TEST_ENV_ID,
    )
    pga.register(
        policy_name="EnvTestPolicy",
        entry_point=RandomPolicy,
        version=5,
        env_id=TEST_ENV_ID,
    )
    pga.register(
        policy_name="EnvTestPolicy",
        entry_point=RandomPolicy,
        version=0,
        env_id=TEST_ENV_ID,
        env_args=TEST_ENV_ARGS,
    )

    pga.register(
        policy_name="EnvUnversionedTestPolicy",
        entry_point=RandomPolicy,
        version=None,
        env_id=TEST_ENV_ID,
    )

    pga.register(
        policy_name="GenericArgumentTestPolicy",
        entry_point=DiscreteFixedDistributionPolicy,
        version=0,
        kwargs={"dist": None},
    )
    pga.register(
        policy_name="EnvArgumentTestPolicy",
        entry_point=DiscreteFixedDistributionPolicy,
        version=0,
        env_id=TEST_ENV_ID,
        kwargs={"dist": None},
    )
    pga.register(
        policy_name="EnvArgumentTestPolicy",
        entry_point=DiscreteFixedDistributionPolicy,
        version=0,
        env_id=TEST_ENV_ID,
        env_args=TEST_ENV_ARGS,
        kwargs={"dist": None},
    )

    yield

    del pga.registry["GenericTestPolicy-v0"]
    del pga.registry[f"{TEST_ENV_ID}/EnvTestPolicy-v1"]
    del pga.registry[f"{TEST_ENV_ID}/EnvTestPolicy-v3"]
    del pga.registry[f"{TEST_ENV_ID}/EnvTestPolicy-v5"]
    del pga.registry[f"{TEST_ENV_ID}/{TEST_ENV_ARGS_ID}/EnvTestPolicy-v0"]
    del pga.registry[f"{TEST_ENV_ID}/EnvUnversionedTestPolicy"]
    del pga.registry["GenericArgumentTestPolicy-v0"]
    del pga.registry[f"{TEST_ENV_ID}/EnvArgumentTestPolicy-v0"]
    del pga.registry[f"{TEST_ENV_ID}/{TEST_ENV_ARGS_ID}/EnvArgumentTestPolicy-v0"]


def test_make_generic():
    env_id = "MultiAccessBroadcastChannel-v0"
    env = posggym.make(env_id, disable_env_checker=True)

    policy = pga.make("Random-v0", env.model, env.agents[0])
    spec = policy.spec
    assert spec is not None
    assert spec.id == "Random-v0"
    assert spec.policy_name == "Random"
    assert spec.version == 0
    assert spec.env_id is None
    assert spec.env_args is None
    assert spec.env_args_id is None

    policy = pga.make(f"{env_id}/Random-v0", env.model, env.agents[0])
    spec = policy.spec
    assert spec is not None
    assert spec.id == "Random-v0"
    assert spec.policy_name == "Random"
    assert spec.version == 0
    assert spec.env_id is None
    assert spec.env_args is None
    assert spec.env_args_id is None

    env.close()


def test_make_env_specific(register_make_testing_policies):
    env = posggym.make(TEST_ENV_ID, disable_env_checker=True)
    policy_id = f"{TEST_ENV_ID}/EnvTestPolicy-v5"
    policy = pga.make(policy_id, env.model, env.agents[0])
    spec = policy.spec
    assert spec is not None
    assert spec.id == policy_id
    assert spec.policy_name == "EnvTestPolicy"
    assert spec.version == 5
    assert spec.env_id == TEST_ENV_ID
    assert spec.env_args is None
    assert spec.env_args_id is None
    env.close()


def test_make_env_and_env_args_specific(register_make_testing_policies):
    env = posggym.make(TEST_ENV_ID, disable_env_checker=True, **TEST_ENV_ARGS)
    policy_id = f"{TEST_ENV_ID}/{TEST_ENV_ARGS_ID}/EnvTestPolicy-v0"
    policy = pga.make(policy_id, env.model, env.agents[0])
    spec = policy.spec
    assert spec is not None
    assert spec.id == policy_id
    assert spec.policy_name == "EnvTestPolicy"
    assert spec.version == 0
    assert spec.env_id == TEST_ENV_ID
    assert_equals(spec.env_args, TEST_ENV_ARGS, "env_args: ")
    assert spec.env_args_id == TEST_ENV_ARGS_ID


def test_make_policy_with_kwargs(register_make_testing_policies):
    env = posggym.make(TEST_ENV_ID, disable_env_checker=True)
    dist = DiscreteActionDistribution({0: 0.3, 1: 0.7})
    policy = pga.make(
        "GenericArgumentTestPolicy-v0",
        env.model,
        env.agents[0],
        dist=dist,
    )
    assert policy.spec is not None
    assert policy.spec.id == "GenericArgumentTestPolicy-v0"
    assert policy.spec.policy_name == "GenericArgumentTestPolicy"
    assert policy.spec.version == 0
    assert policy.spec.env_id is None
    assert policy.spec.env_args is None
    assert policy.spec.env_args_id is None
    assert_equals(policy.dist, dist)

    policy = pga.make(
        f"{TEST_ENV_ID}/GenericArgumentTestPolicy-v0",
        env.model,
        env.agents[0],
        dist=dist,
    )
    assert policy.spec is not None
    assert policy.spec.id == "GenericArgumentTestPolicy-v0"
    assert policy.spec.policy_name == "GenericArgumentTestPolicy"
    assert policy.spec.version == 0
    assert policy.spec.env_id is None
    assert policy.spec.env_args is None
    assert policy.spec.env_args_id is None
    assert_equals(policy.dist, dist)

    policy = pga.make(
        f"{TEST_ENV_ID}/EnvArgumentTestPolicy-v0",
        env.model,
        env.agents[0],
        dist=dist,
    )
    assert policy.spec is not None
    assert policy.spec.id == f"{TEST_ENV_ID}/EnvArgumentTestPolicy-v0"
    assert policy.spec.policy_name == "EnvArgumentTestPolicy"
    assert policy.spec.version == 0
    assert policy.spec.env_id == TEST_ENV_ID
    assert policy.spec.env_args is None
    assert policy.spec.env_args_id is None
    assert_equals(policy.dist, dist)

    policy = pga.make(
        f"{TEST_ENV_ID}/{TEST_ENV_ARGS_ID}/EnvArgumentTestPolicy-v0",
        env.model,
        env.agents[0],
        dist=dist,
    )
    assert policy.spec is not None
    assert (
        policy.spec.id == f"{TEST_ENV_ID}/{TEST_ENV_ARGS_ID}/EnvArgumentTestPolicy-v0"
    )
    assert policy.spec.policy_name == "EnvArgumentTestPolicy"
    assert policy.spec.version == 0
    assert policy.spec.env_id == TEST_ENV_ID
    assert_equals(policy.spec.env_args, TEST_ENV_ARGS, "env_args: ")
    assert policy.spec.env_args_id == TEST_ENV_ARGS_ID
    assert_equals(policy.dist, dist)

    env.close()


@pytest.mark.parametrize(
    "policy_id_input, policy_id_suggested",
    [
        ("random-v0", "Random"),
        ("RAnDom-v0", "Random"),
        ("Discretefixeddistributionpolicy-v10", "DiscreteFixedDistributionPolicy"),
        ("MultiAccessBroadcastChnnel-v0/EnvTestPolicy-v1", TEST_ENV_ID_UNV),
        ("MultiAccessBroadcastChnnel-v0/EnvUnversionedTestPolicy", TEST_ENV_ID_UNV),
        (f"{TEST_ENV_ID}/EnvUnversioneTestPolicy", "EnvUnversionedTestPolicy"),
        (f"{TEST_ENV_ID}/EnvTesPolicy", "EnvTestPolicy"),
    ],
)
def test_policy_suggestions(
    register_make_testing_policies, policy_id_input, policy_id_suggested
):
    env = posggym.make(TEST_ENV_ID)
    with pytest.raises(
        error.UnregisteredPolicy, match=f"Did you mean: `{policy_id_suggested}`?"
    ):
        pga.make(policy_id_input, env.model, env.agents[0])


@pytest.mark.parametrize(
    "policy_id_input, suggested_versions, default_version",
    [
        ("Random-v12", "`v0`", False),
        (f"{TEST_ENV_ID}/EnvTestPolicy-v6", "`v1`, `v3`, `v5`", False),
        (f"{TEST_ENV_ID}/EnvUnversionedTestPolicy-v6", "", True),
    ],
)
def test_env_version_suggestions(
    register_make_testing_policies,
    policy_id_input,
    suggested_versions,
    default_version,
):
    env = posggym.make(TEST_ENV_ID)
    if default_version:
        with pytest.raises(
            error.DeprecatedPolicy,
            match="It provides the default version",
        ):
            pga.make(policy_id_input, env.model, env.agents[0])
    else:
        with pytest.raises(
            error.UnregisteredPolicy,
            match=f"It provides versioned policies: \\[ {suggested_versions} \\]",
        ):
            pga.make(policy_id_input, env.model, env.agents[0])


def test_make_deprecated():
    # Making policy version that is no longer supported will raise an error
    # Note: making an older version (i.e. not the latest version) will only raise a
    #       warning if the older version is still supported (i.e. is in the registry)
    pga.register(policy_name="DummyPolicy", entry_point=RandomPolicy, version=1)

    env = posggym.make(TEST_ENV_ID)
    with warnings.catch_warnings(record=True), pytest.raises(
        error.Error,
        match=re.escape(
            "Policy version v0 for `DummyPolicy` is deprecated. Please use "
            "`DummyPolicy-v1` instead."
        ),
    ):
        pga.make("DummyPolicy-v0", env.model, env.agents[0])

    del pga.registry["DummyPolicy-v1"]


def test_make_latest_versioned_env(register_make_testing_policies):
    env = posggym.make(TEST_ENV_ID)
    with pytest.warns(
        UserWarning,
        match=re.escape(
            f"Using the latest versioned policy `{TEST_ENV_ID}/EnvTestPolicy-v5` "
            f"instead of the unversioned policy `{TEST_ENV_ID}/EnvTestPolicy`."
        ),
    ):
        policy = pga.make(f"{TEST_ENV_ID}/EnvTestPolicy", env.model, env.agents[0])
    assert policy.spec is not None
    assert policy.spec.id == f"{TEST_ENV_ID}/EnvTestPolicy-v5"
