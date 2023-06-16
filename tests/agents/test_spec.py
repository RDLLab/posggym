"""Tests that posggym_agents.spec works as expected.

Reference:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_spec.py
"""
import re

import posggym
import posggym.agents as pga
import pytest
from posggym import error
from posggym.agents.random_policies import DiscreteFixedDistributionPolicy, RandomPolicy
from posggym.agents.registration import get_env_args_id


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
    """Registers testing policies for `posggym_agents.make`"""
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


def test_generic_spec():
    spec = pga.spec("Random-v0")
    assert spec.id == "Random-v0"
    assert spec is pga.registry["Random-v0"]


def test_generic_env_spec():
    spec = pga.spec("MultiAccessBroadcastChannel-v0/Random-v0")
    assert spec.id == "Random-v0"
    assert spec is pga.registry["Random-v0"]


def test_env_spec(register_make_testing_policies):
    policy_id = f"{TEST_ENV_ID}/EnvTestPolicy-v5"
    spec = pga.spec(policy_id)
    assert spec.id == policy_id
    assert spec is pga.registry[policy_id]


def test_generic_spec_kwargs():
    env = posggym.make("MultiAccessBroadcastChannel-v0")
    action_dist = {0: 0.3, 1: 0.7}
    policy = pga.make(
        "DiscreteFixedDistributionPolicy-v0", env.model, env.agents[0], dist=action_dist
    )
    assert policy.spec is not None
    assert policy.spec.kwargs["dist"] == action_dist


def test_generic_spec_missing_lookup(register_make_testing_policies):
    pga.register("Test1", entry_point="no-entry-point", version=0)
    pga.register("Test1", entry_point="no-entry-point", version=15)
    pga.register("Test1", entry_point="no-entry-point", version=9)
    pga.register("Other1", entry_point="no-entry-point", version=100)

    with pytest.raises(
        error.DeprecatedPolicy,
        match=re.escape(
            "Policy version v1 for `Test1` is deprecated. Please use `Test1-v15` "
            "instead."
        ),
    ):
        pga.spec("Test1-v1")

    with pytest.raises(
        error.UnregisteredPolicy,
        match=re.escape(
            "Policy version `v1000` for policy `Test1` doesn't exist. "
            "It provides versioned policies: [ `v0`, `v9`, `v15` ]."
        ),
    ):
        pga.spec("Test1-v1000")

    with pytest.raises(
        error.UnregisteredPolicy,
        match=re.escape("Policy Unknown1 doesn't exist. "),
    ):
        pga.spec("Unknown1-v1")


def test_env_spec_missing_lookup():
    env_id = "MultiAccessBroadcastChannel-v0"
    pga.register("Test1", entry_point="no-entry-point", version=0, env_id=env_id)
    pga.register("Test1", entry_point="no-entry-point", version=15, env_id=env_id)
    pga.register("Test1", entry_point="no-entry-point", version=9, env_id=env_id)
    pga.register("Other1", entry_point="no-entry-point", version=100, env_id=env_id)

    with pytest.raises(
        error.DeprecatedPolicy,
        match=re.escape(
            f"Policy version v1 for `{env_id}/Test1` is deprecated. Please use "
            f"`{env_id}/Test1-v15` instead."
        ),
    ):
        pga.spec(f"{env_id}/Test1-v1")

    with pytest.raises(
        error.UnregisteredPolicy,
        match=re.escape(
            f"Policy version `v1000` for policy `{env_id}/Test1` doesn't exist. "
            "It provides versioned policies: [ `v0`, `v9`, `v15` ]."
        ),
    ):
        pga.spec(f"{env_id}/Test1-v1000")

    with pytest.raises(
        error.UnregisteredPolicy,
        match=re.escape(f"Policy Unknown1 doesn't exist for env ID {env_id}. "),
    ):
        pga.spec(f"{env_id}/Unknown1-v1")


def test_spec_malformed_lookup():
    expected_error_msg = (
        "Malformed policy ID: “Random-v0”. Currently all IDs must be of the form "
        "[env_id/][env_args_id/](policy_name)-v(version) (env_id and env_args_id may "
        "be optional, depending on the policy)."
    )
    with pytest.raises(
        error.Error,
        match=f"^{re.escape(expected_error_msg)}$",
    ):
        pga.spec("“Random-v0”")


def test_spec_default_lookups():
    env_id = "MultiAccessBroadcastChannel-v0"
    pga.register("Test3", entry_point="no-entry-point", version=None, env_id=env_id)
    pga.register("Test4", entry_point="no-entry-point", version=None, env_id=None)

    with pytest.raises(
        error.DeprecatedPolicy,
        match=re.escape(
            f"Policy version `v0` for policy `{env_id}/Test3` doesn't exist. "
            f"It provides the default version {env_id}/Test3`."
        ),
    ):
        pga.spec(f"{env_id}/Test3-v0")

    assert pga.spec(f"{env_id}/Test3") is not None

    with pytest.raises(
        error.DeprecatedPolicy,
        match=re.escape(
            "Policy version `v0` for policy `Test4` doesn't exist. "
            "It provides the default version Test4`."
        ),
    ):
        pga.spec("Test4-v0")

    assert pga.spec("Test4") is not None

    with pytest.raises(
        error.DeprecatedPolicy,
        match=re.escape(
            "Policy version `v0` for policy `Test4` doesn't exist. "
            "It provides the default version Test4`."
        ),
    ):
        pga.spec(f"{env_id}/Test4-v0")

    assert pga.spec(f"{env_id}/Test4") is not None
