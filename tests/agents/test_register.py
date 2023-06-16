"""Test that `posggym_agents.register` works as expected.

Adapted from:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_register.py

"""
import re
from typing import Any, Dict, Optional

import posggym.agents as pga
import pytest
from posggym import error
from posggym.agents.registration import get_env_args_id


@pytest.mark.parametrize(
    "env_id, env_args, policy_name, version",
    [
        (
            "MyAwesomeEnv-v0",
            {"k1": 1, "k2": (2, 3)},
            "MyAwesomePolicy",
            0,
        ),
        ("MyAwesomeEnv-v0", None, "MyAwesomePolicy", 0),
        (None, None, "MyAwesomePolicy", 0),
        (None, None, "MyAwesomePolicy", None),
        (None, None, "MyAwesomePolicy-vfinal", 0),
        (None, None, "MyAwesomePolicy-vfinal", None),
        (None, None, "MyAwesomePolicy--", None),
        (None, None, "MyAwesomePolicy-v", None),
    ],
)
def test_register(
    env_id: Optional[str],
    env_args: Optional[Dict[str, Any]],
    policy_name: str,
    version: Optional[int],
):
    pga.register(
        policy_name=policy_name,
        entry_point="no-entry-point",
        version=version,
        env_id=env_id,
        env_args=env_args,
    )

    policy_id = f"{policy_name}"
    env_args_id = None if env_args is None else get_env_args_id(env_args)
    if env_args is not None:
        policy_id = f"{env_args_id}/{policy_id}"
    if env_id:
        policy_id = f"{env_id}/{policy_id}"
    if version is not None:
        policy_id = f"{policy_id}-v{version}"

    assert policy_id in pga.registry

    spec = pga.spec(policy_id)
    assert spec.id == policy_id
    assert spec.policy_name == policy_name
    assert spec.version == version
    assert spec.env_id == env_id
    assert spec.env_args == env_args
    assert spec.env_args_id == env_args_id

    del pga.registry[policy_id]


@pytest.mark.parametrize(
    "env_id, env_args, policy_name, version",
    [
        (None, None, "“Random”", 0),
        (None, None, "MyNotSoAwesomePolicy\n", None),
        ("MyEnvID\n", None, "MyNotSoAwesomePolicy", None),
    ],
)
def test_register_error(
    env_id: Optional[str],
    env_args: Optional[Dict[str, Any]],
    policy_name: str,
    version: Optional[int],
):
    with pytest.raises(error.Error, match="^Malformed policy ID:"):
        pga.register(
            policy_name=policy_name,
            entry_point="no-entry-point",
            version=version,
            env_id=env_id,
            env_args=env_args,
        )


@pytest.mark.parametrize(
    "env_id, env_args, policy_name, version",
    [
        (None, {"k1": 1, "k2": 2}, "MyNotSoAwesomePolicy", None),
    ],
)
def test_register_error2(
    env_id: Optional[str],
    env_args: Optional[Dict[str, Any]],
    policy_name: str,
    version: Optional[int],
):
    with pytest.raises(error.Error, match="^Cannot create policy ID."):
        pga.register(
            policy_name=policy_name,
            entry_point="no-entry-point",
            version=version,
            env_id=env_id,
            env_args=env_args,
        )


def test_register_versioned_unversioned():
    # Register versioned then unversioned
    versioned_policy_id = "MyPolicy-v0"
    pga.register(policy_name="MyPolicy", entry_point="no-entry-point", version=0)
    assert pga.spec(versioned_policy_id).id == versioned_policy_id

    unversioned_policy_id = "MyPolicy"
    with pytest.raises(
        error.PolicyRegistrationError,
        match=re.escape(
            "Can't register the unversioned policy `MyPolicy` when the versioned"
            " policy `MyPolicy-v0` of the same name already exists."
        ),
    ):
        pga.register(policy_name="MyPolicy", entry_point="no-entry-point", version=None)

    # Clean everything
    del pga.registry[versioned_policy_id]

    # Register unversioned then versioned
    pga.register(policy_name="MyPolicy", entry_point="no-entry-point", version=None)
    assert pga.spec(unversioned_policy_id).id == unversioned_policy_id
    with pytest.raises(
        error.PolicyRegistrationError,
        match=re.escape(
            "Can't register the versioned policy `MyPolicy-v0` when the "
            "unversioned policy `MyPolicy` of the same name already exists."
        ),
    ):
        pga.register(policy_name="MyPolicy", entry_point="no-entry-point", version=0)

    # Clean everything
    del pga.registry[unversioned_policy_id]
