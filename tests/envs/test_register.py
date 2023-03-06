"""Test that `posggym.register` works as expected.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_register.py

"""
import re
from typing import Optional

import pytest

import posggym


@pytest.fixture(scope="function")
def register_registration_testing_envs():
    """Register testing envs for `posggym.register`."""
    namespace = "MyAwesomeNamespace"
    versioned_name = "MyAwesomeVersionedEnv"
    unversioned_name = "MyAwesomeUnversionedEnv"
    versions = [1, 3, 5]
    for version in versions:
        env_id = f"{namespace}/{versioned_name}-v{version}"
        posggym.register(
            id=env_id,
            entry_point="tests.envs.utils_envs:ArgumentEnv",
            kwargs={
                "arg1": "arg1",
                "arg2": "arg2",
                "arg3": "arg3",
            },
        )
    posggym.register(
        id=f"{namespace}/{unversioned_name}",
        entry_point="tests.env.utils_envs:ArgumentEnv",
        kwargs={
            "arg1": "arg1",
            "arg2": "arg2",
            "arg3": "arg3",
        },
    )

    yield

    for version in versions:
        env_id = f"{namespace}/{versioned_name}-v{version}"
        del posggym.envs.registry[env_id]
    del posggym.envs.registry[f"{namespace}/{unversioned_name}"]


@pytest.mark.parametrize(
    "env_id, namespace, name, version",
    [
        (
            "MyAwesomeNamespace/MyAwesomeEnv-v0",
            "MyAwesomeNamespace",
            "MyAwesomeEnv",
            0,
        ),
        ("MyAwesomeEnv-v0", None, "MyAwesomeEnv", 0),
        ("MyAwesomeEnv", None, "MyAwesomeEnv", None),
        ("MyAwesomeEnv-vfinal-v0", None, "MyAwesomeEnv-vfinal", 0),
        ("MyAwesomeEnv-vfinal", None, "MyAwesomeEnv-vfinal", None),
        ("MyAwesomeEnv--", None, "MyAwesomeEnv--", None),
        ("MyAwesomeEnv-v", None, "MyAwesomeEnv-v", None),
    ],
)
def test_register(
    env_id: str, namespace: Optional[str], name: str, version: Optional[int]
):
    posggym.register(env_id, "no-entry-point")
    assert posggym.spec(env_id).id == env_id

    full_name = f"{name}"
    if namespace:
        full_name = f"{namespace}/{full_name}"
    if version is not None:
        full_name = f"{full_name}-v{version}"

    assert full_name in posggym.envs.registry

    del posggym.envs.registry[env_id]


@pytest.mark.parametrize(
    "env_id",
    [
        "“MultiAccessBroadcastChannel-v0”",
        "MyNotSoAwesomeEnv-vNone\n",
        "MyNamespace///MyNotSoAwesomeEnv-vNone",
    ],
)
def test_register_error(env_id):
    with pytest.raises(
        posggym.error.Error, match=f"^Malformed environment ID: {env_id}"
    ):
        posggym.register(env_id, "no-entry-point")


@pytest.mark.parametrize(
    "env_id_input, env_id_suggested",
    [
        ("MultiAccessbroadcastchannel-v0", "MultiAccessBroadcastChannel"),
        ("MultiOccessBroadcastChannel-v0", "MultiAccessBroadcastChannel"),
        ("Multiaccessbroadcastchannel-v10", "MultiAccessBroadcastChannel"),
        ("MyAwesomeNamspce/MyAwesomeVersionedEnv-v1", "MyAwesomeNamespace"),
        ("MyAwesomeNamspce/MyAwesomeUnversionedEnv", "MyAwesomeNamespace"),
        ("MyAwesomeNamespace/MyAwesomeUnversioneEnv", "MyAwesomeUnversionedEnv"),
        ("MyAwesomeNamespace/MyAwesomeVersioneEnv", "MyAwesomeVersionedEnv"),
    ],
)
def test_env_suggestions(
    register_registration_testing_envs, env_id_input, env_id_suggested
):
    with pytest.raises(
        posggym.error.UnregisteredEnv, match=f"Did you mean: `{env_id_suggested}`?"
    ):
        posggym.make(env_id_input, disable_env_checker=True)


@pytest.mark.parametrize(
    "env_id_input, suggested_versions, default_version",
    [
        ("MultiAccessBroadcastChannel-v12", "`v0`", False),
        ("MyAwesomeNamespace/MyAwesomeVersionedEnv-v6", "`v1`, `v3`, `v5`", False),
        ("MyAwesomeNamespace/MyAwesomeUnversionedEnv-v6", "", True),
    ],
)
def test_env_version_suggestions(
    register_registration_testing_envs,
    env_id_input,
    suggested_versions,
    default_version,
):
    if default_version:
        with pytest.raises(
            posggym.error.DeprecatedEnv,
            match="It provides the default version",  # env name,
        ):
            posggym.make(env_id_input, disable_env_checker=True)
    else:
        with pytest.raises(
            posggym.error.UnregisteredEnv,
            match=f"It provides versioned environments: \\[ {suggested_versions} \\]",
        ):
            posggym.make(env_id_input, disable_env_checker=True)


def test_register_versioned_unversioned():
    # Register versioned then unversioned
    versioned_env = "Test/MyEnv-v0"
    posggym.register(versioned_env, "no-entry-point")
    assert posggym.envs.spec(versioned_env).id == versioned_env

    unversioned_env = "Test/MyEnv"
    with pytest.raises(
        posggym.error.RegistrationError,
        match=re.escape(
            "Can't register the unversioned environment `Test/MyEnv` when the versioned"
            " environment `Test/MyEnv-v0` of the same name already exists"
        ),
    ):
        posggym.register(unversioned_env, "no-entry-point")

    # Clean everything
    del posggym.envs.registry[versioned_env]

    # Register unversioned then versioned
    posggym.register(unversioned_env, "no-entry-point")
    assert posggym.envs.spec(unversioned_env).id == unversioned_env
    with pytest.raises(
        posggym.error.RegistrationError,
        match=re.escape(
            "Can't register the versioned environment `Test/MyEnv-v0` when the "
            "unversioned environment `Test/MyEnv` of the same name already exists."
        ),
    ):
        posggym.register(versioned_env, "no-entry-point")

    # Clean everything
    del posggym.envs.registry[unversioned_env]


def test_make_latest_versioned_env(register_registration_testing_envs):
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Using the latest versioned environment "
            "`MyAwesomeNamespace/MyAwesomeVersionedEnv-v5` instead of the unversioned "
            "environment `MyAwesomeNamespace/MyAwesomeVersionedEnv`."
        ),
    ):
        env = posggym.make(
            "MyAwesomeNamespace/MyAwesomeVersionedEnv", disable_env_checker=True
        )
    assert env.spec is not None
    assert env.spec.id == "MyAwesomeNamespace/MyAwesomeVersionedEnv-v5"


def test_namespace():
    # Check if the namespace context manager works
    with posggym.envs.registration.namespace("MyDefaultNamespace"):
        posggym.register("MyDefaultEnvironment-v0", "no-entry-point")
    posggym.register("MyDefaultEnvironment-v1", "no-entry-point")
    assert "MyDefaultNamespace/MyDefaultEnvironment-v0" in posggym.envs.registry
    assert "MyDefaultEnvironment-v1" in posggym.envs.registry

    del posggym.envs.registry["MyDefaultNamespace/MyDefaultEnvironment-v0"]
    del posggym.envs.registry["MyDefaultEnvironment-v1"]
