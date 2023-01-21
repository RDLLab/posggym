"""Tests that posggym.spec works as expected.

Reference:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_spec.py
"""
import re

import pytest

import posggym


def test_spec():
    spec = posggym.spec("MultiAccessBroadcastChannel-v0")
    assert spec.id == "MultiAccessBroadcastChannel-v0"
    assert spec is posggym.envs.registry["MultiAccessBroadcastChannel-v0"]


def test_spec_kwargs():
    observation_prob = 0.8
    env = posggym.make(
        "MultiAccessBroadcastChannel-v0",
        observation_prob=observation_prob,
        disable_env_checker=True,
    )
    assert env.spec is not None
    assert env.spec.kwargs["observation_prob"] == observation_prob


def test_spec_missing_lookup():
    posggym.register(id="Test1-v0", entry_point="no-entry-point")
    posggym.register(id="Test1-v15", entry_point="no-entry-point")
    posggym.register(id="Test1-v9", entry_point="no-entry-point")
    posggym.register(id="Other1-v100", entry_point="no-entry-point")

    with pytest.raises(
        posggym.error.DeprecatedEnv,
        match=re.escape(
            "Environment version v1 for `Test1` is deprecated. Please use `Test1-v15` "
            "instead."
        ),
    ):
        posggym.spec("Test1-v1")

    with pytest.raises(
        posggym.error.UnregisteredEnv,
        match=re.escape(
            "Environment version `v1000` for environment `Test1` doesn't exist. "
            "It provides versioned environments: [ `v0`, `v9`, `v15` ]."
        ),
    ):
        posggym.spec("Test1-v1000")

    with pytest.raises(
        posggym.error.UnregisteredEnv,
        match=re.escape("Environment Unknown1 doesn't exist. "),
    ):
        posggym.spec("Unknown1-v1")


def test_spec_malformed_lookup():
    expected_error_msg = (
        "Malformed environment ID: “MultiAccessBroadcastChannel-v0”. "
        "(Currently all IDs must be of the form "
        "[namespace/](env-name)-v(version) (namespace is optional))."
    )
    with pytest.raises(
        posggym.error.Error,
        match=f"^{re.escape(expected_error_msg)}$",
    ):
        posggym.spec("“MultiAccessBroadcastChannel-v0”")


def test_spec_versioned_lookups():
    posggym.register("test/Test2-v5", "no-entry-point")

    with pytest.raises(
        posggym.error.VersionNotFound,
        match=re.escape(
            "Environment version `v9` for environment `test/Test2` doesn't exist. "
            "It provides versioned environments: [ `v5` ]."
        ),
    ):
        posggym.spec("test/Test2-v9")

    with pytest.raises(
        posggym.error.DeprecatedEnv,
        match=re.escape(
            "Environment version v4 for `test/Test2` is deprecated. Please use "
            "`test/Test2-v5` instead."
        ),
    ):
        posggym.spec("test/Test2-v4")

    assert posggym.spec("test/Test2-v5") is not None


def test_spec_default_lookups():
    posggym.register("test/Test3", "no-entry-point")

    with pytest.raises(
        posggym.error.DeprecatedEnv,
        match=re.escape(
            "Environment version `v0` for environment `test/Test3` doesn't exist. "
            "It provides the default version test/Test3`."
        ),
    ):
        posggym.spec("test/Test3-v0")

    assert posggym.spec("test/Test3") is not None
