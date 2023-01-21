"""Tests that `posggym.pprint_registry` works as expected.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_pprint_registry.py
"""
import posggym


# To ignore the trailing whitespaces, will need flake to ignore this file.
# flake8: noqa

reduced_registry = {env_id: env_spec for env_id, env_spec in posggym.registry.items()}


def test_pprint_custom_registry():
    """Testing a registry different from default."""
    a = {
        "MultiAccessBroadcastChannel-v0": posggym.envs.registry[
            "MultiAccessBroadcastChannel-v0"
        ],
    }
    out = posggym.pprint_registry(a, disable_print=True)

    correct_out = """===== classic =====
MultiAccessBroadcastChannel-v0

"""
    assert out == correct_out
