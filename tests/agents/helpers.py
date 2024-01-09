"""Finds all the specs of policies that we can test with.

Adapted from:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/utils.py

"""
from typing import List, Optional

import numpy as np
import posggym
import posggym.agents as pga
import torch
from posggym.agents import torch_policy
from posggym.agents.registration import PolicySpec
from tests.conftest import env_id_prefix


def try_make_policy(spec: PolicySpec) -> Optional[pga.Policy]:
    """Tries to make the policy showing if it is possible."""
    try:
        if spec.env_id is None:
            env = posggym.make("MultiAccessBroadcastChannel-v0")
        elif spec.env_args is None:
            env = posggym.make(spec.env_id)
        else:
            env = posggym.make(spec.env_id, **spec.env_args)

        if spec.valid_agent_ids:
            agent_id = spec.valid_agent_ids[0]
        else:
            agent_id = env.possible_agents[0]
        return pga.make(spec, env.model, agent_id)
    except (
        ImportError,
        posggym.error.DependencyNotInstalled,
        posggym.error.MissingArgument,
    ) as e:
        posggym.logger.warn(
            f"Not testing posggym.agents policy spec `{spec.id}` due to error: {e}"
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Error trying to make posggym.agents policy spec `{spec.id}`."
        ) from e
    return None


# Tries to make all policies to test with
_all_testing_initialised_policies: List[Optional[pga.Policy]] = [
    try_make_policy(policy_spec)
    for policy_spec in pga.registry.values()
    if env_id_prefix is None or policy_spec.id.startswith(env_id_prefix)
]
all_testing_initialised_policies: List[pga.Policy] = [
    policy for policy in _all_testing_initialised_policies if policy is not None
]
all_testing_initialised_torch_policies: List[torch_policy.PPOPolicy] = [
    policy
    for policy in all_testing_initialised_policies
    if isinstance(policy, torch_policy.PPOPolicy)
]

# All testing posggym-agents policy specs
all_testing_policy_specs: List[PolicySpec] = [
    policy.spec
    for policy in all_testing_initialised_policies
    if policy.spec is not None
]
# All testing posggym-agents policy specs that use torch
all_testing_torch_policy_specs: List[PolicySpec] = [
    policy.spec
    for policy in all_testing_initialised_torch_policies
    if policy.spec is not None
]


def assert_equals(a, b, prefix=None):
    """Assert equality of data structures `a` and `b`.

    Arguments
    ---------
    a: first data structure
    b: second data structure
    prefix: prefix for failed assertion message for types and dicts

    """
    assert type(a) == type(b), f"{prefix}Differing types: {a} and {b}"
    if isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), f"{prefix}Key sets differ: {a} and {b}"
        for k in a:
            v_a = a[k]
            v_b = b[k]
            assert_equals(v_a, v_b, f"{k:}")
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f"{prefix}Tensors differ: {a} and {b}"
    elif isinstance(a, (tuple, list)):
        for elem_from_a, elem_from_b in zip(a, b):
            assert_equals(elem_from_a, elem_from_b, prefix)
    else:
        assert a == b
