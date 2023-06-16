"""General tests for PyTorch policies."""
import pickle
import warnings

import posggym
import posggym.agents as pga
import pytest
from posggym.agents.policy import Policy
from posggym.agents.registration import PolicySpec
from tests.agents.helpers import (
    all_testing_initialised_torch_policies,
    all_testing_torch_policy_specs,
    assert_equals,
)


SEED = 0
NUM_STEPS = 50


@pytest.mark.parametrize(
    "spec",
    all_testing_torch_policy_specs,
    ids=[policy.id for policy in all_testing_torch_policy_specs],
)
def test_policy(spec: PolicySpec):
    """Run a policy in environment and checks basic functionality."""
    # Ignore warnings for env creation
    with warnings.catch_warnings(record=False):
        env_args = {} if spec.env_args is None else spec.env_args
        env = posggym.make(spec.env_id, **env_args)

    obs, _ = env.reset(seed=SEED)

    if spec.valid_agent_ids:
        test_agent_id = list(set(env.agents).intersection(spec.valid_agent_ids))[0]
    else:
        test_agent_id = env.agents[0]

    test_policy = pga.make(spec, env.model, test_agent_id)
    assert isinstance(test_policy, Policy)

    test_policy.reset(seed=SEED + 1)

    for _ in range(2):
        joint_action = {}
        for i in env.agents:
            if i == test_agent_id and test_policy.observes_state:
                a = test_policy.step(env.state)
            elif i == test_agent_id:
                a = test_policy.step(obs[i])
            else:
                a = env.action_spaces[i].sample()
            joint_action[i] = a

        obs, _, _, _, _, _ = env.step(joint_action)

    env.close()
    test_policy.close()


@pytest.mark.parametrize(
    "spec",
    all_testing_torch_policy_specs,
    ids=[policy.id for policy in all_testing_torch_policy_specs],
)
def test_policy_determinism_rollout(spec: PolicySpec):
    """Run a rollout with two policies and assert equality.

    This test runs a rollout of NUM_STEPS steps with two sets of policies initialized
    with the same seed and asserts that:

    - observation after first reset are the same
    - same actions are sampled by the two policies
    - actions are contained in the environment action space
    - trajectories are the same between the two policies

    """
    # Don't check rollout equality if it's a nondeterministic policy.
    if spec.nondeterministic is True:
        return

    env_id = spec.env_id
    env_args = {} if spec.env_args is None else spec.env_args
    # use two identical environments in case policies utilize models (e.g. for planning)
    env_1 = posggym.make(env_id, **env_args)
    # Don't check rollout equality if environment is nondeterministic since this may
    # affect policies.
    if env_1.spec is None or env_1.spec.nondeterministic is True:
        env_1.close()
        return
    env_2 = posggym.make(env_id, **env_args)

    obs, _ = env_1.reset(seed=SEED)
    env_2.reset(seed=SEED)

    if spec.valid_agent_ids:
        agent_id = list(set(env_1.agents).intersection(spec.valid_agent_ids))[0]
    else:
        agent_id = env_1.agents[0]

    policy_1 = pga.make(spec, env_1.model, agent_id)
    policy_2 = pga.make(spec, env_2.model, agent_id)

    policy_1.reset(seed=SEED + 1)
    policy_2.reset(seed=SEED + 1)

    assert_equals(policy_1.get_state(), policy_2.get_state())

    for _ in range(NUM_STEPS):
        if policy_1.observes_state:
            action_1 = policy_1.step(env_1.state)
            action_2 = policy_2.step(env_1.state)
        else:
            action_1 = policy_1.step(obs[agent_id])
            action_2 = policy_2.step(obs[agent_id])

        assert_equals(action_1, action_2)
        assert env_1.action_spaces[agent_id].contains(action_1)

        assert_equals(policy_1.get_state(), policy_2.get_state())

        actions = {}
        for i in env_1.agents:
            if i == agent_id:
                actions[agent_id] = action_1
            else:
                actions[i] = env_1.action_spaces[i].sample()
                # must sample env 2 action space so rng states are the same
                env_2.action_spaces[i].sample()

        obs, _, terminated, _, done, _ = env_1.step(actions)
        env_2.step(actions)

        if terminated[agent_id] or done:
            obs, _ = env_1.reset()
            env_2.reset()

            if agent_id not in env_1.agents:
                # policy no longer valid, for new environment state so just end test
                break

            policy_1.reset()
            policy_2.reset()
            assert_equals(policy_1.get_state(), policy_2.get_state())

    env_1.close()
    env_2.close()
    policy_1.close()
    policy_2.close()


@pytest.mark.parametrize(
    "policy",
    all_testing_initialised_torch_policies,
    ids=[
        policy.spec.id
        for policy in all_testing_initialised_torch_policies
        if policy.spec is not None
    ],
)
def test_pickle_policy(policy: pga.Policy):
    """Test that policy can be pickled consistently."""
    pickled_policy = pickle.loads(pickle.dumps(policy))
    assert isinstance(pickled_policy, Policy)

    assert_equals(policy.agent_id, pickled_policy.agent_id)
    assert_equals(policy.get_state(), pickled_policy.get_state())

    policy.close()
    pickled_policy.close()
