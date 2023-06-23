"""General tests for ``posggym.Env`` environment implementations.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_envs.py
"""
import pickle
import warnings

import pytest
from tests.envs.utils import (
    all_testing_env_specs,
    all_testing_initialised_envs,
    assert_equals,
)

import posggym
from posggym.envs.registration import EnvSpec
from posggym.utils.env_checker import check_env
from posggym.utils.passive_env_checker import data_equivalence

PASSIVE_CHECK_IGNORE_WARNING = [
    f"\x1b[33mWARN: {message}"
    for message in [
        "It seems a Box observation space is an image but the",
    ]
]


CHECK_ENV_IGNORE_WARNINGS = [
    # f"\x1b[33mWARN: {message}\x1b[0m"
    f"\x1b[33mWARN: {message}"
    for message in [
        "A Box observation space minimum value is -infinity. This is probably too low.",
        (
            "A Box observation space maximum value is -infinity. This is probably too"
            " high."
        ),
        (
            "For Box action spaces, we recommend using a symmetric and normalized space"
            " (range=[-1, 1] or [0, 1]). See "
            "https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html "
            "for more information."
        ),
        (
            "It seems a Box observation space is an image but the `dtype` is not "
            "`np.uint8`"
        ),
        (
            "It seems a Box observation space is an image but the lower and upper "
            "bounds are not [0, 255]."
        ),
    ]
]


@pytest.mark.parametrize(
    "spec",
    all_testing_env_specs,
    ids=[spec.id for spec in all_testing_env_specs],
)
def test_envs_pass_env_checker(spec):
    """Check that all environments pass checker with no unexpected warnings."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        env = spec.make(disable_env_checker=True).unwrapped
        check_env(env, skip_render_check=True)
        env.close()

    for warning in caught_warnings:
        if not any(
            warning.message.args[0].startswith(msg) for msg in CHECK_ENV_IGNORE_WARNINGS
        ):
            raise posggym.error.Error(f"Unexpected warning: {warning.message}")


SEED = 0
NUM_STEPS = 50


@pytest.mark.parametrize(
    "env_spec",
    all_testing_env_specs,
    ids=[env.id for env in all_testing_env_specs],
)
def test_env_determinism_rollout(env_spec: EnvSpec):
    """Run a rollout with two environments and assert equality.

    This test runs a rollout of NUM_STEPS steps with two environments initialized with
    the same seed and assert that:

    - observation after first reset are the same
    - same actions are sampled by the two envs
    - observations are contained in the observation space
    - obs, rew, term, trunc, done, and info are equals between the two envs

    """
    # Don't check rollout equality if it's a nondeterministic environment.
    if env_spec.nondeterministic is True:
        return

    env_1 = env_spec.make(disable_env_checker=True)
    env_2 = env_spec.make(disable_env_checker=True)

    initial_obs_1, initial_info_1 = env_1.reset(seed=SEED)
    initial_obs_2, initial_info_2 = env_2.reset(seed=SEED)
    assert_equals(initial_obs_1, initial_obs_2)

    for time_step in range(NUM_STEPS):
        assert_equals(env_1.agents, env_2.agents, f"[{time_step}][Agents] ")
        assert_equals(env_1.state, env_2.state, f"[{time_step}][State] ")

        # We don't evaluate the determinism of actions
        actions = {i: env_1.action_spaces[i].sample() for i in env_1.agents}

        obs_1, rew_1, term_1, trunc_1, done_1, info_1 = env_1.step(actions)
        obs_2, rew_2, term_2, trunc_2, done_2, info_2 = env_2.step(actions)

        assert_equals(obs_1, obs_2, f"[{time_step}][Observations] ")
        # obs_2 verified by previous assertion
        for i, o_i in obs_1.items():
            assert env_1.observation_spaces[i].contains(o_i)

        assert_equals(rew_1, rew_2, f"[{time_step}][Rewards] ")
        for i, r_i in rew_1.items():
            reward_range = env_1.reward_ranges[i]
            assert reward_range[0] <= r_i <= reward_range[1]
        assert_equals(term_1, term_2, f"[{time_step}][Terminated] ")
        assert_equals(trunc_1, trunc_2, f"[{time_step}][Truncated] ")
        assert done_1 == done_2, f"[{time_step}] done 1={done_1}, done 2={done_2}"
        assert_equals(info_1, info_2, f"[{time_step}][Info] ")

        if done_1:
            # done_2
            env_1.reset()
            env_2.reset()

    env_1.close()
    env_2.close()


@pytest.mark.parametrize(
    "env",
    all_testing_initialised_envs,
    ids=[env.spec.id for env in all_testing_initialised_envs if env.spec is not None],
)
def test_pickle_env(env: posggym.Env):
    """Test that env can be pickled consistently."""
    pickled_env = pickle.loads(pickle.dumps(env))

    data_equivalence(env.reset(), pickled_env.reset())

    actions = {i: env.action_spaces[i].sample() for i in env.agents}
    data_equivalence(env.step(actions), pickled_env.step(actions))
    env.close()
    pickled_env.close()
