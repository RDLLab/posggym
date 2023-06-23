"""General tests for ``posggym.POSGModel`` implementations.

Adapted from:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_envs.py
"""
import warnings

import pytest
from tests.envs.test_envs import CHECK_ENV_IGNORE_WARNINGS
from tests.envs.utils import all_testing_env_specs, assert_equals

import posggym
import posggym.model as M
from posggym.envs.registration import EnvSpec
from posggym.utils.model_checker import check_model


@pytest.mark.parametrize(
    "spec",
    all_testing_env_specs,
    ids=[spec.id for spec in all_testing_env_specs],
)
def test_models_pass_env_checker(spec):
    """Check that all environment models pass checker with no unexpected warnings."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        env = spec.make(disable_env_checker=True).unwrapped
        check_model(env.model)

        env.close()

    for warning in caught_warnings:
        if not any(
            warning.message.args[0].startswith(msg) for msg in CHECK_ENV_IGNORE_WARNINGS
        ):
            raise posggym.error.Error(f"Unexpected warning: {warning.message}")


SEED = 42
NUM_STEPS = 50
NUM_REPEAT_STEPS = 5
NUM_INIT_STEPS = 5


@pytest.mark.parametrize(
    "env_spec",
    all_testing_env_specs,
    ids=[env.id for env in all_testing_env_specs],
)
def test_model_determinism_rollout(env_spec: EnvSpec):
    """Run a rollout with two models and assert equality.

    This test runs two models initialized with the same seed and assert that:

    - observation after first reset are the same
    - same actions are sampled by the two models
    - observations are contained in the observation space
    - step outputs are equal between the two models
    - agents in play are the same for each model

    It tests models on rollouts of NUM_STEPS timesteps.
    It also tests models where the same state (generated from rollout of len
    NUM_INIT_STEPS) is reused for NUM_REPEAT_STEPS timesteps.

    Note, this doesn't test for strict API compliance.

    """
    # Don't check rollout equality if it's a nondeterministic environment.
    if env_spec.nondeterministic is True:
        return

    env_1 = env_spec.make(disable_env_checker=True)
    env_2 = env_spec.make(disable_env_checker=True)

    # Reset envs with seed to ensure models are initialized the same for generated
    # environments that may have different models between episodes
    env_1.reset(seed=SEED)
    env_2.reset(seed=SEED)

    model_1 = env_1.model
    model_2 = env_2.model

    assert isinstance(model_1, M.POSGModel)
    assert isinstance(model_2, M.POSGModel)

    model_1.seed(SEED)
    model_2.seed(SEED)

    initial_state_1 = model_1.sample_initial_state()
    initial_state_2 = model_2.sample_initial_state()
    assert_equals(initial_state_1, initial_state_2)
    # initial_state_2 verified by previous assertion
    if model_1.state_space is not None:
        assert model_1.state_space.contains(initial_state_1)
    assert_equals(
        model_1.get_agents(initial_state_1), model_2.get_agents(initial_state_2)
    )

    initial_obs_1 = model_1.sample_initial_obs(initial_state_1)
    initial_obs_2 = model_2.sample_initial_obs(initial_state_2)
    assert_equals(initial_obs_1, initial_obs_2)
    # obs_2 verified by previous assertion
    assert all(
        model_1.observation_spaces[i].contains(o_i) for i, o_i in initial_obs_1.items()
    )
    assert all(i in initial_obs_1 for i in model_1.get_agents(initial_state_1))

    try:
        for i in model_1.get_agents(initial_state_1):
            agent_initial_state_1 = model_1.sample_agent_initial_state(
                i, initial_obs_1[i]
            )
            agent_initial_state_2 = model_2.sample_agent_initial_state(
                i, initial_obs_2[i]
            )
            assert_equals(
                agent_initial_state_1,
                agent_initial_state_2,
                "[sample_agent_initial_state]",
            )
            if model_1.state_space is not None:
                assert model_1.state_space.contains(agent_initial_state_1)
    except NotImplementedError:
        pass

    # state checked for equality at each step
    state = initial_state_1
    for rollout_mode in [True, False]:
        num_steps = NUM_STEPS if rollout_mode else NUM_INIT_STEPS + NUM_REPEAT_STEPS

        for t in range(num_steps):
            # We don't evaluate the determinism of actions
            actions = {
                i: model_1.action_spaces[i].sample() for i in model_1.get_agents(state)
            }

            result_1 = model_1.step(state, actions)
            result_2 = model_2.step(state, actions)
            assert isinstance(result_1, M.JointTimestep)
            assert isinstance(result_2, M.JointTimestep)

            assert_equals(result_1.state, result_2.state, f"[{t}][State] ")
            assert_equals(
                model_1.get_agents(result_1.state),
                model_2.get_agents(result_2.state),
                f"[{t}][get_agents] ",
            )
            # result_2.state verified by previous assertion
            if model_1.state_space is not None:
                assert model_1.state_space.contains(result_1.state)

            assert_equals(
                result_1.observations, result_2.observations, f"[{t}][Observations] "
            )
            # obs_2 verified by previous assertion
            for i, o_i in result_1.observations.items():
                assert model_1.observation_spaces[i].contains(o_i)
            assert all(i in result_1.observations for i in model_1.get_agents(state))

            assert_equals(result_1.rewards, result_2.rewards, f"[{t}][Rewards] ")
            for i, r_i in result_1.rewards.items():
                reward_range = model_1.reward_ranges[i]
                assert reward_range[0] <= r_i <= reward_range[1]
            assert_equals(
                result_1.terminations, result_2.terminations, f"[{t}][Terminations] "
            )
            assert_equals(
                result_1.truncations, result_2.truncations, f"[{t}][Truncations] "
            )
            assert (
                result_1.all_done == result_2.all_done
            ), f"[{t}] all_done 1={result_1.all_done}, all_done 2={result_2.all_done}"
            assert_equals(result_1.infos, result_2.infos, f"[{t}][Infos] ")

            if not rollout_mode and t >= NUM_INIT_STEPS:
                # don't update state
                pass
            else:
                state = result_1.state
                if result_1.all_done:
                    # done_2
                    env_1.reset(seed=SEED)
                    env_2.reset(seed=SEED)

                    model_1.seed(SEED + t)
                    model_2.seed(SEED + t)

                    state = model_1.sample_initial_state()
                    _ = model_2.sample_initial_state()

    env_1.close()
    env_2.close()
