"""General tests for POSG Gym environments """

import pytest
import numpy as np

from posggym import core
import posggym.model as M
from tests.envs.spec_list import spec_list


@pytest.mark.parametrize("spec", spec_list)
def test_env(spec):
    """Run single test step and reset for each registered environment """

    # Capture warnings
    with pytest.warns(None) as warnings:
        env = spec.make()

    assert isinstance(env.unwrapped, core.Env)

    for warning_msg in warnings:
        assert "autodetected dtype" not in str(warning_msg.message)

    obs_spaces = env.obs_spaces
    action_spaces = env.action_spaces

    assert len(obs_spaces) == env.n_agents
    assert len(action_spaces) == env.n_agents

    model = env.model
    assert isinstance(model, M.POSGModel)

    obs = env.reset()
    for i, o_i in enumerate(obs):
        assert obs_spaces[i].contains(o_i), \
            f"Reset agent {i} observation: {o_i} not in space for {env}"

    a = tuple(action_spaces[i].sample() for i in range(env.n_agents))
    obs, rewards, done, info = env.step(a)

    assert len(obs) == env.n_agents
    assert len(rewards) == env.n_agents

    for i, (o_i, r_i) in enumerate(zip(obs, rewards)):
        assert obs_spaces[i].contains(o_i), \
            f"Reset agent {i} observation: {o_i} not in space for {env}"
        assert np.isscalar(r_i), f"{r_i} is not a scalar for {env}"

    assert isinstance(done, bool), f"Expected {done} to be a boolean for {env}"
    assert isinstance(info, dict), f"Expected {info} to be a dict for {env}"

    for mode in env.metadata.get("render.modes",  []):
        env.render(mode=mode)

    env.close()


@pytest.mark.parametrize("spec", spec_list)
def test_random_rollout(spec):
    """Tests a random rollout in each environment

    N.B. May need to be more selective of the environments to test in the
    future.
    """
    step_limit = 100

    env = spec.make()
    assert isinstance(env.unwrapped, core.Env)

    obs = env.reset()
    for _ in range(step_limit):
        for i, o_i in enumerate(obs):
            assert env.obs_spaces[i].contains(o_i)

        a = tuple(a.sample() for a in env.action_spaces)
        obs, _, done, _ = env.step(a)
        if done:
            break
    env.close()


@pytest.mark.parametrize("spec", spec_list)
def test_model(spec):
    """Run single test step and reset for model of each registered env """

    with pytest.warns(None):
        env = spec.make()

    model = env.model
    assert isinstance(model, M.POSGModel)

    state_space = model.state_space
    obs_spaces = model.obs_spaces
    action_spaces = model.action_spaces

    assert len(obs_spaces) == model.n_agents
    assert len(action_spaces) == model.n_agents

    def check_obs(obs):
        assert len(obs) == model.n_agents
        for i, o_i in enumerate(obs):
            assert obs_spaces[i].contains(o_i), \
                f"Reset agent {i} observation: {o_i} not in space for {env}"

    b_0 = model.initial_belief
    s_0 = b_0.sample()

    assert state_space.contains(s_0)

    obs_0 = model.sample_initial_obs(s_0)
    check_obs(obs_0)

    s_0, obs_0 = model.sample_initial_state_and_obs()
    assert state_space.contains(s_0)
    check_obs(obs_0)

    a = tuple(action_spaces[i].sample() for i in range(model.n_agents))
    timestep = model.step(s_0, a)

    assert state_space.contains(timestep.state)
    check_obs(timestep.observations)
    assert len(timestep.rewards) == model.n_agents

    for i, r_i in enumerate(timestep.rewards):
        assert np.isscalar(r_i), f"{r_i} is not a scalar for {model}"
        assert model.reward_ranges[i][0] <= r_i <= model.reward_ranges[i][1]

    done, outcomes = timestep.done, timestep.outcomes
    assert isinstance(done, bool), f"Expected {done} to be a bool for {model}"
    assert done == model.is_done(timestep.state)

    if outcomes is not None:
        assert len(outcomes) == env.n_agents
        for outcome_i in outcomes:
            assert outcome_i in M.Outcome
        assert outcomes == model.get_outcome(timestep.state)
