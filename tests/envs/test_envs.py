"""General tests for POSG Gym environments """

import pytest
import numpy as np

from posggym import core
import posggym.model as M
from tests.envs.spec_list import spec_list


def _has_prefix(spec, env_name_prefix):
    return spec.id.startswith(env_name_prefix)


@pytest.mark.parametrize("spec", spec_list)
def test_env(spec, test_render, env_name_prefix):
    """Run single test step and reset for each registered environment """
    if env_name_prefix is not None and not _has_prefix(spec, env_name_prefix):
        return

    # Capture warnings
    with pytest.warns(None) as warnings:
        env = spec.make()

    assert isinstance(env.unwrapped, core.Env)

    for warning_msg in warnings:
        assert "autodetected dtype" not in str(warning_msg.message)

    obs_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    assert len(obs_spaces) == env.n_agents
    assert len(action_spaces) == env.n_agents

    model = env.model
    assert isinstance(model, M.POSGModel)

    obs = env.reset()
    if env.observation_first:
        assert len(obs) == env.n_agents
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

    if test_render:
        for mode in env.metadata.get("render.modes",  []):
            env.render(mode=mode)

    env.close()


@pytest.mark.parametrize("spec", spec_list)
def test_random_rollout(spec, env_name_prefix):
    """Tests a random rollout in each environment

    N.B. May need to be more selective of the environments to test in the
    future.
    """
    if env_name_prefix is not None and not _has_prefix(spec, env_name_prefix):
        return

    step_limit = 100

    env = spec.make()
    assert isinstance(env.unwrapped, core.Env)

    obs = env.reset()

    if env.observation_first:
        for i, o_i in enumerate(obs):
            assert env.observation_spaces[i].contains(o_i)

    for _ in range(step_limit):
        a = tuple(a.sample() for a in env.action_spaces)
        obs, _, done, _ = env.step(a)
        if done:
            break
    env.close()


@pytest.mark.parametrize("spec", spec_list)
def test_random_seeding(spec, env_name_prefix):
    """Test random seeding using random rollouts in each environment

    N.B. May need to be more selective of the environments to test in the
    future.
    """
    if env_name_prefix is not None and not _has_prefix(spec, env_name_prefix):
        return

    step_limit = 50
    seed = 42

    trajectories = [[], []]
    states = [[], []]

    for i in range(2):
        env = spec.make()
        action_spaces = env.action_spaces

        obs = env.reset(seed=seed)
        states[i].append(env.state)
        for j in range(len(action_spaces)):
            action_spaces[j].seed(seed + 1 + j)

        for _ in range(step_limit):
            a = tuple(a.sample() for a in action_spaces)
            obs, rews, done, _ = env.step(a)
            trajectories[i].append((a, obs, rews, done))
            if done:
                break
        env.close()

    assert len(trajectories[0]) == len(trajectories[1])
    for ts0, ts1 in zip(trajectories[0], trajectories[1]):
        assert ts0[0] == ts1[0], f"{ts0[0]} != {ts1[0]}"
        for o0, o1 in zip(ts0[1], ts1[1]):
            if isinstance(o0, np.ndarray):
                assert (o0 == o1).all(), f"{o0} != {o1}"
            else:
                assert o0 == o1, f"{o0} != {o1}"
        assert ts0[2] == ts1[2], f"{ts0[2]} != {ts1[2]}"
        assert ts0[3] == ts1[3], f"{ts0[3]} != {ts1[3]}"


@pytest.mark.parametrize("spec", spec_list)
def test_model(spec, env_name_prefix):
    """Run single test step and reset for model of each registered env """
    if env_name_prefix is not None and not _has_prefix(spec, env_name_prefix):
        return

    with pytest.warns(None):
        env = spec.make()

    model = env.model
    assert isinstance(model, M.POSGModel)

    obs_spaces = model.observation_spaces
    action_spaces = model.action_spaces

    assert len(obs_spaces) == model.n_agents
    assert len(action_spaces) == model.n_agents

    def check_obs(obs):
        assert len(obs) == model.n_agents
        for i, o_i in enumerate(obs):
            assert obs_spaces[i].contains(o_i), \
                f"Reset agent {i} observation: {o_i} not in space for {env}"

    def check_state(state):
        hash(state)
        try:
            state_space = model.state_space
            assert state_space.contains(state)
        except NotImplementedError:
            pass

    b_0 = model.initial_belief
    s_0 = b_0.sample()

    check_state(s_0)

    if model.observation_first:
        obs_0 = model.sample_initial_obs(s_0)
        check_obs(obs_0)

        try:
            for i in range(model.n_agents):
                b_i_0 = model.get_agent_initial_belief(i, obs_0[i])
                s_i_0 = b_i_0.sample()
                check_state(s_i_0)
        except NotImplementedError:
            pass

    a = tuple(action_spaces[i].sample() for i in range(model.n_agents))
    timestep = model.step(s_0, a)

    check_state(timestep.state)

    check_obs(timestep.observations)
    assert len(timestep.rewards) == model.n_agents

    for i, r_i in enumerate(timestep.rewards):
        assert np.isscalar(r_i), f"{r_i=} for {i=} is not a scalar for {model}"
        assert model.reward_ranges[i][0] <= r_i <= model.reward_ranges[i][1]

    dones, all_done = timestep.dones, timestep.all_done
    assert len(dones) == model.n_agents
    assert all(isinstance(d, bool) for d in dones), \
        f"Expected {dones=} to be tuple of bools for {model}"
    assert isinstance(all_done, bool), \
        f"Expected {all_done=} to be a bool for {model}"

    outcomes = timestep.outcomes
    assert len(outcomes) == model.n_agents
    for outcome_i in outcomes:
        assert outcome_i in M.Outcome
