
import pytest
import numpy as np

import posggym.model as M

import posggym.envs.grid_world.driving.grid as dgrid
import posggym.envs.grid_world.driving.model as dmodel


@pytest.mark.parametrize(
    "grid, num_agents",
    [
        (dgrid.get_3x3_grid(), 2),
        (dgrid.get_7x7_crisscross_grid(), 2),
        (dgrid.get_7x7_crisscross2_grid(), 6),
        (dgrid.get_7x7_crisscross3_grid(), 2),
        (dgrid.get_7x7_crisscross4_grid(), 6),
        (dgrid.get_7x7_crisscross5_grid(), 2),
    ])
@pytest.mark.parametrize("obs_dim", [(1, 1, 1), (2, 1, 0), (1, 0, 0)])
@pytest.mark.parametrize("infinite_horizon", [False])
def test_single_step(grid, num_agents, obs_dim, infinite_horizon):
    """Run a single test step and reset for model with different params."""
    model = dmodel.DrivingModel(grid, num_agents, obs_dim, infinite_horizon)

    state_space = model.state_space
    obs_spaces = model.observation_spaces
    action_spaces = model.action_spaces

    assert len(obs_spaces) == model.n_agents
    assert len(action_spaces) == model.n_agents

    def check_obs(obs):
        assert len(obs) == model.n_agents
        for i, o_i in enumerate(obs):
            assert obs_spaces[i].contains(o_i), \
                f"Reset agent {i} observation: {o_i} not in space for {model}"

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
        assert len(outcomes) == model.n_agents
        for outcome_i in outcomes:
            assert outcome_i in M.Outcome
        assert outcomes == model.get_outcome(timestep.state)
