
import pytest
import numpy as np

import posggym.model as M

import posggym.envs.grid_world.driving.grid as dgrid
import posggym.envs.grid_world.driving.model as dmodel


@pytest.mark.parametrize("grid_name", list(dgrid.SUPPORTED_GRIDS.keys()))
@pytest.mark.parametrize("num_agents", ["min", "max"])
@pytest.mark.parametrize("obs_dim", [(1, 1, 1), (2, 1, 0), (1, 0, 0)])
@pytest.mark.parametrize("infinite_horizon", [False])
def test_single_step(grid_name, num_agents, obs_dim, infinite_horizon):
    """Run a single test step and reset for model with different params."""
    grid_fn = dgrid.SUPPORTED_GRIDS[grid_name][0]
    grid = grid_fn()
    if num_agents == "min":
        num_agents = 2
    else:
        num_agents = grid.supported_num_agents

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

    a = tuple(action_spaces[i].sample() for i in range(model.n_agents))
    timestep = model.step(s_0, a)

    assert state_space.contains(timestep.state)
    check_obs(timestep.observations)
    assert len(timestep.rewards) == model.n_agents

    for i, r_i in enumerate(timestep.rewards):
        assert np.isscalar(r_i), f"{r_i} is not a scalar for {model}"
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
