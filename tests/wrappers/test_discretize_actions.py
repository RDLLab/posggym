"""Test for DiscretizeActions Wrapper."""
from typing import cast

import numpy as np
import pytest
from gymnasium import spaces

import posggym
from posggym.envs.continuous.driving_continuous import DrivingContinuousModel
from posggym.wrappers import DiscretizeActions


@pytest.mark.parametrize("num_actions", [3, 5, 10])
def test_discretize_actions_flatten(num_actions):
    env = posggym.make(
        "DrivingContinuous-v0",
        disable_env_checker=True,
        num_agents=2,
        world="7x7RoundAbout",
        obs_dist=4.0,
        n_sensors=16,
    )
    wrapped_env = DiscretizeActions(env, num_actions=num_actions, flatten=True)

    model = cast(DrivingContinuousModel, env.model)

    box_act_dim = 2
    base_space = spaces.Box(
        low=np.array([-model.dyaw_limit, -model.dvel_limit], dtype=np.float32),
        high=np.array([model.dyaw_limit, model.dvel_limit], dtype=np.float32),
    )

    n_flat_actions = np.prod([num_actions] * box_act_dim)
    # wrapped_space = spaces.Discrete(n_flat_actions)
    assert all(
        act_space.n == n_flat_actions
        for act_space in wrapped_env.action_spaces.values()
    )

    # perform actions and then check last_action from unwrapped_env
    env.reset()
    wrapped_env.reset()

    # Test action 0
    wrapped_env.step({i: 0 for i in wrapped_env.agents})
    unwrapped_actions = wrapped_env.unwrapped._last_actions

    for i, a in unwrapped_actions.items():
        assert base_space.contains(a), (a, base_space)
        assert np.isclose(a, base_space.low).all(), (i, a, base_space.low)

    wrapped_env.step({i: n_flat_actions - 1 for i in wrapped_env.agents})
    unwrapped_actions = wrapped_env.unwrapped._last_actions

    for i, a in unwrapped_actions.items():
        assert base_space.contains(a), (a, base_space)
        assert np.isclose(a, base_space.high).all(), (i, a, base_space.high)

    wrapped_env.close()


@pytest.mark.parametrize("num_actions", [3, 5, 10])
def test_discretize_actions_multidiscrete(num_actions):
    env = posggym.make(
        "DrivingContinuous-v0",
        disable_env_checker=True,
        num_agents=2,
        world="7x7RoundAbout",
        obs_dist=4.0,
        n_sensors=16,
    )
    wrapped_env = DiscretizeActions(env, num_actions=num_actions, flatten=False)

    model = cast(DrivingContinuousModel, env.model)

    box_act_dim = 2
    base_space = spaces.Box(
        low=np.array([-model.dyaw_limit, -model.dvel_limit], dtype=np.float32),
        high=np.array([model.dyaw_limit, model.dvel_limit], dtype=np.float32),
    )

    # wrapped_space = spaces.MultiDiscrete([num_actions] * box_act_dim)

    # perform actions and then check last_action from unwrapped_env
    env.reset()
    wrapped_env.reset()

    # Test action 0
    wrapped_env.step({i: [0] * box_act_dim for i in wrapped_env.agents})
    unwrapped_actions = wrapped_env.unwrapped._last_actions

    for i, a in unwrapped_actions.items():
        assert base_space.contains(a), (a, base_space)
        assert np.isclose(a, base_space.low).all(), (i, a, base_space.low)

    wrapped_env.step({i: [num_actions - 1] * box_act_dim for i in wrapped_env.agents})
    unwrapped_actions = wrapped_env.unwrapped._last_actions

    for i, a in unwrapped_actions.items():
        assert base_space.contains(a), (a, base_space)
        assert np.isclose(a, base_space.high).all(), (i, a, base_space.high)

    wrapped_env.close()
