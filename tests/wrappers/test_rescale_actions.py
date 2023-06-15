"""Test for RescaleActions Wrapper."""
from typing import cast

import numpy as np
import pytest
from gymnasium import spaces

import posggym
from posggym.envs.continuous.driving_continuous import DrivingContinuousModel
from posggym.wrappers import RescaleActions


@pytest.mark.parametrize(
    ["min_val", "max_val"],
    [
        (-1.0, 1.0),
        (0, 1),
        (-100, 87.0),
        ({"0": -1.0, "1": -3.0}, {"0": 1.0, "1": -1.0}),
        (
            np.array([-1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        ),
    ],
)
def test_rescale_actions(min_val, max_val):
    env = posggym.make(
        "DrivingContinuous-v0",
        disable_env_checker=True,
        num_agents=2,
        world="7x7RoundAbout",
        obs_dist=4.0,
        n_sensors=16,
    )
    wrapped_env = RescaleActions(env, min_action=min_val, max_action=max_val)

    model = cast(DrivingContinuousModel, env.model)

    box_act_dim = 2
    base_space = spaces.Box(
        low=np.array([-model.dyaw_limit, -model.dvel_limit], dtype=np.float32),
        high=np.array([model.dyaw_limit, model.dvel_limit], dtype=np.float32),
    )

    wrapped_spaces = {}
    if isinstance(min_val, (int, float)):
        # assume max_val also (int, float)
        wrapped_spaces = {
            i: spaces.Box(
                low=min_val,
                high=max_val,
                shape=(box_act_dim,),
                dtype=np.float32,
            )
            for i in env.agents
        }
    elif isinstance(min_val, np.ndarray):
        # assume max_val also np.ndarray
        wrapped_spaces = {
            i: spaces.Box(
                low=min_val,
                high=max_val,
                dtype=np.float32,
            )
            for i in env.agents
        }
    else:
        # assume dictionaries of int/float
        wrapped_spaces = {
            i: spaces.Box(
                low=min_val[i],
                high=max_val[i],
                shape=(box_act_dim,),
                dtype=np.float32,
            )
            for i in env.agents
        }

    for i, act_space in wrapped_env.action_spaces.items():
        assert isinstance(act_space, spaces.Box)
        assert np.isclose(wrapped_spaces[i].low, act_space.low).all(), (
            i,
            wrapped_spaces[i].low,
            act_space.low,
        )
        assert np.isclose(wrapped_spaces[i].high, act_space.high).all(), (
            i,
            wrapped_spaces[i].high,
            act_space.high,
        )

    # perform actions and then check last_action from unwrapped_env
    env.reset()
    wrapped_env.reset()

    # Test lower bound actions in wrapped space is lower bound action in base space
    wrapped_env.step({i: wrapped_spaces[i].low for i in wrapped_env.agents})
    unwrapped_actions = wrapped_env.unwrapped._last_actions
    for i, a in unwrapped_actions.items():
        assert base_space.contains(a), (a, base_space)
        assert np.isclose(a, base_space.low).all(), (i, a, base_space.low)

    # Test mid action in wrapped space is mid action in base space
    wrapped_env.step(
        {
            i: (
                wrapped_spaces[i].low
                + (wrapped_spaces[i].high - wrapped_spaces[i].low) / 2
            )
            for i in wrapped_env.agents
        }
    )
    unwrapped_actions = wrapped_env.unwrapped._last_actions
    for i, a in unwrapped_actions.items():
        assert base_space.contains(a), (a, base_space)
        base_mid = base_space.low + (base_space.high - base_space.low) / 2
        assert np.isclose(a, base_mid, atol=1e-5).all(), (i, a, base_mid)

    # Test upper bound actions in wrapped space is upper bound action in base space
    wrapped_env.step({i: wrapped_spaces[i].high for i in wrapped_env.agents})
    unwrapped_actions = wrapped_env.unwrapped._last_actions
    for i, a in unwrapped_actions.items():
        assert base_space.contains(a), (a, base_space)
        assert np.isclose(a, base_space.high, atol=1e-5).all(), (i, a, base_space.high)

    wrapped_env.close()
