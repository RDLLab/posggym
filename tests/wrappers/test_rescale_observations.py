"""Test for RescaleObservations Wrapper."""
import math

import numpy as np
import pytest
from gymnasium import spaces

import posggym
from posggym.wrappers import RescaleObservations


@pytest.mark.parametrize(
    ["min_val", "max_val"],
    [
        (-1.0, 1.0),
        (0, 1),
        (-100, 87.0),
        ({"0": -1.0, "1": -3.0}, {"0": 1.0, "1": -1.0}),
        (
            np.full((16 * 2 + 5,), -1.0, dtype=np.float32),
            np.full((16 * 2 + 5,), 1.0, dtype=np.float32),
        ),
    ],
)
def test_rescale_observation(min_val, max_val):
    obs_dist, n_sensors, size = 4.0, 16, 7

    env = posggym.make(
        "DrivingContinuous-v0",
        disable_env_checker=True,
        num_agents=2,
        world="7x7RoundAbout",
        obs_dist=obs_dist,
        n_sensors=n_sensors,
    )
    wrapped_env = RescaleObservations(env, min_val, max_val)

    obs, info = env.reset()
    wrapped_obs, wrapped_obs_info = wrapped_env.reset()

    sensors_dim, obs_dim = n_sensors * 2, n_sensors * 2 + 5
    sensor_low, sensor_high = [0.0] * sensors_dim, [obs_dist] * sensors_dim
    base_space = spaces.Box(
        low=np.array([*sensor_low, -2 * math.pi, -1, -1, 0, 0], dtype=np.float32),
        high=np.array([*sensor_high, 2 * math.pi, 1, 1, size, size], dtype=np.float32),
    )

    wrapped_spaces = {}
    if isinstance(min_val, (int, float)):
        # assume max_val also (int, float)
        wrapped_spaces = {
            i: spaces.Box(
                low=min_val,
                high=max_val,
                shape=(obs_dim,),
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
                shape=(obs_dim,),
                dtype=np.float32,
            )
            for i in env.agents
        }

    assert all(i in obs for i in env.agents)
    for i, obs_i in obs.items():
        assert base_space.contains(obs_i), (obs_i, base_space)

    assert all(i in wrapped_obs for i in env.agents)
    for i, wrapped_obs_i in wrapped_obs.items():
        assert wrapped_spaces[i].contains(wrapped_obs_i), (
            wrapped_obs_i,
            wrapped_spaces[i],
        )

    assert isinstance(info, dict)
    assert isinstance(wrapped_obs_info, dict)
    wrapped_env.close()
