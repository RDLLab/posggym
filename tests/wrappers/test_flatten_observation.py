"""Test for FlattenObservation Wrapper.

Ref
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/wrappers/test_flatten_observation.py

"""
import numpy as np
from gymnasium import spaces

import posggym
from posggym.envs.grid_world.driving import CELL_OBS, Speed
from posggym.wrappers import FlattenObservation


def test_flatten_observation():
    env = posggym.make("Driving-7x7RoundAbout-n2-v0", disable_env_checker=True)
    wrapped_env = FlattenObservation(env)

    obs, info = env.reset()
    wrapped_obs, wrapped_obs_info = wrapped_env.reset()

    grid_width, grid_height = 7, 7
    obs_depth, obs_width = 5, 3
    space = spaces.Tuple(
        (
            spaces.Tuple(
                tuple(
                    spaces.Discrete(len(CELL_OBS)) for _ in range(obs_depth * obs_width)
                )
            ),
            spaces.Discrete(len(Speed)),
            spaces.Tuple(
                (spaces.Discrete(grid_width), spaces.Discrete(grid_height))
            ),  # dest coord,
            spaces.Discrete(2),  # dest reached
            spaces.Discrete(2),  # crashed
        )
    )
    wrapped_space = spaces.Box(
        0,
        1,
        [
            len(CELL_OBS) * obs_depth * obs_width
            + len(Speed)
            + grid_width
            + grid_height
            + 2
            + 2
        ],
        dtype=np.int64,
    )

    assert all(i in obs for i in env.agents)
    for i, obs_i in obs.items():
        assert space.contains(obs_i)
    assert all(i in wrapped_obs for i in env.agents)
    for i, wrapped_obs_i in wrapped_obs.items():
        assert wrapped_space.contains(wrapped_obs_i), wrapped_obs_i.shape
    assert isinstance(info, dict)
    assert isinstance(wrapped_obs_info, dict)
