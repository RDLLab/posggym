"""Test for FlattenObservations Wrapper.

Ref
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/wrappers/test_flatten_observation.py

"""
import numpy as np
import posggym
from gymnasium import spaces
from posggym.envs.grid_world.driving import CELL_OBS, Speed
from posggym.wrappers import FlattenObservations


def test_flatten_observation():
    obs_dim = (3, 1, 1)
    env = posggym.make(
        "Driving-v1",
        disable_env_checker=True,
        num_agents=2,
        grid="7x7RoundAbout",
        obs_dim=obs_dim,
    )
    wrapped_env = FlattenObservations(env)

    obs, info = env.reset()
    wrapped_obs, wrapped_obs_info = wrapped_env.reset()

    grid_width, grid_height = 7, 7
    obs_depth, obs_width = (obs_dim[0] + obs_dim[1] + 1), 2 * obs_dim[2] + 1
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
            ),  # current coord
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
