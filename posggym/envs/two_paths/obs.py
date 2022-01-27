"""Observations for the Two-Paths problem """
from typing import Tuple

import numpy as np
from posg.model import Observation

import posg.envs.two_paths.grid as grid_lib

# Cell obs
OPPONENT = 0
WALL = 1
EMPTY = 2

CELL_OBS = [OPPONENT, WALL, EMPTY]
CELL_OBS_STR = ["X", "#", "0"]


class TPObs(Observation):
    """An observation for the Two-Paths problem

    Includes an observation of the contents of the adjacent cells, and if a
    terminal state was reached (either runner was caught or reached the goal).

    The terminal observation is needed for the infinite horizon mode.
    """

    def __init__(self,
                 adj_obs: Tuple[int, ...],
                 terminal: bool,
                 grid: grid_lib.TPGrid):
        self.adj_obs = adj_obs
        self.terminal = terminal

    def as_vec(self, dtype=np.float32) -> np.ndarray:
        # need to use one hot encodings
        return np.concatenate([
            np.concatenate(np.eye(len(CELL_OBS))[np.array(self.adj_obs)]),
            np.array([int(self.terminal)])
        ]).astype(dtype)

    def __str__(self):
        adj_obs_str = ",".join(CELL_OBS_STR[i] for i in self.adj_obs)
        return f"<({adj_obs_str}),{self.terminal}>"

    def __eq__(self, other):
        return (
            (self.adj_obs, self.terminal)
            == (other.adj_obs, other.terminal)
        )

    def __hash__(self):
        return hash((self.adj_obs, self.terminal))
