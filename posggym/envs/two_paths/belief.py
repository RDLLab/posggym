"""A Belief in the Two-Paths problem  """
from typing import Dict, Sequence

import posg.model as M
import posg.model.belief as belief_lib

import posg.envs.two_paths.state as S
import posg.envs.two_paths.grid as grid_lib


class TPB0(M.Belief):
    """The initial belief in a Two-Paths problem """

    def __init__(self, grid: grid_lib.TPGrid, dist_res: int = 1000):
        self._grid = grid
        self._dist_res = dist_res

    def sample(self) -> M.State:
        runner_loc = self._grid.init_runner_loc
        chaser_loc = self._grid.init_chaser_loc
        return S.TPState(runner_loc, chaser_loc, self._grid)

    def sample_k(self, k: int) -> Sequence[M.State]:
        return [self.sample() for _ in range(k)]

    def get_dist(self) -> Dict[M.State, float]:
        return belief_lib.sample_belief_dist(self, self._dist_res)
