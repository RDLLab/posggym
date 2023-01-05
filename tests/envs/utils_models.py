"""Test models for posggym."""
from typing import Dict, List

from gymnasium import spaces

import posggym.model as M
from posggym.utils import seeding


class TestModel(M.POSGModel):
    """Basic test model."""

    def __init__(self):
        self.possible_agents = (0, 1)
        self.action_spaces = {i: spaces.Discrete(2) for i in self.possible_agents}
        self.observation_spaces = {i: spaces.Discrete(2) for i in self.possible_agents}

    @property
    def observation_first(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: int) -> List[M.AgentID]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> int:
        return 0

    def sample_initial_obs(self, state: int) -> Dict[M.AgentID, int]:
        return {i: 0 for i in self.possible_agents}

    def step(
        self, state: int, actions: Dict[M.AgentID, int]
    ) -> M.JointTimestep[int, int]:
        return M.JointTimestep(
            0,
            {i: 0 for i in self.possible_agents},
            {i: 0 for i in self.possible_agents},
            {i: False for i in self.possible_agents},
            {i: False for i in self.possible_agents},
            False,
            {i: {} for i in self.possible_agents},
        )
