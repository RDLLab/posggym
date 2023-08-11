"""Checks that the core posggym model API is implemented as expected."""
from typing import Dict, List

from gymnasium import spaces

import posggym.model as M
from posggym.utils import seeding


class ExampleModel(M.POSGModel[int, int, int]):
    """Example discrete testing model."""

    def __init__(self):
        self.possible_agents = (0, 1)
        self.action_spaces = {i: spaces.Discrete(2) for i in self.possible_agents}
        self.observation_spaces = {i: spaces.Discrete(2) for i in self.possible_agents}
        self.is_symmetric = True

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: int) -> List[str]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> int:
        return 0

    def sample_initial_obs(self, state: int) -> Dict[str, int]:
        return {i: 0 for i in self.possible_agents}

    def step(self, state: int, actions: Dict[str, int]) -> M.JointTimestep[int, int]:
        return M.JointTimestep(
            0,
            {i: 0 for i in self.possible_agents},
            {i: 0 for i in self.possible_agents},
            {i: False for i in self.possible_agents},
            {i: False for i in self.possible_agents},
            False,
            {i: {} for i in self.possible_agents},
        )


def test_posggym_model():
    """Tests general posggym model API."""
    model = ExampleModel()

    assert model.state_space is None
    assert model.reward_ranges == {
        i: (-float("inf"), float("inf")) for i in model.possible_agents
    }
    assert model.spec is None
    assert model._rng is None  # pylint: disable=protected-access
