"""Test models for posggym."""
from typing import Dict, Tuple, SupportsFloat

from gymnasium import spaces

import posggym.model as M


class TestModel1(M.POSGModel):
    """For `test_registration.py` to check `env.make` can import and register env."""

    def __init__(self):
        self.possible_agents = [0, 1]
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(1)

    @property
    def observation_first(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def state_space(self) -> spaces.Space:
        return spaces.Discrete(1)

    @property
    def action_spaces(self) -> Dict[M.AgentID, spaces.Space]:
        return {i: spaces.Discrete(1) for i in self.possible_agents}

    @property
    def observation_spaces(self) -> Dict[M.AgentID, spaces.Space]:
        return {i: spaces.Discrete(1) for i in self.possible_agents}

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[SupportsFloat, SupportsFloat]]:
        return {i: (0, 1) for i in self.possible_agents}
