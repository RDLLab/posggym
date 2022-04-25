"""A uniform random bot policy."""

import posggym.model as M
from posggym.bots.core import BotPolicy


class RandomPolicy(BotPolicy):
    """Uniform random policy."""

    def __init__(self, model: M.POSGModel, agent_id: M.AgentID):
        self._model = model
        self._agent_id = agent_id

    @property
    def agent_id(self) -> M.AgentID:
        return self._agent_id

    def step(self, obs: M.Observation) -> M.Action:
        return self.get_action()

    def get_action(self) -> M.Action:
        return self._model.action_spaces[self._agent_id].sample()

    def update(self, action: M.Action, obs: M.Observation) -> None:
        return None

    def reset(self) -> None:
        return None
