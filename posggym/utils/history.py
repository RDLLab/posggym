"""Utilities for storing and managing agent action-observation histories."""
from typing import Dict, Generic, List, Optional, Tuple

import posggym.model as M


class AgentHistory(Generic[M.ActType, M.ObsType]):
    """An Action-Observation history for a single agent in a POSG environment.

    A History is an ordered Tuple of (Action, Observation) tuples with one
    entry for each time step in the environment.
    """

    def __init__(
        self, history: Tuple[Tuple[Optional[M.ActType], Optional[M.ObsType]], ...]
    ):
        self.history = history
        self.t = len(history) - 1

    def extend(self, action: M.ActType, obs: M.ObsType) -> "AgentHistory":
        """Extend the current history with given action, observation pair."""
        new_history = list(self.history)
        new_history.append((action, obs))
        return AgentHistory(tuple(new_history))

    def get_sub_history(self, horizon: int) -> "AgentHistory":
        """Get a subset of history up to given horizon."""
        assert 0 < horizon <= len(self.history), (
            "Cannot get sub history horizon must be 0 < horizon <= "
            f"len(history: 0 < {horizon} <= len(self.history) invalid"
        )
        if horizon == self.horizon:
            return self
        return AgentHistory(self.history[:horizon])

    def get_last_step(self) -> Tuple[Optional[M.ActType], Optional[M.ObsType]]:
        """Get the last step in the history."""
        return self.history[-1]

    @property
    def horizon(self) -> int:
        """Get the time horizon of this history.

        This is equal to the number of steps taken in the history
        """
        return len(self.history)

    @classmethod
    def get_init_history(cls, obs: Optional[M.ObsType] = None) -> "AgentHistory":
        """Get Initial history."""
        if obs is None:
            return cls(())
        return cls(((None, obs),))

    def __len__(self) -> int:
        return len(self.history)

    def __hash__(self):
        return hash(self.history)

    def __eq__(self, other):
        if not isinstance(other, AgentHistory):
            return False
        return self.history == other.history

    def __str__(self):
        h_str = []
        for a, o in self.history:
            h_str.append(f"({a}, {o})")
        h_str = ",".join(h_str)
        return f"h_{self.t}=<{h_str}>"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, key):
        return self.history[key]

    def __iter__(self):
        return _AgentHistoryIterator(self)


class _AgentHistoryIterator:
    def __init__(self, history: AgentHistory):
        self.history = history
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx < len(self.history.history):
            self._idx += 1
            return self.history[self._idx - 1]
        raise StopIteration


class JointHistory:
    """A joint history for all agents in the environment."""

    def __init__(self, agent_histories: Dict[str, AgentHistory]):
        self.agent_histories = agent_histories
        self.agent_ids = sorted(agent_histories.keys())
        self.num_agents = len(self.agent_histories)

    @classmethod
    def get_init_history(
        cls, agent_ids: List[str], obs: Optional[Dict[str, M.ObsType]] = None
    ) -> "JointHistory":
        """Get Initial joint history."""
        if obs is None:
            return cls({i: AgentHistory.get_init_history() for i in agent_ids})
        return cls({i: AgentHistory.get_init_history(obs[i]) for i in agent_ids})

    def get_agent_history(self, agent_id: str) -> AgentHistory:
        """Get the history of given agent."""
        return self.agent_histories[agent_id]

    def extend(
        self, action: Dict[str, M.ActType], obs: Dict[str, M.ObsType]
    ) -> "JointHistory":
        """Extend the current history with given action, observation pair."""
        new_agent_histories = {
            i: self.agent_histories[i].extend(action[i], obs[i]) for i in self.agent_ids
        }
        return JointHistory(new_agent_histories)

    def get_sub_history(self, horizon: int) -> "JointHistory":
        """Get a subset of history up to given horizon."""
        sub_agent_histories = {
            i: self.agent_histories[i].get_sub_history(horizon) for i in self.agent_ids
        }
        return JointHistory(sub_agent_histories)

    def get_history_tm1(self) -> "JointHistory":
        """Get history at time t-1."""
        sub_agent_histories = {
            i: AgentHistory(self.agent_histories[i][:-1]) for i in self.agent_ids
        }
        return JointHistory(sub_agent_histories)

    def __len__(self) -> int:
        return len(self.agent_histories[self.agent_ids[0]])

    def __hash__(self):
        return hash(tuple(self.agent_histories[i] for i in self.agent_ids))

    def __eq__(self, other):
        if not isinstance(other, JointHistory):
            return False
        return self.agent_histories == other.agent_histories

    def __getitem__(self, key):
        # get actions and observations at time step key
        actions = {i: h[key][0] for i, h in self.agent_histories.items()}
        obs = {i: h[key][1] for i, h in self.agent_histories.items()}
        return actions, obs

    def __str__(self):
        h_str = [f"{i} {h}" for i, h in self.agent_histories.items()]
        h_str.insert(0, "<JointHistory:")
        h_str.append(">")
        return "\n".join(h_str)

    def __repr__(self):
        return self.__str__()
