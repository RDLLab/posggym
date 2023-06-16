"""The Policy class defining the core API for posggym.agents policies."""
from __future__ import annotations

import abc
import copy
from typing import TYPE_CHECKING, Any, Dict, Generic

from posggym.model import ActType, ObsType


if TYPE_CHECKING:
    from posggym.agents.registration import PolicySpec
    from posggym.agents.utils.action_distributions import ActionDistribution
    from posggym.model import AgentID, POSGModel
    from posggym.utils.history import AgentHistory


# Convenient type definitions
PolicyID = str
PolicyState = Dict[str, Any]


class Policy(abc.ABC, Generic[ActType, ObsType]):
    """Abstract policy interface."""

    # PolicySpec used to generate policy instance
    # This is set when policy is made using make function
    spec: PolicySpec | None = None

    # Whether the policy expects the full state as it's observation or not
    observes_state: bool = False

    def __init__(self, model: POSGModel, agent_id: AgentID, policy_id: PolicyID):
        self.model = model
        self.agent_id = agent_id
        self.policy_id = policy_id
        self._state = self.get_initial_state()

    def step(self, obs: ObsType) -> ActType:
        """Get the next action from the policy.

        This function updates the policy's current internal state and computes the
        next action.

        Arguments
        ---------
        obs: the latest observation.

        Returns
        -------
        action: the next action

        """
        self._state = self.get_next_state(obs, self._state)
        action = self.sample_action(self._state)
        self._state["action"] = action
        return action

    def reset(self, *, seed: int | None = None):
        """Reset the policy to it's initial state.

        This resets the policy's internal state, and should be called at the start of
        each episode.

        Subclasses that use random number generators (RNG) should seed override this
        method and seed the RNG if a `seed` is provided. The expected behaviour is that
        this is provided once by the user, just after the policy is first created
        (before it interacts with an environment.)

        Arguments
        ---------
        seed: seed for random number generator.

        """
        self._state = self.get_initial_state()

    def get_initial_state(self) -> PolicyState:
        """Get the policy's initial state.

        Subclasses that utilize custom internal states (e.g. RNN policies) should
        override this method, but should first call `super.().get_initial_state()` to
        get the base policy state can then be extended.

        Returns
        -------
        initial_state: initial policy state

        """
        return {"action": None}

    @abc.abstractmethod
    def get_next_state(
        self,
        obs: ObsType,
        state: PolicyState,
    ) -> PolicyState:
        """Get the next policy state.

        Subclasses that utilize custom internal states (e.g. RNN policies) should
        override this method, but should first call `super.().get_next_state()` to get
        the base policy state can then be extended.

        Arguments
        ---------
        action: the last action performed, may be None if this is the first update
        obs: the observation received
        state: the policy's state before action was performed and obs received

        Returns
        -------
        next_state: the next policy state

        """

    @abc.abstractmethod
    def sample_action(self, state: PolicyState) -> ActType:
        """Get action given agent's current state."""

    @abc.abstractmethod
    def get_pi(self, state: PolicyState) -> ActionDistribution:
        """Get policy's distribution over actions for given policy state."""

    @abc.abstractmethod
    def get_value(self, state: PolicyState) -> float:
        """Get a value estimate of a history."""

    def set_state(self, state: PolicyState):
        """Set the policy's internal state.

        Subclasses that utilize custom internal states (e.g. RNN policies) may wish to
        override this method, to set any attributes used for the by the class to store
        policy state.

        Arguments
        ---------
        state: the new policy state

        Raises
        ------
        AssertionError: if new policy state is not valid.

        """
        if self._state is not None and state is not None:
            assert all(k in state for k in self._state), f"Invalid policy state {state}"
        self._state = state

    def get_state(self) -> PolicyState:
        """Get the policy's current state.

        Returns
        -------
        state: policy's current internal state.

        """
        return copy.deepcopy(self._state)

    def get_state_from_history(self, history: AgentHistory) -> PolicyState:
        """Get the policy's state given history.

        This function essentially unrolls the policy using the actions and observations
        contained in the agent history.

        Arguments
        ---------
        history: the history

        Returns
        -------
        state: policy state given history

        """
        state = self.get_initial_state()
        for a, o in history:
            state["action"] = a
            state = self.get_next_state(o, state)
        return state

    def close(self):
        """Close policy and perform any necessary cleanup.

        Should be overridden in subclasses as necessary.
        """
        pass
