"""The Policy class defining the core API for posggym.agents policies."""
from __future__ import annotations

import abc
import copy
from typing import TYPE_CHECKING, Any, Dict, Generic

from posggym.model import ActType, ObsType


if TYPE_CHECKING:
    from posggym.agents.registration import PolicySpec
    from posggym.agents.utils.action_distributions import ActionDistribution
    from posggym.model import POSGModel
    from posggym.utils.history import AgentHistory


# Convenient type definitions
PolicyID = str
PolicyState = Dict[str, Any]


class Policy(abc.ABC, Generic[ActType, ObsType]):
    """The main POSGGym Agents class implementing a policy for a single agent.

    This class defines the core POSGGym Agents policy API for a single agent acting in
    a partially observable, multi-agent (POSG) environment. The policy can be used to
    interact with an environment vis the :meth:`step` method, which returns the next
    action to take given the latest observation.

    The API is designed to be as general as possible while also following a similar
    design to the POSGGym Env API (which in turn follows the Gymnasium API).

    The main Agent Policy API methods that users of this class need to know are:

    - :meth:`step`
    - :meth:`reset`
    - :meth:`close`

    And the main attributes are:

    - :attr:`model`
    - :attr:`agent_id`
    - :attr:`policy_id`
    - :attr:`spec`
    - :attr:`observes_state`

    The policy API also supports finer grained control over the policy's internal state
    via the following methods:

    - :meth:`get_initial_state`
    - :meth:`get_next_state`
    - :meth:`sample_action`
    - :meth:`get_pi`
    - :meth:`get_value`
    - :meth:`set_state`
    - :meth:`get_state`
    - :meth:`get_state_from_history`

    The policy maintains an internal state, which is a dictionary and stored in the
    :attr:`_state` attribute. This state is updated each time the :meth:`step` method is
    called and reset to an initial state whenever the :meth:`reset` method is called.
    By default the policy state is an empty dictionary. However, subclasses should
    extend this state to include any additional information required by the policy to
    select the next action given it's action-observation history.

    The policy also maintains a reference to the last action it returned by the
    :meth:`step` method, which is stored in the :attr:`_last_action` attribute. This
    attribute is set to None when the policy is first created and whenever the
    :meth:`reset` method is called. It is used to update the policy's internal state
    each time the :meth:`step` method is called.


    """

    # PolicySpec used to initialize policy instance
    # This is set when policy is made using posggym.agents.make function
    spec: PolicySpec | None = None

    # Whether the policy expects the full state as it's observation or not
    observes_state: bool = False

    def __init__(self, model: POSGModel, agent_id: str, policy_id: PolicyID):
        self.model = model
        self.agent_id = agent_id
        self.policy_id = policy_id
        self._state = self.get_initial_state()
        self._last_action = None

    def step(self, obs: ObsType) -> ActType:
        """Get the next action from the policy.

        This function updates the policy's current internal state given the most recent
        observation, and returns the next action for the policy.

        Arguments
        ---------
        obs : ObsType
            the latest observation.

        Returns
        -------
        action : ActType
            the next action

        """
        self._state = self.get_next_state(self._last_action, obs, self._state)
        self._last_action = self.sample_action(self._state)
        return self._last_action

    def reset(self, *, seed: int | None = None):
        """Reset the policy to it's initial state.

        This resets the policy's internal state, and should be called at the start of
        each episode.

        Subclasses that use random number generators (RNG) should override this
        method and seed the RNG if a `seed` is provided. The expected behaviour is that
        this is that the seed provided once by the user, just after the policy is first
        created and before it interacts with an environment.

        Arguments
        ---------
        seed : int, optional
            seed for random number generator.

        """
        self._state = self.get_initial_state()
        self._last_action = None

    def close(self):
        """Close policy and perform any necessary cleanup.

        Should be overridden in subclasses as necessary.
        """
        pass

    def get_initial_state(self) -> PolicyState:
        """Get the policy's initial state.

        Subclasses that utilize custom internal states (e.g. RNN policies) should
        override this method, but should first call `super.().get_initial_state()` to
        get the base policy state that can then be extended.

        Returns
        -------
        initial_state : PolicyState
            the initial policy state

        """
        return {}

    @abc.abstractmethod
    def get_next_state(
        self,
        action: ActType | None,
        obs: ObsType,
        state: PolicyState,
    ) -> PolicyState:
        """Get the next policy state.

        Subclasses must implement this method.

        Arguments
        ---------
        action : ActType, optional
            the action performed. May be None if this is the first observation.
        obs : ObsType
            the observation received
        state : PolicyState
            the policy's state before action was performed and obs received

        Returns
        -------
        next_state : PolicyState
            the next policy state

        """

    @abc.abstractmethod
    def sample_action(self, state: PolicyState) -> ActType:
        """Sample an action given policy's current state.

        If the policy is deterministic then this will return the same action each time,
        given the same state. If the policy is stochastic then the action may change
        each time even if the state is the same.

        Subclasses must implement this method.

        Arguments
        ---------
        state : PolicyState
            the policy's current state

        Returns
        -------
        action : ActType
            the sampled action

        """

    @abc.abstractmethod
    def get_pi(self, state: PolicyState) -> ActionDistribution:
        """Get policy's distribution over actions for given policy state.

        Subclasses must implement this method, and should return an instance of a
        subclass of the
        :py:class:`posggym.agents.utils.action_distributions.ActionDistribution`
        class.

        Arguments
        ---------
        state : PolicyState
            the policy's current state

        Returns
        -------
        pi : ActionDistribution
            the policy's distribution over actions

        """

    @abc.abstractmethod
    def get_value(self, state: PolicyState) -> float:
        """Get a value estimate of a history.

        Subclasses must implement this method, but may set it to raise a
        NotImplementedError if the policy does not support value estimates.

        Arguments
        ---------
        state : PolicyState
            the policy's current state

        Returns
        -------
        value : float
            the value estimate

        """

    def set_state(self, state: PolicyState, last_action: ActType | None = None):
        """Set the policy's internal state.

        Subclasses that utilize custom internal states (e.g. RNN policies) may wish to
        override this method, to set any attributes used for the by the class to store
        policy state.

        Arguments
        ---------
        state : PolicyState
            the new policy state
        last_action : ActType, optional
            the last action taken by the policy. If not provided then the last action
            will be set to None.

        Raises
        ------
        AssertionError :
            if new policy state is not valid.

        """
        if self._state is not None and state is not None:
            assert all(k in state for k in self._state), f"Invalid policy state {state}"
        self._state = state
        self._last_action = last_action

    def get_state(self) -> PolicyState:
        """Get the policy's current state.

        Returns
        -------
        state : PolicyState
            policy's current internal state.

        """
        return copy.deepcopy(self._state)

    def get_state_from_history(self, history: AgentHistory) -> PolicyState:
        """Get the policy's state given history.

        This function essentially unrolls the policy using the actions and observations
        contained in the agent history.

        Note, this function will return None for the action in the final output state,
        as this would correspond to the action that was selected by the policy to action


        Arguments
        ---------
        history : AgentHistory
            the agent's action-observation history

        Returns
        -------
        state : PolicyState
            policy state given history

        """
        state = self.get_initial_state()
        for a, o in history:
            state = self.get_next_state(a, o, state)
        return state
