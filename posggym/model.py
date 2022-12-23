"""The model data structure."""
from __future__ import annotations

import abc
import enum
from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
)

from gymnasium import spaces


if TYPE_CHECKING:
    from posggym.envs.registration import EnvSpec


AgentID = Union[int, str]
StateType = TypeVar("StateType")
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class Outcome(enum.Enum):
    """Final POSG episode Outcome for an agent."""

    LOSS = -1
    DRAW = 0
    WIN = 1
    NA = None

    def __str__(self):
        return self.name


class JointTimestep(NamedTuple, Generic[StateType, ObsType]):
    """Values returned by model after a single step."""

    state: StateType
    observations: Dict[AgentID, ObsType]
    rewards: Dict[AgentID, SupportsFloat]
    terminated: Dict[AgentID, bool]
    truncated: Dict[AgentID, bool]
    all_done: bool
    outcomes: Dict[AgentID, Outcome] | None
    info: Dict[AgentID, Dict]


class Belief(abc.ABC, Generic[StateType]):
    """An abstract belief class."""

    @abc.abstractmethod
    def sample(self) -> StateType:
        """Return a state from the belief."""

    def sample_k(self, k: int) -> Sequence[StateType]:
        """Sample k states from the belief."""
        return [self.sample() for _ in range(k)]

    def get_dist(self) -> Dict[StateType, float]:
        """Get belief as a distribution: S -> prob map."""
        return self.sample_belief_dist()

    def sample_belief_dist(self, num_samples: int = 1000) -> Dict[StateType, float]:
        """Construct a belief distribution via Monte-Carlo sampling.

        Requires that the State objects for the given belief are hashable.
        """
        b_map: Dict[StateType, float] = {}

        s_prob = 1.0 / num_samples
        for s in (self.sample() for _ in range(num_samples)):
            if s not in b_map:
                b_map[s] = 0.0
            b_map[s] += s_prob

        return b_map


class POSGModel(abc.ABC, Generic[StateType, ObsType, ActType]):
    """A Partially Observable Stochastic Game model.

    This class defines functions and attributes necessary for a generative POSG
    model for use in simulation-based planners (e.g. MCTS).

    The API includes the following,

    Attributes
    ----------
    possible_agents : List[AgentID]
        All agents that may appear in the environment
    observation_first : bool
        whether the environment is observation (True) or action (False) first.
        See the POSGModel.observation_first property function for details.
    state_space : gym.space.Space
        the space of all possible environment states. Note that an explicit
        state space definition is not needed by many simulation-based
        algorithms (including RL and MCTS) and can be hard to define so
        implementing this property should be seen as optional. In cases where
        it is not implemented it should raise a NotImplementedError.
    initial_belief : Belief
        the initial belief over states
    action_spaces : Tuple[gym.space.Space, ...]
        the action space for each agent (A_0, ..., A_n)
    observation_spaces : Tuple[gym.space.Space, ...]
        the observation space for each agent (O_0, ..., O_n)
    reward_ranges : Tuple[Tuple[Reward, Reward], ...]
        the minimum and maximim possible step reward for each agent
    is_symmetric : bool
        whether the environment is symmetric, that is whether all agents are
        identical irrespective of their ID (i.e. same actions, observation, and
        reward spaces and dynamics)

    Methods
    -------
    get_agents :
        returns the IDs of agents currently active for a given state
    step :
        the generative step function
        G(s, a) -> (s', o, r, dones, all_done, outcomes)
    set_seed :
        set the seed for the model's RNG
    sample_initial_state :
        samples an initial state from the initial belief.
    sample_initial_obs :
        samples an initial observation from a state. This function should only
        be used in observation first environments.

    Optional Functions and Attributes
    ---------------------------------
    get_agent_initial_belief :
        get the initial belief for an agent given an initial observation. This
        function is only applicable in observation first environments and is
        optional, since the initial agent belief can be generated from the
        sample_initial_state and sample_initial_obs functions. However,
        implementing this function can speed up the initial belief update for
        problems with a large number of possible starting states.

    """

    # EnvSpec used to instantiate env instance this model is for
    # This is set when env is made using posggym.make function
    spec: Optional["EnvSpec"] = None

    # All agents that may appear in the environment
    possible_agents: Tuple[AgentID, ...]

    @property
    @abc.abstractmethod
    def observation_first(self) -> bool:
        """Get whether environment is observation or action first.

        "Observation first" environments start by providing the agents with an
        observation from the initial belief before any action is taken. Most
        Reinforcement Learning algorithms typically assume this setting.

        "Action first" environments expect the agents to take an action from the initial
        belief before providing an observation. Many planning algorithms use this
        paradigm.

        Note
        ----
        "Action first" environments can always be converted into "Observation first"
          by introducing a dummy initial observation. Similarly, "Action first"
          algorithms can be made compatible with "Observation first" environments by
          introducing a single dummy action for the first step only.

        Returns
        -------
        bool
          ``True`` if environment is observation first, ``False`` if environment is
          action first.

        """

    @property
    @abc.abstractmethod
    def is_symmetric(self) -> bool:
        """Get whether environment is symmetric.

        An environment is "symmetric" if the ID of an agent in the environment does not
        affect the agent in anyway (i.e. all agents have the same action and observation
        spaces, same reward functions, and there are no differences in initial
        conditions all things considered). Classic examples include Rock-Paper-Scissors,
        Chess, Poker. In "symmetric" environments the same "policy" should do equally
        well independent of the ID of the agent the policy is used for.

        If an environment is not "symmetric" then it is "asymmetric", meaning that
        there are differences in agent properties based on the agent's ID. In
        "asymmetric" environments there is no guarantee that the same "policy" will
        work for different agent IDs. Examples include Pursuit-Evasion games, any
        environments where action and/or observation space differs by agent ID.

        Returns
        -------
        bool
          ``True`` if environment is symmetric, ``False`` if environment is asymmetric.

        """

    @property
    @abc.abstractmethod
    def state_space(self) -> spaces.Space:
        """Get the state space."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_spaces(self) -> Dict[AgentID, spaces.Space]:
        """Get the action space for each agent."""

    @property
    @abc.abstractmethod
    def observation_spaces(self) -> Dict[AgentID, spaces.Space]:
        """Get the observation space for each agent."""

    @property
    @abc.abstractmethod
    def reward_ranges(self) -> Dict[AgentID, Tuple[SupportsFloat, SupportsFloat]]:
        """Get the minimum and maximum  possible rewards for each agent."""

    @property
    @abc.abstractmethod
    def initial_belief(self) -> Belief:
        """Get the initial belief over states."""

    def sample_initial_state(self) -> StateType:
        """Sample an initial state from initial belief."""
        return self.initial_belief.sample()

    @abc.abstractmethod
    def get_agents(self, state: StateType) -> List[AgentID]:
        """Get list of IDs for agents that are active in given state.

        This will be :attr:`possible_agents`, independent of state, for any environment
        where the number of agents remains constant during and across episodes.
        """

    @abc.abstractmethod
    def step(
        self, state: StateType, actions: Dict[AgentID, ActType]
    ) -> JointTimestep[StateType, ObsType]:
        """Perform generative step."""

    @abc.abstractmethod
    def set_seed(self, seed: Optional[int] = None):
        """Set the seed for the model RNG."""

    def sample_initial_obs(self, state: StateType) -> Dict[AgentID, ObsType]:
        """Sample an initial observation given initial state."""
        if self.observation_first:
            raise NotImplementedError
        raise AssertionError(
            "Model is action_first so expects agents to perform an action "
            "before the initial observations are generated. This is done "
            "using the step() function."
        )

    def get_agent_initial_belief(
        self, agent_id: AgentID, obs: ObsType
    ) -> Belief[StateType]:
        """Get the initial belief for an agent given it's initial observation.

        Only applicable in observation first environments and is optional.
        """
        if self.observation_first:
            raise NotImplementedError
        raise AssertionError(
            "get_agent_initial_belief is not supported for action first "
            "environments. Instead get initial belief using the initial_belief"
            " property."
        )


class POSGFullModel(POSGModel[StateType, ObsType, ActType], abc.ABC):
    """A Fully definte Partially Observable Stochastic Game model.

    This class includes implementions for all components of a POSG, including
    the

    Attributes
    ----------
    n_agents : the number of agents in the environment
    state_space : the list of all states, S
    action_spaces : the list of actions for each agent (A_0, ..., A_n)
    observation_spaces: the list of observations for each agent (O_0, ..., O_n)
    b_0 : the initial belief over states

    Functions
    ---------
    transition_fn : the trainsition function T(s, a, s')
    observation_fn : the observation function Z(o, s', a)
    reward_fn : the reward function R(s, a)

    This is in addition to the functions and properties defined in the
    POSGModel class.

    """

    @abc.abstractmethod
    def transition_fn(
        self, state: StateType, actions: Dict[AgentID, ActType], next_state: StateType
    ) -> float:
        """Transition function Pr(next_state | state, action)."""

    @abc.abstractmethod
    def observation_fn(
        self,
        obs: Dict[AgentID, ObsType],
        next_state: StateType,
        actions: Dict[AgentID, ActType],
    ) -> float:
        """Observation function Pr(obs | next_state, action)."""

    @abc.abstractmethod
    def reward_fn(
        self, state: StateType, actions: Dict[AgentID, ActType]
    ) -> Dict[AgentID, SupportsFloat]:
        """Reward Function R: S X (a_0, ..., a_n) -> (r_0, ..., r_n)."""
