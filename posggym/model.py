"""The model data structure."""
from __future__ import annotations

import abc
import dataclasses
import enum
import random
from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    List,
    Optional,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from gymnasium import spaces

from posggym import error
from posggym.utils import seeding


if TYPE_CHECKING:
    from posggym.envs.registration import EnvSpec


AgentID = Union[int, str]
StateType = TypeVar("StateType")
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


@dataclasses.dataclass(order=True)
class JointTimestep(Generic[StateType, ObsType]):
    """Stores values returned by model after a single step.

    Supports iteration.

    A dataclass is used instead of a Namedtuple so that generic typing is seamlessly
    supported.

    """

    state: StateType
    observations: Dict[AgentID, ObsType]
    rewards: Dict[AgentID, SupportsFloat]
    terminated: Dict[AgentID, bool]
    truncated: Dict[AgentID, bool]
    all_done: bool
    info: Dict[AgentID, Dict]

    def __iter__(self):
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)


class POSGModel(abc.ABC, Generic[StateType, ObsType, ActType]):
    r"""A Partially Observable Stochastic Game model.

    This class defines functions and attributes necessary for a generative POSG
    model for use in simulation-based planners (e.g. MCTS), and for reinforcement
    learning.

    The main API methods that users of this class need to know are:

    - :meth:`get_agents` - Returns the IDs of agents currently active for a given state
    - :meth:`sample_initial_state` - Returns an a sampled initial state of the
      environment.
    - :meth:`sample_initial_obs` - Returns a sampled observation given an initial state.
    - :meth:`step` - The generative step function which given a state and agent actions,
      returns the next state, agent observations, agent rewards, whether the environment
      has terminated or truncated for each agent, whether the environment has reached
      a terminal state for all agents, and information from the environment about the
      step
    - :meth:`seed` - Sets the seed for the models random number generator.

    Additionally, the main API attributes are:

    - :attr:`possible_agents` - All possible agents that may appear in the environment
      across all states.
    - :attr:`state_space` - The space of all environment states. Noting that an explicit
      state space definition is not needed by many simulation-based algorithms
      (including RL and MCTS) and can be hard to define so implementing this property
      should be seen as optional. In cases where it is not implemented it should be
      None.
    - :attr:`action_spaces` - The action space for each agent
    - :attr:`observations_spaces` - The observation space for each agent
    - :attr:`reward_ranges` - The minimum and maximum possible rewards over an episode
      for each agent. The default reward range is set to :math:`(-\infty,+\infty)`.
    - :attr:`observation_first` - Whether the environment is observation or action first
    - :attr:`is_symmetric` - Whether the environment is symmetric or asymmetric. That is
      whether all agents are identical irrespective of their ID (i.e. same actions,
      observation, and reward spaces and dynamics)
    - :attr:`spec` - An environment spec that contains the information used to
      initialize the environment from :meth:`posggym.make`
    - :attr:`rng` - the model's internal random number generator (RNG).

    Custom environments should inherit from this class and implement the
    :meth:`get_agents`, :meth:`sample_initial_state`, :meth:`step` methods and the
    :attr:`possible_agents`, :attr:`action_spaces`, :attr:`observations_spaces`,
    :attr:`observation_first`, :attr:`is_symmetric`, :attr:`rng` attributes.

    Furthermore, if the environment is an `observation_first` environment (which is the
    case for most environments) it must also implement the
    :meth:`sample_initial_obs` method.

    Custom environments may optionally provide implementations for the
    :meth:`sample_agent_initial_state` method (`observation_first` environments only)
    and :attr:`state_space` attribute.

    """

    # EnvSpec used to instantiate env instance this model is for
    # This is set when env is made using posggym.make function
    spec: Optional["EnvSpec"] = None

    # All agents that may appear in the environment
    possible_agents: Tuple[AgentID, ...]
    # State space
    state_space: spaces.Space | None = None
    # Action space for each agent
    action_spaces: Dict[AgentID, spaces.Space]
    # Observation space for each agent
    observation_spaces: Dict[AgentID, spaces.Space]
    # Whether environment is observation or action first.
    observation_first: bool
    # Whether the environment is symmetric or not (is asymmetric)
    is_symmetric: bool
    # Random number generator, created as needed by `rng` method.
    _rng: seeding.RNG | None = None

    @property
    def reward_ranges(self) -> Dict[AgentID, Tuple[SupportsFloat, SupportsFloat]]:
        r"""A mapping from Agent ID to min and max possible rewards for that agent.

        Each reward tuple corresponding to the minimum and maximum possible rewards for
        a given agent over an episode. The default reward range for each agent is set to
        :math:`(-\infty,+\infty)`.

        Returns
        -------
        Dict[AgentID, Tuple[SupportsFloat, SupportsFloat]]

        """
        return {i: (-float("inf"), float("inf")) for i in self.possible_agents}

    @abc.abstractmethod
    def get_agents(self, state: StateType) -> List[AgentID]:
        """Get list of IDs for all agents that are active in given state.

        These list of active agents may change depending on state.

        For any environment where the number of agents remains constant during AND
        across episodes. This will be :attr:`possible_agents`, independent of state.

        Returns
        -------
        List[AgentID]
          List of IDs for all agents that active in given state,

        """

    @abc.abstractmethod
    def sample_initial_state(self) -> StateType:
        """Sample an initial state.

        Returns
        -------
        StateType
          An initial state.

        """

    def sample_initial_obs(self, state: StateType) -> Dict[AgentID, ObsType]:
        """Sample initial agent observations given an initial state.

        This method must be implemented for `observation_first` models.

        Arguments
        ---------
        state : StateType
          The initial state.

        Returns
        -------
        Dict[AgentID, ObsType]
          A mapping from AgentID to initial observation.

        Raises
        ------
        AssertionError
          If this method is called for action first model.

        """
        if self.observation_first:
            raise NotImplementedError
        raise AssertionError(
            "Model is action_first so expects agents to perform an action "
            "before the initial observations are generated. This is done "
            "using the step() function."
        )

    @abc.abstractmethod
    def step(
        self, state: StateType, actions: Dict[AgentID, ActType]
    ) -> JointTimestep[StateType, ObsType]:
        """Perform generative step.

        For custom environments that have win/loss or success/fail conditions, you are
        encouraged to include this information in the `info` property of the returned
        value. We suggest using the the "outcomes" key with an instance of the
        ``Outcome`` class for values.

        """

    @property
    @abc.abstractmethod
    def rng(self) -> seeding.RNG:
        """Return the model's internal random number generator (RNG).

        Initializes RNG with a random seed if not yet initialized.

        `posggym` models and environments support the use of both the python built-in
        random library and the numpy library, unlike `gymnasium` which only explicitly
        supports the numpy library. Support for the built-in library is included as it
        can be 2-3X faster than the numpy library when drawing single samples, providing
        a significant speed-up for many environments.

        Theres also nothing stopping users from using other RNG libraries so long as
        they implement the model API. However, explicit support in the form of tests
        and type hints is only provided for the random and numpy libraries.

        Returns
        -------
        rng : model's internal random number generator.

        """

    def seed(self, seed: Optional[int] = None):
        """Set the seed for the model RNG.

        Also handles seeding for the action, observation, and (if it exists) state
        spaces.

        """
        if isinstance(self.rng, random.Random):
            self._rng, seed = seeding.std_random(seed)
        elif isinstance(self.rng, np.random.Generator):
            self._rng, seed = seeding.np_random(seed)
        else:
            raise error.UnseedableEnv(
                "{self.__class__.__name__} unseedable. Please ensure the model has "
                "implemented the rng property. The model class must also overwrite "
                "the `seed` method if it uses a RNG not from the `random` or "
                "`numpy.random` libraries."
            )

        seed += 1
        for act_space in self.action_spaces.values():
            act_space.seed(seed)
            seed += 1

        for obs_space in self.observation_spaces.values():
            obs_space.seed(seed)
            seed += 1

        if self.state_space is not None:
            self.state_space.seed(seed)

    def sample_agent_initial_state(self, agent_id: AgentID, obs: ObsType) -> StateType:
        """Sample an initial state for an agent given it's initial observation.

        Only applicable in observation first environments.

        It is optional to implement this method but can helpful in environments that are
        used for planning and where there are a huge number of possible initial states.

        Arguments
        ---------
        agent_id : Union[int, str]
          The ID of the agent to get initial state for.
        obs : ObsType
          The initial observation of the agent.

        Returns
        -------
        StateType
          An initial state for the agent conditioned on their initial observation.

        Raises
        ------
        AssertionError
          If this method is called for action first model.
        NotImplementedError
          If this method is not implemented.

        """
        if self.observation_first:
            raise NotImplementedError
        raise AssertionError(
            "The `sample_agent_initial_state` method is not supported for action first "
            "environments. Use the `sample_initial_state` method instead."
        )


class POSGFullModel(POSGModel[StateType, ObsType, ActType], abc.ABC):
    r"""A Fully defined Partially Observable Stochastic Game model.

    This class inherits from the :py:class:`POSGModel`, adding implementions for all
    individual components of a POSG. It is designed for use by planners which utilize
    all model components (e.g. dynamic programming and full-width planners).

    The additional methods that need to be implemented are:

    - :meth:`get_initial_belief` - the initial belief distribution :math:`b_{0}`,
      mapping initial states to probabilities.
    - :meth:`transition_fn` - the transition function
      :math:`T(s, a, s') \rightarrow [0, 1]` defining :math:`Pr(s'|s, a)`, the
      probability of getting next state `s'` given the environment was in state `s` and
      joint action `a` was performed.
    - :meth:`observation_fn` - the observation function
      :math:`Z(o, s', a) \rightarrow [0, 1]` defining :math:`Pr(o|s', a)`, the
      probability of joint observation `o` given the joint action `a` was performed and
      the environment ended up in state `s'`
    - :meth:`reward_fn` : the reward function
      :math:`R(s, a) \rightarrow \mathbf{R}^n` where `n` is the number of agents,
      defines the reward each agent recieves given joint action `a` was performed in
      state `s`.


    """

    @abc.abstractmethod
    def get_initial_belief(self) -> Dict[StateType, float]:
        """The initial belief distribution: :math:`b_{0}`.

        Returns
        -------
        Dict[StateType, float]
           :math:`Pr(s_{0}=s)` the initial probability of each state. If a state is not
           included in the initial distribution object, it should be assumed to have
           probability 0.

        """

    @abc.abstractmethod
    def transition_fn(
        self, state: StateType, actions: Dict[AgentID, ActType], next_state: StateType
    ) -> float:
        """Transition function :math:`T(s', a, s)`.

        Arguments
        ---------
        state : StateType
          the state the environment was in
        actions : Dict[AgentID, ActType]
          the joint action performed
        next_state : StateType
          the state of the environment after actions were performed

        Returns
        -------
        float
          :math:`Pr(s'|s, a)`, the probability of getting next state `s'` given the
          environment was in state `s` and joint action `a` was performed.

        """

    @abc.abstractmethod
    def observation_fn(
        self,
        obs: Dict[AgentID, ObsType],
        next_state: StateType,
        actions: Dict[AgentID, ActType],
    ) -> float:
        """Observation function :math:`Z(o, s', a) `.

        Arguments
        ---------
        obs : Dict[AgentID, ObsType]
          the observation received
        actions : Dict[AgentID, ActType]
          the joint action performed
        next_state : StateType
          the state of the environment after actions were performed

        Returns
        -------
        float
          :math:`Pr(o|s', a)`, the probability of joint observation `o` given the joint
          action `a` was performed and the environment ended up in state `s'`.

        """

    @abc.abstractmethod
    def reward_fn(
        self, state: StateType, actions: Dict[AgentID, ActType]
    ) -> Dict[AgentID, SupportsFloat]:
        """The reward Function :math:`R(s, a)`.

        Arguments
        ---------
        state : StateType
          the state the environment was in
        actions : Dict[AgentID, ActType]
          the joint action performed

        Returns
        -------
        Dict[AgentID, SupportsFloat]
          The reward each agent recieves given joint action `a` was performed in state
          `s`.

        """


class Outcome(enum.Enum):
    """An enum for final episode Outcome of an agent.

    This is supplied for user convinience. For environments where agents can WIN/LOSE,
    this class can be used to supply that information to users in a standard format via
    the ``info`` return value of the :meth:`POSGModel.step()` function.

    Has the following possible values:

    - :attr:`LOSS` = -1
    - :attr:`DRAW` = 0
    - :attr:`WIN` = 1
    - :attr:`NA` = None

    """

    LOSS = -1
    DRAW = 0
    WIN = 1
    NA = None

    def __str__(self):
        return self.name
