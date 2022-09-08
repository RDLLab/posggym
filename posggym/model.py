"""The model data structure."""
import abc
import enum
from typing import Tuple, NamedTuple, Optional, Any, Sequence, Dict
from typing import TYPE_CHECKING

from gym import spaces

if TYPE_CHECKING:
    from posggym.envs.registration import EnvSpec

AgentID = int
State = Any
Action = Any
JointAction = Tuple[Action, ...]
Reward = float
JointReward = Tuple[Reward, ...]
Observation = Any
JointObservation = Tuple[Observation, ...]


class Outcome(enum.Enum):
    """Final POSG episode Outcome for an agent."""
    LOSS = -1
    DRAW = 0
    WIN = 1
    NA = None

    def __str__(self):
        return self.name


class JointTimestep(NamedTuple):
    """Values returned by model after a single step."""
    state: State
    observations: JointObservation
    rewards: JointReward
    dones: Tuple[bool, ...]
    all_done: bool
    outcomes: Optional[Tuple[Outcome, ...]]


class Belief(abc.ABC):
    """An abstract belief class."""

    @abc.abstractmethod
    def sample(self) -> State:
        """Return a state from the belief."""

    def sample_k(self, k: int) -> Sequence[State]:
        """Sample k states from the belief."""
        return [self.sample() for _ in range(k)]

    def get_dist(self) -> Dict[State, float]:
        """Get belief as a distribution: S -> prob map."""
        return self.sample_belief_dist()

    def sample_belief_dist(self,
                           num_samples: int = 1000) -> Dict[State, float]:
        """Construct a belief distribution via Monte-Carlo sampling.

        Requires that the State objects for the given belief are hashable.
        """
        b_map: Dict[State, float] = {}

        s_prob = 1.0 / num_samples
        for s in (self.sample() for _ in range(num_samples)):
            if s not in b_map:
                b_map[s] = 0.0
            b_map[s] += s_prob

        return b_map


class POSGModel(abc.ABC):
    """A Partially Observable Stochastic Game model.

    This class defines functions and attributes necessary for a generative POSG
    model for use in simulation-based planners (e.g. MCTS).

    The API includes the following,

    Attributes
    ----------
    n_agents : int
        the number of agents in the environment
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

    Functions
    ---------
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
    spec: "EnvSpec" = None

    # pylint: disable=unused-argument
    def __init__(self, n_agents: int, **kwargs):
        self.n_agents = n_agents

    @property
    @abc.abstractmethod
    def observation_first(self) -> bool:
        """Get whether model is observation (True) or action (False) first.

        Observation first environments start by providing the agents with an
        observation from the initial belief before any action is taken.

        Action first environment expect the agents to take an action from the
        initial belief before providing an observation.

        Note: Observation first environment are equivalent to action first
        environments where all agents know when they are performing the first
        action and there is only a single first action available.
        """

    @property
    @abc.abstractmethod
    def is_symmetric(self) -> bool:
        """Get swhether the environment is symmetric.

        In symmetric environments all agents are identical irrespective of
        their ID (i.e. same actions, observation, and reward spaces and
        dynamics)
        """

    @property
    @abc.abstractmethod
    def state_space(self) -> spaces.Space:
        """Get the state space."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        """Get the action space for each agent."""

    @property
    @abc.abstractmethod
    def observation_spaces(self) -> Tuple[spaces.Space, ...]:
        """Get the observation space for each agent."""

    @property
    @abc.abstractmethod
    def reward_ranges(self) -> Tuple[Tuple[Reward, Reward], ...]:
        """Get the minimum and maximum  possible rewards for each agent."""

    @property
    @abc.abstractmethod
    def initial_belief(self) -> Belief:
        """Get the initial belief over states."""

    def sample_initial_state(self) -> State:
        """Sample an initial state from initial belief."""
        return self.initial_belief.sample()

    @abc.abstractmethod
    def step(self, state: State, actions: JointAction) -> JointTimestep:
        """Perform generative step."""

    @abc.abstractmethod
    def set_seed(self, seed: Optional[int] = None):
        """Set the seed for the model RNG."""

    def sample_initial_obs(self, state: State) -> JointObservation:
        """Sample an initial observation given initial state."""
        if self.observation_first:
            raise NotImplementedError
        raise AssertionError(
            "Model is action_first so expects agents to perform an action "
            "before the initial observations are generated. This is done "
            "using the step() function."
        )

    def get_agent_initial_belief(self,
                                 agent_id: AgentID,
                                 obs: Observation) -> Belief:
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


class POSGFullModel(POSGModel, abc.ABC):
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
    def transition_fn(self,
                      state: State,
                      actions: JointAction,
                      next_state: State) -> float:
        """Transition function Pr(next_state | state, action)."""

    @abc.abstractmethod
    def observation_fn(self,
                       obs: JointObservation,
                       next_state: State,
                       actions: JointAction) -> float:
        """Observation function Pr(obs | next_state, action)."""

    @abc.abstractmethod
    def reward_fn(self,
                  state: State,
                  actions: JointAction) -> JointReward:
        """Reward Function R: S X (a_0, ..., a_n) -> (r_0, ..., r_n)."""
