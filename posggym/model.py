"""The model data structure."""
import abc
import enum
from typing import Tuple, NamedTuple, Optional, Any, Sequence, Dict

from gym import spaces

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
    done: bool
    outcomes: Optional[Tuple[Outcome, ...]]


class Belief(abc.ABC):
    """An abstract belief class."""

    @abc.abstractmethod
    def sample(self) -> State:
        """Return a state from the belief."""

    @abc.abstractmethod
    def sample_k(self, k: int) -> Sequence[State]:
        """Sample k states from the belief."""

    @abc.abstractmethod
    def get_dist(self) -> Dict[State, float]:
        """Get belief as a distribution: S -> prob map."""

    def sample_belief_dist(self, num_samples: int) -> Dict[State, float]:
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
    model for use in simulation-based planners (e.g. POMCP).

    The API includes implementions of the following,

    Attributes
    ----------
        n_agents : the number of agents in the environment
        action_spaces : the list of actions for each agent (A_0, ..., A_n)
        b_0 : the initial belief over states

    Functions
    ---------
        step : the generative step function G(s, a) -> (s', o, r, done)

    """

    # pylint: disable=unused-argument
    def __init__(self, n_agents: int, **kwargs):
        self.n_agents = n_agents

    @property
    @abc.abstractmethod
    def state_space(self) -> spaces.Space:
        """Get the state space."""

    @property
    @abc.abstractmethod
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        """Get the action space for each agent."""

    @property
    @abc.abstractmethod
    def obs_spaces(self) -> Tuple[spaces.Space, ...]:
        """Get the observation space for each agent."""

    @property
    @abc.abstractmethod
    def reward_ranges(self) -> Tuple[Tuple[Reward, Reward], ...]:
        """Get the minimum and maximum  possible rewards for each agent."""

    @property
    @abc.abstractmethod
    def initial_belief(self) -> Belief:
        """Get the initial belief over states."""

    @abc.abstractmethod
    def get_agent_initial_belief(self,
                                 agent_id: AgentID,
                                 obs: Observation) -> Belief:
        """Get the initial obs conditioned belief for a given agent."""

    def sample_initial_state(self) -> State:
        """Sample an initial state from initial belief."""
        return self.initial_belief.sample()

    @abc.abstractmethod
    def sample_initial_obs(self, state: State) -> JointObservation:
        """Sample an initial observation given initial state."""

    def sample_initial_state_and_obs(self) -> Tuple[State, JointObservation]:
        """Sample initial state and observations for an episode.

        Returns an initial state, sampled from the initial belief, and an
        initial joint observation, sampled from initial state.
        """
        state = self.sample_initial_state()
        joint_obs = self.sample_initial_obs(state)
        return state, joint_obs

    @abc.abstractmethod
    def step(self, state: State, actions: JointAction) -> JointTimestep:
        """Perform generative step."""

    @abc.abstractmethod
    def is_done(self, state: State) -> bool:
        """Check if state is a terminal episode state."""

    @abc.abstractmethod
    def get_outcome(self, state: State) -> Tuple[Outcome, ...]:
        """Get outcome for each agent for given state.

        This function can be used to determine if an episode ended in a
        win, draw, loss, or undefined (for models with no winning or losing).
        """


class POSGFullModel(POSGModel, abc.ABC):
    """A Fully definte Partially Observable Stochastic Game model.

    This class includes implementions for all components of a POSG, including
    the

    Attributes
    ----------
        n_agents : the number of agents in the environment
        state_space : the list of all states, S
        action_spaces : the list of actions for each agent (A_0, ..., A_n)
        obs_spaces: the list of observations for each agent (O_0, ..., O_n)
        b_0 : the initial belief over states

    Functions
    ---------
        transition_fn : the trainsition function T(s, a, s')
        obs_fn : the observation function Z(o, s', a)
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
    def obs_fn(self,
               obs: JointObservation,
               next_state: State,
               actions: JointAction) -> float:
        """Observation function Pr(obs | next_state, action)."""

    @abc.abstractmethod
    def reward_fn(self,
                  state: State,
                  actions: JointAction) -> JointReward:
        """Reward Function R: S X (a_0, ..., a_n) -> (r_0, ..., r_n)."""
