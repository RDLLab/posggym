"""Contains the main POSG environment class and functions.

The implementation is heavily inspired by Open AI Gym
https://github.com/openai/gym

And, the more recent, Farama Foundation Gymnasium
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/core.py

"""
from __future__ import annotations

import abc
import copy
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    SupportsFloat,
    Tuple,
    TypeVar,
)

from gymnasium import spaces
from gymnasium.core import RenderFrame

import posggym.model as M


if TYPE_CHECKING:
    from posggym.envs.registration import EnvSpec


class Env(abc.ABC, Generic[M.StateType, M.ObsType, M.ActType]):
    """The main POSGGym class for implementing POSG environments.

    The class encapsulates an environment and a POSG model. The environment maintains an
    internal state and can be interacted with by  multiple agents in parallel through
    the :meth:`step` and :meth:`reset` functions. The POSG model can be accessed via the
    :attr:`model` attribute and exposes model of the environment which can be used for
    planning or for anything else (see :py:class:`posggym.POSGModel` class for details).

    The implementation is heavily inspired by the Farama Foundation Gymnasium (Open AI
    Gym) ([https://github.com/Farama-Foundation/Gymnasium]) and PettingZoo
    (https://github.com/Farama-Foundation/PettingZoo) APIs. It aims to be consistent
    with these APIs and easily compatible with the PettingZoo API.

    The main API methods that users of this class need to know are:

    - :meth:`step`
    - :meth:`reset`
    - :meth:`render`
    - :meth:`close`

    And the main attributes:

    - :attr:`model` - the POSG model of the environment (:py:class:`posggym.POSGModel`)
    - :attr:`possible_agents` - all agents that may appear in the environment
    - :attr:`agents` - the agents currently active in the environment
    - :attr:`action_spaces` - the action space specs for each agent
    - :attr:`observation_spaces` - the observation space specs for each agent
    - :attr:`state` - the current state of the environment
    - :attr:`observation_first` - whether environment is observation or action first
    - :attr:`reward_specs` - the reward specs for each agent

    Environments have additional methods and attributes that provide more environment
    information and access:

    - :attr:`unwrapped`

    """

    # Set this in SOME subclasses
    metadata: Dict[str, Any] = {"render_modes": []}

    # Define render_mode if your environment supports rendering
    render_mode: str | None = None

    # EnvSpec used to instantiate env instance
    # This is set when env is made using posggym.make function
    spec: EnvSpec | None = None

    @abc.abstractmethod
    def step(
        self, actions: Dict[M.AgentID, M.ActType]
    ) -> Tuple[
        Dict[M.AgentID, M.ObsType],
        Dict[M.AgentID, SupportsFloat],
        Dict[M.AgentID, bool],
        Dict[M.AgentID, bool],
        bool,
        Dict[M.AgentID, Dict],
    ]:
        """Run one timestep in the environment using the agents' actions.

        When the end of an episode is reached, the user is responsible for
        calling :meth:`reset()` to reset this environments state.

        Arguments
        ---------
        actions : Dict[M.AgentID, M.ActType]
          a joint action containing one action per agent in the environment.

        Returns
        -------
        observations : Dict[M.AgentID, M.ObsType]
          the joint observation containing one observation per agent.
        rewards : Dict[M.AgentID, SupportsFloat]
          the joint rewards containing one reward per agent.
        terminated : Dict[M.AgentID, bool]
          whether each agent has reached a terminal state in the environment.
          Contains one value for each agent in the environment. It's possible,
          depending on the environment, for only some of the agents to be in a
          terminal during a given step.
        truncated : Dict[M.AgentID, bool]
          whether the episode has been truncated for each agent in the environment.
          Contains one value for each agent in the environment. Truncation for an
          agent signifies that the episode was ended for that agent (e.g. due to
          reaching the time limit) before the agent reached a terminal state.
        done : bool
          whether the episode is finished. Provided for convenience and is equivalent
          to checking if all agents are either in a terminated or truncated state. If
          true, the user needs to call :py:func:`reset()`.
        info : Dict[M.AgentID, Dict]
          contains auxiliary diagnostic information (helpful for debugging, learning,
          and logging) for each agent.

        """

    @abc.abstractmethod
    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Optional[Dict[M.AgentID, M.ObsType]], Dict[M.AgentID, Dict]]:
        """Resets the environment and returns an initial observations and info.

        Arguments
        ---------
        seed : int, optional
          The seed that is used to initialize the environment's PRNG. If the
          ``seed=None`` is passed, the PRNG will *not* be reset. If you pass an
          integer, the PRNG will be reset even if it already exists. Usually, you want
          to pass an integer *right after the environment has been initialized and
          then never again*.
        options: dict, optional
          Additional information to specify how the environment is reset (optional,
          depending on the specific environment)


        Returns
        -------
        observations : JointObservation, optional
          the joint observation containing one observation per agent in the environment.
          Note in environments that are not observation first (i.e. they expect an
          action before the first observation) this function should reset the state and
          return ``None``.
        info : Dict[M.AgentID, Dict]
          auxiliary information for each agent. It should be analogous to the ``info``
          returned by :meth:`step()`

        """

    def render(
        self,
    ) -> RenderFrame | Dict[M.AgentID, RenderFrame] | List[RenderFrame] | None:
        """Render the environment as specified by environment :attr:`render_mode`.

        The render mode attribute :attr:`render_mode` is set during the initialization
        of the environment.The environment's :attr:`metadata` render modes
        (`env.metadata["render_modes"]`) should contain the possible ways to implement
        the render modes.

        The set of supported modes varies per environment (some environments do not
        support rendering at all). By convention, if :attr:`render_mode` is:

        - None (default): no render is computed.
        - "human": Environment is rendered to the current display or terminal usually
          for human consumption. Returns ``None``.
        - "rgb_array": Return an ``np.ndarray`` with shape ``(x, y, 3)``  representing
          RGB values for an x-by-y pixel image, suitable for turning into a video.
        - "ansi": Return a string (``str``) or ``StringIO.StringIO`` containing a
          terminal-style text representation for each timestep. The text can include
          newlines and ANSI escape sequences (e.g. for colors).
        - "rgb_array_dict" and "ansi_dict": Return ``dict`` mapping ``AgentID`` to
          render frame (RGB or ANSI depending on render mode). Each render frame is
          represents the agent-centric view for the given agent.

        Note
        ----
        Make sure that your class's :attr:`metadata` ``"render_modes"`` key includes
          the list of supported modes.

        Returns
        -------
        Value :  Any
          the return value depends on the mode.

        """
        raise NotImplementedError

    def close(self) -> None:
        """Close environment and perform any necessary cleanup.

        Should be overriden in subclasses as necessary.
        """
        pass

    @property
    @abc.abstractmethod
    def model(self) -> M.POSGModel[M.StateType, M.ObsType, M.ActType]:
        """Get the model for this environment."""

    @property
    @abc.abstractmethod
    def state(self) -> M.StateType:
        """Get the current state for this environment."""

    @property
    def possible_agents(self) -> Tuple[M.AgentID, ...]:
        """Get list of agents that may appear in the environment."""
        return self.model.possible_agents

    @property
    def agents(self) -> List[M.AgentID]:
        """Get list of agents active in the environment for current state.

        This will be :attr:`possible_agents`, independent of state, for any environment
        where the number of agents remains constant during and across episodes.

        Returns
        -------
        agents: List of IDs of agents currently active in the environment.

        """
        return self.model.get_agents(self.state)

    @property
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
        return self.model.observation_first

    @property
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
        return self.model.is_symmetric

    @property
    def action_spaces(self) -> Dict[M.AgentID, spaces.Space]:
        """Get the action space for each agent."""
        return self.model.action_spaces

    @property
    def observation_spaces(self) -> Dict[M.AgentID, spaces.Space]:
        """Get the observation space for each agent."""
        return self.model.observation_spaces

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[SupportsFloat, SupportsFloat]]:
        """The minimum and maximum  possible rewards for each agent."""
        return self.model.reward_ranges

    @property
    def unwrapped(self) -> "Env":
        """Completely unwrap this env.

        Returns
        -------
        env: posggym.Env
            The base non-wrapped posggym.Env instance

        """
        return self

    def __str__(self):
        """Returns a string of the environment with ID of :attr:`spec` if :attr:`spec.

        Returns
        -------
        str
            A string identifying the environment.

        """
        if self.spec is None:
            return f"<{type(self).__name__} instance>"
        else:
            return f"<{type(self).__name__}<{self.spec.id}>>"

    def __enter__(self):
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment."""
        self.close()
        # propagate exception
        return False


class DefaultEnv(Env[M.StateType, M.ObsType, M.ActType]):
    """Default Environment implementation from environment model.

    This class implements some of the main environment functions using the
    environment model in a default manner.

    Specifically it implements the following functions and attributes:

        step
        reset
        state

    Users will need to implement:

        model

    and optionally implement:

        render
        close

    """

    def __init__(self) -> None:
        assert self.model, (
            "self.model property must be initialized before calling init "
            "function of parent class"
        )
        self._state = self.model.sample_initial_state()
        self._last_obs: Dict[M.AgentID, M.ObsType] | None = None
        if self.model.observation_first:
            self._last_obs = self.model.sample_initial_obs(self._state)
        self._step_num = 0
        self._last_actions: Dict[M.AgentID, M.ActType] | None = None
        self._last_rewards: Dict[M.AgentID, SupportsFloat] | None = None

    def step(
        self, actions: Dict[M.AgentID, M.ActType]
    ) -> Tuple[
        Dict[M.AgentID, M.ObsType],
        Dict[M.AgentID, SupportsFloat],
        Dict[M.AgentID, bool],
        Dict[M.AgentID, bool],
        bool,
        Dict[M.AgentID, Dict],
    ]:
        step = self.model.step(self._state, actions)
        self._step_num += 1
        self._state = step.state
        self._last_obs = step.observations
        self._last_actions = actions
        self._last_rewards = step.rewards

        if step.outcomes is not None:
            info = {i: {"outcome": o} for i, o in step.outcomes.items()}
        else:
            info = {}

        for i, i_info in step.info.items():
            if i not in info:
                info[i] = {}
            info[i].update(i_info)

        return (
            step.observations,
            step.rewards,
            step.terminated,
            step.truncated,
            step.all_done,
            info,
        )

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Optional[Dict[M.AgentID, M.ObsType]], Dict[M.AgentID, Dict]]:
        if seed is not None:
            self.model.set_seed(seed)

        self._state = self.model.sample_initial_state()
        if self.model.observation_first:
            self._last_obs = self.model.sample_initial_obs(self._state)
        else:
            self._last_obs = None
        self._last_actions = None
        self._last_rewards = None
        self._step_num = 0
        # TODO Handle info from model
        return self._last_obs, {}

    @property
    def state(self) -> M.StateType:
        return copy.copy(self._state)


WrapperStateType = TypeVar("WrapperStateType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


class Wrapper(Env[WrapperStateType, WrapperObsType, WrapperActType]):
    """Wraps a :class:`posggym.Env` to allow a modular transformation.

    This class is the base class for all wrappers. The subclass could override
    some methods to change the bahavior of the original environment without
    touching the original code.

    Note
    ----
    Don't forget to call ``super().__init__(env)`` if the subclass overrides
    the `__init__` method.

    """

    def __init__(self, env: Env[M.StateType, M.ObsType, M.ActType]):
        self.env = env

        self._action_spaces: Dict[M.AgentID, spaces.Space] | None = None
        self._observation_spaces: Dict[M.AgentID, spaces.Space] | None = None
        self._reward_ranges: Dict[
            M.AgentID, Tuple[SupportsFloat, SupportsFloat]
        ] | None = None
        self._metadata: Dict[str, Any] | None = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    @classmethod
    def class_name(cls):
        """Return the name of the wrapper class."""
        return cls.__name__

    @property
    def model(self) -> M.POSGModel:
        return self.env.model

    @property
    def state(self) -> WrapperStateType:
        return self.env.state  # type: ignore

    @property
    def possible_agents(self) -> Tuple[M.AgentID, ...]:
        return self.env.possible_agents

    @property
    def agents(self) -> List[M.AgentID]:
        return self.env.agents

    @property
    def action_spaces(self) -> Dict[M.AgentID, spaces.Space]:
        if self._action_spaces is None:
            return self.env.action_spaces
        return self._action_spaces

    @action_spaces.setter
    def action_spaces(self, action_spaces: Dict[M.AgentID, spaces.Space]):
        self._action_spaces = action_spaces

    @property
    def observation_spaces(self) -> Dict[M.AgentID, spaces.Space]:
        """Get the observation space for each agent."""
        if self._observation_spaces is None:
            return self.env.observation_spaces
        return self._observation_spaces

    @observation_spaces.setter
    def observation_spaces(self, observation_spaces: Dict[M.AgentID, spaces.Space]):
        self._observation_spaces = observation_spaces

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[SupportsFloat, SupportsFloat]]:
        if self._reward_ranges is None:
            return self.env.reward_ranges
        return self._reward_ranges

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get wrapper metadata."""
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = value

    @property
    def spec(self) -> EnvSpec | None:
        """Return the :attr:`Env` :attr:`spec` attribute."""
        return self.env.spec

    @spec.setter
    def spec(self, env_spec: EnvSpec):
        self.env.spec = env_spec

    @property
    def render_mode(self) -> str | None:
        """Return the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @render_mode.setter
    def render_mode(self, render_mode: str | None):
        self.env.render_mode = render_mode

    def step(
        self, actions: Dict[M.AgentID, WrapperActType]
    ) -> Tuple[
        Dict[M.AgentID, WrapperObsType],
        Dict[M.AgentID, SupportsFloat],
        Dict[M.AgentID, bool],
        Dict[M.AgentID, bool],
        bool,
        Dict[M.AgentID, Dict],
    ]:
        return self.env.step(actions)  # type: ignore

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Optional[Dict[M.AgentID, WrapperObsType]], Dict[M.AgentID, Dict]]:
        return self.env.reset(seed=seed, options=options)  # type: ignore

    def render(
        self,
    ) -> RenderFrame | Dict[M.AgentID, RenderFrame] | List[RenderFrame] | None:
        return self.env.render()

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self) -> Env:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`posggym.Env` environment, underneath all layers
        of wrappers.
        """
        return self.env.unwrapped

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)


class ObservationWrapper(Wrapper[M.StateType, WrapperObsType, M.ActType]):
    """Wraps environment to allow modular transformations of observations.

    Subclasses should at least implement the observations function.
    """

    def __init__(self, env: Env[M.StateType, M.ObsType, M.ActType]):
        super().__init__(env)

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Optional[Dict[M.AgentID, WrapperObsType]], Dict[M.AgentID, Dict]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if obs is None:
            return obs, info
        return self.observations(obs), info

    def step(
        self, actions: Dict[M.AgentID, M.ActType]
    ) -> Tuple[
        Dict[M.AgentID, WrapperObsType],
        Dict[M.AgentID, SupportsFloat],
        Dict[M.AgentID, bool],
        Dict[M.AgentID, bool],
        bool,
        Dict[M.AgentID, Dict],
    ]:
        obs, reward, term, trunc, done, info = self.env.step(actions)  # type: ignore
        return self.observations(obs), reward, term, trunc, done, info

    def observations(
        self, obs: Dict[M.AgentID, M.ObsType]
    ) -> Dict[M.AgentID, WrapperObsType]:
        """Transforms observations recieved from wrapped environment."""
        raise NotImplementedError


class RewardWrapper(Wrapper[M.StateType, M.ObsType, M.ActType]):
    """Wraps environment to allow modular transformations of rewards.

    Subclasses should atleast implement the rewards function.
    """

    def __init__(self, env: Env[M.StateType, M.ObsType, M.ActType]):
        super().__init__(env)

    def step(
        self, actions: Dict[M.AgentID, M.ActType]
    ) -> Tuple[
        Dict[M.AgentID, M.ObsType],
        Dict[M.AgentID, SupportsFloat],
        Dict[M.AgentID, bool],
        Dict[M.AgentID, bool],
        bool,
        Dict[M.AgentID, Dict],
    ]:
        obs, reward, term, trunc, done, info = self.env.step(actions)  # type: ignore
        return obs, self.rewards(reward), term, trunc, done, info  # type: ignore

    def rewards(
        self, rewards: Dict[M.AgentID, SupportsFloat]
    ) -> Dict[M.AgentID, SupportsFloat]:
        """Transforms rewards recieved from wrapped environment."""
        raise NotImplementedError


class ActionWrapper(Wrapper[M.StateType, M.ObsType, WrapperActType]):
    """Wraps environment to allow modular transformations of actions.

    Subclasses should atleast implement the actions function.
    """

    def __init__(self, env: Env[M.StateType, M.ObsType, M.ActType]):
        super().__init__(env)

    def step(
        self, actions: Dict[M.AgentID, M.ActType]
    ) -> Tuple[
        Dict[M.AgentID, M.ObsType],
        Dict[M.AgentID, SupportsFloat],
        Dict[M.AgentID, bool],
        Dict[M.AgentID, bool],
        bool,
        Dict[M.AgentID, Dict],
    ]:
        return self.env.step(self.actions(actions))  # type: ignore

    def actions(
        self, actions: Dict[M.AgentID, M.ActType]
    ) -> Dict[M.AgentID, WrapperActType]:
        """Transform actions for wrapped environment."""
        raise NotImplementedError
