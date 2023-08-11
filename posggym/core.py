"""Contains the main POSG environment class and functions.

The implementation is heavily inspired by Open AI Gym
https://github.com/openai/gym

And, the more recent, Farama Foundation Gymnasium
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/core.py

"""
from __future__ import annotations

import abc
import copy
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple, TypeVar

from posggym.model import ActType, ObsType, POSGModel, StateType

if TYPE_CHECKING:
    import numpy as np
    from gymnasium import spaces

    from posggym.envs.registration import EnvSpec


class Env(abc.ABC, Generic[StateType, ObsType, ActType]):
    r"""The main POSGGym class for implementing POSG environments.

    The class encapsulates an environment and a POSG model. The environment maintains an
    internal state and can be interacted with by multiple agents in parallel through
    the :meth:`step` and :meth:`reset` functions. The POSG model can be accessed via the
    :attr:`model` attribute and exposes the model of the environment which can be used
    for planning or for anything else (see :py:class:`posggym.POSGModel` class for
    details).

    The implementation is heavily inspired by the Farama Foundation Gymnasium
    (https://github.com/Farama-Foundation/Gymnasium) and PettingZoo
    (https://github.com/Farama-Foundation/PettingZoo) APIs. It aims to be consistent
    with these APIs and easily compatible with the PettingZoo API.

    The main API methods that users of this class need to know are:

    - :meth:`step`
    - :meth:`reset`
    - :meth:`render`
    - :meth:`close`

    And the main attributes:

    - :attr:`model` - The POSG model of the environment (:py:class:`posggym.POSGModel`)
    - :attr:`state` - The current state of the environment
    - :attr:`possible_agents` - All agents that may appear in the environment
    - :attr:`agents` - The agents currently active in the environment
    - :attr:`action_spaces` - The action space for each agent
    - :attr:`observation_spaces` - The observation space for each agent
    - :attr:`reward_ranges` - The minimum and maximum possible rewards each agent may
      receive for single step in the environment. The default reward range is set to
      :math:`(-\infty,+\infty)`.
    - :attr:`is_symmetric` - Whether the environment is symmetric or asymmetric
    - :attr:`spec` - An environment spec that contains the information used to
      initialize the environment from :meth:`posggym.make`
    - :attr:`metadata` - The metadata of the environment, i.e. render modes, render fps
    - :attr:`render_mode` - The current render mode of the environment

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

    # The model used by the environment
    model: POSGModel[StateType, ObsType, ActType]

    @abc.abstractmethod
    def step(
        self, actions: Dict[str, ActType]
    ) -> Tuple[
        Dict[str, ObsType],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        bool,
        Dict[str, Dict[str, Any]],
    ]:
        """Run one timestep in the environment using the agents' actions.

        When the end of an episode is reached, the user is responsible for
        calling :meth:`reset()` to reset this environments state.

        Arguments
        ---------
        actions : Dict[str, ActType]
          a joint action containing one action per active agent in the environment.

        Returns
        -------
        observations : Dict[str, ObsType]
          the joint observation containing one observation per agent.
        rewards : Dict[str, float]
          the joint rewards containing one reward per agent.
        terminations : Dict[str, bool]
          whether each agent has reached a terminal state in the environment.
          Contains one value for each agent in the environment. It's possible,
          depending on the environment, for only some of the agents to be in a
          terminal during a given step.
        truncations : Dict[str, bool]
          whether the episode has been truncated for each agent in the
          environment. Contains one value for each agent in the environment. Truncation
          for an agent signifies that the episode was ended for that agent (e.g. due to
          reaching the time limit) before the agent reached a terminal state.
        all_done : bool
          whether the episode is finished. Provided for convenience and to handle
          the case where agents may be added and removed during an episode. For
          environments where the active agents remains constant during each episode,
          this is equivalent to checking if all agents are either in a terminated or
          truncated state. If true, the user needs to call :py:func:`reset()`.
        infos : Dict[str, Dict[str, Any]]
          contains auxiliary diagnostic information (helpful for debugging, learning,
          and logging) for each agent.

        """

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Dict[str, ObsType], Dict[str, Dict]]:
        """Resets the environment and returns an initial observations and info.

        This method generates a new starting state often with some randomness. This
        randomness can be controlled with the ``seed`` parameter. If the
        environment already has a random number generator (RNG) and :meth:`reset` is
        called with ``seed=None``, the RNG is not reset. Note, that the RNG is handled
        by the environment :attr:`model`, rather than the environment class itself.

        Therefore, :meth:`reset` should (in the typical use case) be called with a seed
        right after initialization and then never again.

        For Custom environments, the first line of :meth:`reset` should be
        ``super().reset(seed=seed)`` which implements the seeding correctly.

        Arguments
        ---------
        seed : int,  optional
          The seed that is used to initialize the environment's RNG. If the
          ``seed=None`` is passed, the RNG will *not* be reset. If you pass an
          integer, the RNG will be reset even if it already exists. Usually, you want
          to pass an integer *right after the environment has been initialized and
          then never again*.
        options: dict, optional
          Additional information to specify how the environment is reset (optional,
          depending on the specific environment)

        Returns
        -------
        observations : Dict[str, ObsType], optional
          The joint observation containing one observation per agent in the environment.
        infos : Dict[str, Dict]
          Auxiliary information for each agent. It should be analogous to the ``info``
          returned by :meth:`step()` and can be empty.

        """
        # initialize the RNG if the seed is manually passed
        if seed is not None:
            self.model.seed(seed)
        return {}, {}

    def render(
        self,
    ) -> None | np.ndarray | str | Dict[str, np.ndarray] | Dict[str, str]:
        """Render the environment as specified by environment :attr:`render_mode`.

        The render mode attribute :attr:`render_mode` is set during the initialization
        of the environment. While the environment's :attr:`metadata` render modes
        (`env.metadata["render_modes"]`) should contain the supported render modes.

        The set of supported modes varies per environment (some environments do not
        support rendering at all). By convention, if :attr:`render_mode` is:

        - ``None`` (default): no render is computed.
        - ``"human"``: Environment is rendered to the current display or terminal
          usually for human consumption. Returns ``None``.
        - ``"rgb_array"``: Return an ``np.ndarray`` with shape ``(x, y, 3)``
          representing RGB values for an x-by-y pixel image of the entire environment,
          suitable for turning into a video.
        - ``"ansi"``: Return a string (``str``) or ``StringIO.StringIO`` containing a
          terminal-style text representation for each timestep. The text can include
          newlines and ANSI escape sequences (e.g. for colors).
        - ``"rgb_array_dict"`` and ``"ansi_dict"``: Return ``dict`` mapping agent ID
          to render frame (RGB or ANSI depending on render mode). Each render frame is
          represents the agent-centric view for the given agent. May also return a
          render for the entire environment (like `"rgb_array"` and `"ansi"` render
          modes) which should be mapped to the `"env"` key in the dictionary by default.

        Note
        ----
        Make sure that your class's :attr:`metadata` ``"render_modes"`` key includes
        the list of supported modes.

        """
        raise NotImplementedError

    def close(self):
        """Close environment and perform any necessary cleanup.

        Should be overridden in subclasses as necessary.
        """
        pass

    @property
    @abc.abstractmethod
    def state(self) -> StateType:
        """The current state for this environment.

        This must be implemented in custom environments.

        Returns
        -------
        StateType

        """

    @property
    def possible_agents(self) -> Tuple[str, ...]:
        """The list of all possible agents that may appear in the environment.

        Returns
        -------
        Tuple[str, ...]

        """
        return self.model.possible_agents

    @property
    def agents(self) -> List[str]:
        """The list of agents active in the environment for current state.

        This will be :attr:`possible_agents`, independent of state, for any environment
        where the number of agents remains constant during and across episodes.

        Returns
        -------
        List[str]

        """
        return self.model.get_agents(self.state)

    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        """A mapping from Agent ID to the space of valid actions for that agent.

        Returns
        -------
        Dict[str, spaces.Space]

        """
        return self.model.action_spaces

    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        """A mapping from Agent ID to the space of valid observations for that agent.

        Returns
        -------
        Dict[str, spaces.Space]

        """
        return self.model.observation_spaces

    @property
    def reward_ranges(self) -> Dict[str, Tuple[float, float]]:
        r"""A mapping from Agent ID to the min and max possible rewards for that agent.

        Each reward tuple corresponding to the minimum and maximum possible rewards for
        a given agent for a single step. The default reward range for each agent is set
        to :math:`(-\infty,+\infty)`.

        Returns
        -------
        Dict[str, Tuple[float, float]]

        """
        return self.model.reward_ranges

    @property
    def is_symmetric(self) -> bool:
        """Whether the environment is symmetric.

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
    def unwrapped(self) -> "Env":
        """Completely unwrap this env.

        Returns
        -------
        posggym.Env
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


class DefaultEnv(Env[StateType, ObsType, ActType]):
    """A default environment implementation using an environment model.

    This class implements the main environment methods - :meth:`reset`,
    :meth:`step`, :attr:`state` - using the environment model.

    Users need only initialize the class with a :class:`posggym.POSGModel` instance by
    calling ``super().__init__(custom_model)`` in their custom environment class that
    inherits from this class.

    The custom environment needs only (optionally) implement rendering and clean-up
    methods:

    - :meth:`render`
    - :meth:`close`

    """

    def __init__(self, model: POSGModel, render_mode: Optional[str] = None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.model = model
        self.render_mode = render_mode

        self._state = self.model.sample_initial_state()
        self._last_obs = self.model.sample_initial_obs(self._state)
        self._step_num = 0
        self._last_actions: Dict[str, ActType] | None = None
        self._last_rewards: Dict[str, float] | None = None

    def step(
        self, actions: Dict[str, ActType]
    ) -> Tuple[
        Dict[str, ObsType],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        bool,
        Dict[str, Dict],
    ]:
        step = self.model.step(self._state, actions)
        self._step_num += 1
        self._state = step.state
        self._last_obs = step.observations
        self._last_actions = actions
        self._last_rewards = step.rewards
        return (
            step.observations,
            step.rewards,
            step.terminations,
            step.truncations,
            step.all_done,
            step.infos,
        )

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Dict[str, ObsType], Dict[str, Dict]]:
        super().reset(seed=seed)
        self._state = self.model.sample_initial_state()
        self._last_obs = self.model.sample_initial_obs(self._state)
        self._last_actions = None
        self._last_rewards = None
        self._step_num = 0
        return self._last_obs, {i: {} for i in self._last_obs}

    @property
    def state(self) -> StateType:
        return copy.copy(self._state)


WrapperStateType = TypeVar("WrapperStateType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


class Wrapper(Env[WrapperStateType, WrapperObsType, WrapperActType]):
    """Wraps a :class:`posggym.Env` to allow a modular transformation.

    This class is the base class for all wrappers. Wrappers that inherit from this class
    can modify :attr:`action_spaces`, :attr:`observation_spaces`, :attr:`reward_ranges`
    and :attr:`metadata` attributes , without changing the underlying environment's
    attributes.

    Moreover, the behavior of the :meth:`step` and :meth:`reset` methods can be changed
    by these wrappers. Some attributes (:attr:`spec`, :attr:`render_mode`) will point
    back to the wrapper's environment (i.e. to the corresponding attributes of
    :attr:`env`).

    Note
    ----
    If you inherit from :class:`Wrapper`, don't forget to call ``super().__init__(env)``
    if the subclass overrides the `__init__` method.

    """

    def __init__(self, env: Env[StateType, ObsType, ActType]):
        self.env = env
        self._action_spaces: Dict[str, spaces.Space] | None = None
        self._observation_spaces: Dict[str, spaces.Space] | None = None
        self._reward_ranges: Dict[str, Tuple[float, float]] | None = None
        self._metadata: Dict[str, Any] | None = None

    def __getattr__(self, name):
        """Returns attribute with ``name``, unless ``name`` starts with underscore."""
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    @classmethod
    def class_name(cls):
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def model(self) -> POSGModel:
        """Returns the :attr:`Env` :attr:`model`."""
        return self.env.model

    @model.setter
    def model(self, value: POSGModel):
        self.env.model = value

    @property
    def state(self) -> WrapperStateType:
        """Returns the :attr:`Env` :attr:`state`."""
        return self.env.state  # type: ignore

    @property
    def possible_agents(self) -> Tuple[str, ...]:
        """Returns the :attr:`Env` :attr:`possible_agents`."""
        return self.env.possible_agents

    @property
    def agents(self) -> List[str]:
        """Returns the :attr:`Env` :attr:`agents`."""
        return self.env.agents

    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        """Return the :attr:`Env` :attr:`action_spaces`.

        This is the :attr:`Env` :attr:`action_spaces` unless it's overwritten then the
        wrapper :attr:`action_spaces` is used.
        """
        if self._action_spaces is None:
            return self.env.action_spaces
        return self._action_spaces

    @action_spaces.setter
    def action_spaces(self, action_spaces: Dict[str, spaces.Space]):
        self._action_spaces = action_spaces

    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        """Return the :attr:`Env` :attr:`observation_spaces`.

        This is the :attr:`Env` :attr:`observation_spaces` unless it's overwritten then
        the wrapper :attr:`observation_spaces` is used.
        """
        if self._observation_spaces is None:
            return self.env.observation_spaces
        return self._observation_spaces

    @observation_spaces.setter
    def observation_spaces(self, observation_spaces: Dict[str, spaces.Space]):
        self._observation_spaces = observation_spaces

    @property
    def reward_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return the :attr:`Env` :attr:`reward_ranges`.

        This is the :attr:`Env` :attr:`reward_ranges`, unless it's overwritten, then
        the wrapper :attr:`reward_ranges` is used.
        """
        if self._reward_ranges is None:
            return self.env.reward_ranges
        return self._reward_ranges

    @reward_ranges.setter
    def reward_ranges(self, reward_ranges: Dict[str, Tuple[float, float]]):
        self._reward_ranges = reward_ranges

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns the :attr:`Env` :attr:`metadata`."""
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
        self, actions: Dict[str, WrapperActType]
    ) -> Tuple[
        Dict[str, WrapperObsType],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        bool,
        Dict[str, Dict],
    ]:
        """Uses the :meth:`step` of the :attr:`env`.

        Can be overwritten to change the returned data.
        """
        return self.env.step(actions)  # type: ignore

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Dict[str, WrapperObsType], Dict[str, Dict]]:
        """Uses the :meth:`reset` of the :attr:`env`.

        Can be overwritten to change the returned data.
        """
        return self.env.reset(seed=seed, options=options)  # type: ignore

    def render(
        self,
    ) -> None | np.ndarray | str | Dict[str, np.ndarray] | Dict[str, str]:
        """Uses the :meth:`render` of the :attr:`env`.

        Can be overwritten to change the returned data.
        """
        return self.env.render()

    def close(self):
        """Closes the wrapper and :attr:`env`."""
        return self.env.close()

    @property
    def unwrapped(self) -> Env:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`posggym.Env` environment, underneath all layers
        of wrappers.
        """
        return self.env.unwrapped

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)


class ObservationWrapper(Wrapper[StateType, WrapperObsType, ActType]):
    """Wraps environment to allow modular transformations of observations.

    Subclasses should at least implement the observations function.
    """

    def __init__(self, env: Env[StateType, ObsType, ActType]):
        super().__init__(env)

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Dict[str, WrapperObsType], Dict[str, Dict]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if obs is None:
            return obs, info
        return self.observations(obs), info

    def step(
        self, actions: Dict[str, ActType]
    ) -> Tuple[
        Dict[str, WrapperObsType],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        bool,
        Dict[str, Dict],
    ]:
        obs, reward, term, trunc, done, infos = self.env.step(actions)  # type: ignore
        return self.observations(obs), reward, term, trunc, done, infos

    def observations(self, obs: Dict[str, ObsType]) -> Dict[str, WrapperObsType]:
        """Transforms observations received from wrapped environment."""
        raise NotImplementedError


class RewardWrapper(Wrapper[StateType, ObsType, ActType]):
    """Wraps environment to allow modular transformations of rewards.

    Subclasses should at least implement the rewards function.
    """

    def __init__(self, env: Env[StateType, ObsType, ActType]):
        super().__init__(env)

    def step(
        self, actions: Dict[str, ActType]
    ) -> Tuple[
        Dict[str, ObsType],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        bool,
        Dict[str, Dict],
    ]:
        obs, reward, term, trunc, done, info = self.env.step(actions)  # type: ignore
        return obs, self.rewards(reward), term, trunc, done, info  # type: ignore

    def rewards(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Transforms rewards received from wrapped environment."""
        raise NotImplementedError


class ActionWrapper(Wrapper[StateType, ObsType, WrapperActType]):
    """Wraps environment to allow modular transformations of actions.

    Subclasses should at least implement the actions function.
    """

    def __init__(self, env: Env[StateType, ObsType, ActType]):
        super().__init__(env)

    def step(
        self, actions: Dict[str, ActType]
    ) -> Tuple[
        Dict[str, ObsType],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        bool,
        Dict[str, Dict],
    ]:
        return self.env.step(self.actions(actions))  # type: ignore

    def actions(self, actions: Dict[str, ActType]) -> Dict[str, WrapperActType]:
        """Transform actions for wrapped environment."""
        raise NotImplementedError
