"""Contains the main POSG environment class and functions.

The implementation is heavily inspired by the Open AI Gym
https://github.com/openai/gym

"""
import abc
import copy
from typing import Tuple, Optional, Dict, TYPE_CHECKING

from gym import spaces

import posggym.model as M

if TYPE_CHECKING:
    from posggym.envs.registration import EnvSpec


class Env(abc.ABC):
    """The main POSG environment class.

    The implementation is heavily inspired by the Open AI Gym API and
    implementation: https://github.com/openai/gym

    It encapsulates an environment and POSG model.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        unwrapped

    And the main attributes:

        n_agents : the number of agents in the environment
        model : the POSG model of the environment (posggym.model.POSGModel)
        state : the current state of the environment
        observation_first : whether environment is observation or action first
        action_spaces : the action space specs for each agent
        observation_spaces : the observation space specs for each agent
        reward_specs : the reward specs for each agent

    """

    # Set this in SOME subclasses
    metadata: Dict = {"render.modes": []}

    # EnvSpec used to instantiate env instance
    # This is set when env is made using posggym.make function
    spec: "EnvSpec" = None

    @abc.abstractmethod
    def step(self,
             actions: Tuple[M.Action, ...]
             ) -> Tuple[M.JointObservation, M.JointReward, bool, Dict]:
        """Run one timestep in the environment.

        When the end of an episode is reached, the user is responsible for
        calling `reset()` to reset this environments state.

        Accepts a joint action and returns a tuple (observations, rewards,
        done, info)

        Arguments
        ---------
        actions : object
            a joint action containing one action per agent in the environment.

        Returns
        -------
        observations : object
            the joint observation containing one observation per agent in the
            environment.
        rewards : object
            the joint rewards containing one reward per agent in the
            environment.
        done : bool
            whether the episode has ended, in which case further step() calls
            will return undefined results
        info : dict
            contains auxiliary diagnostic information (helpful for debugging)

        """

    @abc.abstractmethod
    def reset(self,
              *,
              seed: Optional[int] = None) -> Optional[M.JointObservation]:
        """Reset the environment returns the initial observations.

        Arguments
        ---------
        seed : int, optional
            The seed that is used to initialize the environment's PRNG. If the
            ``seed=None`` is passed, the PRNG will *not* be reset. If you pass
            an integer, the PRNG will be reset even if it already exists.
            Usually, you want to pass an integer *right after the environment
            has been initialized and then never again*.

        Returns
        -------
        observations : object
            the joint observation containing one observation per agent in the
            environment. Note in environments that are not observation first
            (i.e. they expect an action before the first observation) this
            function should reset the state and return None.

        """

    def render(self, mode: str = "human"):
        """Render the environment.

        This function is based on conventions of gym.core.Env class and the
        documentation from the original function is reproduced here for
        convinience.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note
        ----
        Make sure that your class's metadata 'render.modes' key includes
          the list of supported modes. It's recommended to call super()
          in implementations to use the functionality of this method.

        Arguments
        ---------
        mode : str, optional
            the mode to render with (default='human')

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

    @property
    @abc.abstractmethod
    def model(self) -> M.POSGModel:
        """Get the model for this environment."""

    @property
    @abc.abstractmethod
    def state(self) -> M.State:
        """Get the current state for this environment."""

    @property
    def n_agents(self) -> int:
        """Get the number of agents in this environment."""
        return self.model.n_agents

    @property
    def observation_first(self) -> bool:
        """Get whether environment is observation or action first."""
        return self.model.observation_first

    @property
    def is_symmetric(self) -> bool:
        """Get whether environment is symmetric."""
        return self.model.is_symmetric

    @property
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        """Get the action space for each agent."""
        return self.model.action_spaces

    @property
    def observation_spaces(self) -> Tuple[spaces.Space, ...]:
        """Get the observation space for each agent."""
        return self.model.observation_spaces

    @property
    def reward_ranges(self) -> Tuple[Tuple[M.Reward, M.Reward], ...]:
        """The minimum and maximum  possible rewards for each agent."""
        return self.model.reward_ranges

    @property
    def unwrapped(self) -> 'Env':
        """Completely unwrap this env.

        Returns
        -------
        env: posggym.Env
            The base non-wrapped posggym.Env instance

        """
        return self

    def __str__(self):
        return f"<{type(self).__name__} instance>"

    def __enter__(self):
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment."""
        self.close()
        # propagate exception
        return False


class DefaultEnv(Env):
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

    def __init__(self):
        assert self.model, (
            "self.model property must be initialized before calling init "
            "function of parent class"
        )
        self._state = self.model.sample_initial_state()
        if self.model.observation_first:
            self._last_obs = self.model.sample_initial_obs(self._state)
        else:
            self._last_obs = None
        self._step_num = 0
        self._last_actions: Optional[M.JointAction] = None
        self._last_rewards: Optional[M.JointReward] = None

    def step(self,
             actions: M.JointAction
             ) -> Tuple[M.JointObservation, M.JointReward, bool, dict]:
        step = self.model.step(self._state, actions)
        self._step_num += 1
        self._state = step.state
        self._last_obs = step.observations
        self._last_actions = actions
        self._last_rewards = step.rewards
        aux = {
            "dones": step.dones,
            "outcome": step.outcomes
        }
        return (step.observations, step.rewards, step.all_done, aux)

    def reset(self,
              *,
              seed: Optional[int] = None) -> Optional[M.JointObservation]:
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
        return self._last_obs

    @property
    def state(self) -> M.State:
        return copy.copy(self._state)


class Wrapper(Env):
    """Wraps the environment to allow a modular transformation.

    This class is the base class for all wrappers. The subclass could override
    some methods to change the bahavior of the original environment without
    touching the original code.

    Don't forget to call ``super().__init__(env)`` if the subclass overrides
    the `__init__` method.
    """

    def __init__(self, env: Env):
        self.env = env

        self._action_spaces: Optional[Tuple[spaces.Space, ...]] = None
        self._observation_spaces: Optional[Tuple[spaces.Space, ...]] = None
        self._reward_ranges: Optional[
            Tuple[Tuple[M.Reward, M.Reward], ...]
        ] = None
        self._metadata: Optional[Dict] = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                f"attempted to get missing private attribute '{name}'"
            )
        return getattr(self.env, name)

    @classmethod
    def class_name(cls):
        """Get the name of the wrapper class."""
        return cls.__name__

    @property
    def model(self) -> M.POSGModel:
        return self.env.model

    @property
    def state(self) -> M.State:
        return self.env.state

    @property
    def n_agents(self) -> int:
        return self.env.n_agents

    @property
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        if self._action_spaces is None:
            return self.env.action_spaces
        return self._action_spaces

    @action_spaces.setter
    def action_spaces(self, action_spaces: Tuple[spaces.Space, ...]):
        self._action_spaces = action_spaces

    @property
    def observation_spaces(self) -> Tuple[spaces.Space, ...]:
        if self._observation_spaces is None:
            return self.env.observation_spaces
        return self._observation_spaces

    @observation_spaces.setter
    def observation_spaces(self, observation_spaces: Tuple[spaces.Space, ...]):
        self._observation_spaces = observation_spaces

    @property
    def reward_ranges(self) -> Tuple[Tuple[M.Reward, M.Reward], ...]:
        if self._reward_ranges is None:
            return self.env.reward_ranges
        return self._reward_ranges

    @property                      # type: ignore
    def metadata(self) -> Dict:    # type: ignore
        """Get wrapper metadata."""
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def spec(self):
        """Returns the environment specification."""
        return self.env.spec

    def step(self,
             actions: M.JointAction
             ) -> Tuple[M.JointObservation, M.JointReward, bool, Dict]:
        return self.env.step(actions)

    # pylint: disable=[arguments-differ]
    def reset(self, **kwargs) -> Optional[M.JointObservation]:   # type: ignore
        return self.env.reset(**kwargs)

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)


class ObservationWrapper(Wrapper):
    """Wraps environment to allow modular transformations of observations.

    Subclasses should at least implement the observations function.
    """

    def reset(self, **kwargs):
        observations = self.env.reset(**kwargs)
        return self.observations(observations)

    def step(self, actions):
        observations, reward, done, info = self.env.step(actions)
        return self.observations(observations), reward, done, info

    @abc.abstractmethod
    def observations(self, observations):
        """Transforms observations recieved from wrapped environment."""
        raise NotImplementedError


class RewardWrapper(Wrapper):
    """Wraps environment to allow modular transformations of rewards.

    Subclasses should atleast implement the rewards function.
    """

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, actions):
        observations, reward, done, info = self.env.step(actions)
        return observations, self.rewards(reward), done, info

    @abc.abstractmethod
    def rewards(self, rewards):
        """Transforms rewards recieved from wrapped environment."""
        raise NotImplementedError


class ActionWrapper(Wrapper):
    """Wraps environment to allow modular transformations of actions.

    Subclasses should atleast implement the actions and reverse_actions
    functions.
    """

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, actions):
        return self.env.step(self.actions(actions))

    @abc.abstractmethod
    def actions(self, action):
        """Transform actions for wrapped environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def reverse_actions(self, actions):
        """Revers transformation of actions."""
        raise NotImplementedError
