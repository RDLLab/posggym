"""Contains the main POSG environment class and functions """
import abc
from typing import Tuple, Optional, Dict

from gym import spaces

import posggym.model as M

# TODO Handle seeds properly


class Env(abc.ABC):
    """The main POSG environment class.

    It encapsulates an environment and POSG model.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    And the main attributes:

        n_agents: the number of agents in the environment
        model: the POSG model of the environment (posggym.model.POSGModel)
        action_specs : the action space specs for each agent
        observation_spacs : the observation space specs for each agent
        reward_specs : the reward specs for each agent
    """

    # Set this in SOME subclasses
    metadata: Dict = {"render.modes": []}

    @abc.abstractmethod
    def step(self,
             actions: Tuple[M.Action, ...]
             ) -> Tuple[M.JointAction, M.JointReward, bool, dict]:
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
    def reset(self, seed: Optional[int] = None) -> M.JointObservation:
        """Resets the environment returns the initial observations

        Arguments
        ---------
        seed : int, optional
            RNG seed (default=None)

        Returns
        -------
        observations : object
            the joint observation containing one observation per agent in the
            environment
        """

    @abc.abstractmethod
    def render(self, mode: str = "human") -> None:
        """Renders the environment.

        The set of supported modes varies per environment.

        Arguments
        ---------
        mode : str, optional
            the mode to render with (default='human')
        """

    def close(self) -> None:
        """Close environment and perform any necessary cleanup.

        Should be overriden in subclasses as necessary.
        """

    @property
    @abc.abstractmethod
    def model(self) -> M.POSGModel:
        """The model for this environment. """

    @property
    def n_agents(self) -> int:
        """The number of agents in this environment """
        return self.model.n_agents

    @property
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        """The action space for each agent """
        return self.model.action_spaces

    @property
    def obs_spaces(self) -> Tuple[spaces.Space, ...]:
        """The observation space for each agent """
        return self.model.obs_spaces

    @property
    def reward_ranges(self) -> Tuple[Tuple[M.Reward, M.Reward], ...]:
        """The minimum and maximum  possible rewards for each agent """
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
