"""Wrapper for rescaling observations to within min and max values."""
from typing import Dict, Union

import numpy as np
from gymnasium import spaces

from posggym import Env, ObservationWrapper


class RescaleObservations(ObservationWrapper):
    """Observation wrapper that rescales observations to the range [min_obs, max_obs].

    The base environment :attr:`env` must have an observation space of type
    :class:`spaces.Box` for each agent. If :attr:`min_obs` or :attr:`max_obs` are numpy
    arrays, the shape must match the shape of the environment's observation space for
    the given agent. If :attr:`min_obs` or :attr:`max_obs` are dictionaries then they
    must have an entry for each possible agent ID in the wrapped environment.

    Arguments
    ---------
    env : posggym.Env
        The environment to apply the wrapper
    min_obs : float, int, np.ndarray, Dict[str, Union[float, int, np.ndarray]]
        The minimum value for the scaled observations.
    max_obs : float, int, np.ndarray, Dict[str, Union[float, int, np.ndarray]]
        The maximum value for the scaled observations.


    Note
    ----
    Explanation of how to scale number from one interval into new interval:
    https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range

    """

    def __init__(
        self,
        env: Env,
        min_obs: Union[
            float, int, np.ndarray, Dict[str, Union[float, int, np.ndarray]]
        ],
        max_obs: Union[
            float, int, np.ndarray, Dict[str, Union[float, int, np.ndarray]]
        ],
    ):
        self.min_obs = {}
        self.max_obs = {}
        self.rescale_factor = {}
        for i, obs_space in env.observation_spaces.items():
            assert isinstance(obs_space, spaces.Box), (
                f"Expected Box observation space for all agents, got {type(obs_space)} "
                f"for agent '{i}'."
            )

            min_obs_i = min_obs[i] if isinstance(min_obs, dict) else min_obs
            if isinstance(min_obs_i, (float, int)):
                self.min_obs[i] = np.full_like(obs_space.low, min_obs_i)
            else:
                assert isinstance(min_obs_i, np.ndarray), min_obs_i
                assert min_obs_i.shape == obs_space.shape, (
                    min_obs_i.shape,
                    obs_space.shape,
                )
                self.min_obs[i] = min_obs_i

            max_obs_i = max_obs[i] if isinstance(max_obs, dict) else max_obs
            if isinstance(max_obs_i, (float, int)):
                self.max_obs[i] = np.full_like(obs_space.high, max_obs_i)
            else:
                assert isinstance(max_obs_i, np.ndarray), max_obs_i
                assert max_obs_i.shape == obs_space.shape, (
                    max_obs_i.shape,
                    obs_space.shape,
                )
                self.max_obs[i] = max_obs_i

            assert np.less_equal(self.min_obs[i], self.max_obs[i]).all(), (
                self.min_obs[i],
                self.max_obs[i],
            )
            self.rescale_factor[i] = (self.max_obs[i] - self.min_obs[i]) / (
                obs_space.high - obs_space.low
            )

        super().__init__(env)
        self._observation_spaces = {
            i: spaces.Box(low=self.min_obs[i], high=self.max_obs[i])
            for i in env.observation_spaces
        }

    def observations(self, obs):
        rescaled_obs = {}
        for i, obs_i in obs.items():
            low = self.env.observation_spaces[i].low
            rescaled_obs[i] = (obs_i - low) * self.rescale_factor[i] + self.min_obs[i]
        return rescaled_obs
