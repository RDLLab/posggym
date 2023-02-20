"""Wrapper to flatten observations into 1D numpy arrays."""
from gymnasium import spaces

from posggym import Env, ObservationWrapper


class FlattenObservation(ObservationWrapper):
    """Observation wrapper that flattens the observation.

    Ref:
    https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/wrappers/flatten_observation.py

    """

    def __init__(self, env: Env):
        super().__init__(env)
        self._observation_spaces = {
            i: spaces.flatten_space(obs_space)
            for i, obs_space in env.observation_spaces.items()
        }

    def observations(self, obs):
        return {
            i: spaces.flatten(self.env.observation_spaces[i], obs_i)
            for i, obs_i in obs.items()
        }
