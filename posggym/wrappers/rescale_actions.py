"""Wrapper to rescale continuous actions from [min, max] range."""
from typing import Dict, Union

import numpy as np
from gymnasium import spaces

import posggym


class RescaleActions(posggym.ActionWrapper):
    """Action wrapper that rescales continuous action space.

    Rescales actions from [min, max] range to the action space range of the environment.

    The base environment :attr:`env` must have an action space of type
    :class:`spaces.Box` for each agent. If :attr:`min_action` or :attr:`max_action` are
    numpy arrays, the shape must match the shape of the environment's action space for
    the given agent. If :attr:`min_action` or :attr:`max_action` are dictionaries then
    they must have an entry for each possible agent ID in the wrapped environment.

    Arguments
    ---------
    env : posggym.Env
        The environment to apply the wrapper
    min_action : float, int, np.ndarray, Dict[str, Union[float, int, np.ndarray]]
        The minimum value for the scaled actions.
    max_action : float, int, np.ndarray, Dict[str, Union[float, int, np.ndarray]]
        The maximum value for the scaled actions.

    Note
    ----
    Explanation of how to scale number from one interval into new interval:
    https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range

    Implementation also based on similar wrapper from gymnasium:
    https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/wrappers/rescale_action.py

    """

    def __init__(
        self,
        env: posggym.Env,
        min_action: Union[
            float, int, np.ndarray, Dict[str, Union[float, int, np.ndarray]]
        ],
        max_action: Union[
            float, int, np.ndarray, Dict[str, Union[float, int, np.ndarray]]
        ],
    ):
        self.min_action = {}
        self.max_action = {}
        self.rescale_factor = {}
        for i, action_space in env.action_spaces.items():
            assert isinstance(action_space, spaces.Box), (
                f"Expected Box action space for all agents, got {type(action_space)} "
                f"for agent '{i}'."
            )

            min_action_i = min_action[i] if isinstance(min_action, dict) else min_action
            if isinstance(min_action_i, (float, int)):
                self.min_action[i] = np.full_like(action_space.low, min_action_i)
            else:
                assert isinstance(min_action_i, np.ndarray), min_action_i
                assert min_action_i.shape == action_space.shape, (
                    min_action_i.shape,
                    action_space.shape,
                )
                self.min_action[i] = min_action_i

            max_action_i = max_action[i] if isinstance(max_action, dict) else max_action
            if isinstance(max_action_i, (float, int)):
                self.max_action[i] = np.full_like(action_space.high, max_action_i)
            else:
                assert isinstance(max_action_i, np.ndarray), max_action_i
                assert max_action_i.shape == action_space.shape, (
                    max_action_i.shape,
                    action_space.shape,
                )
                self.max_action[i] = max_action_i

            assert np.less_equal(self.min_action[i], self.max_action[i]).all(), (
                self.min_action[i],
                self.max_action[i],
            )
            self.rescale_factor[i] = (action_space.high - action_space.low) / (
                self.max_action[i] - self.min_action[i]
            )

        super().__init__(env)
        self._action_spaces = {
            i: spaces.Box(low=self.min_action[i], high=self.max_action[i])
            for i in env.action_spaces
        }

    def actions(self, actions):
        # rescales from [min_action, max_action] to the action space of wrapped env.
        rescaled_actions = {}
        for i, act_i in actions.items():
            low = self.env.action_spaces[i].low
            high = self.env.action_spaces[i].high
            rescaled_action = (act_i - self.min_action[i]) * self.rescale_factor[
                i
            ] + low
            rescaled_actions[i] = np.clip(rescaled_action, low, high)
        return rescaled_actions
