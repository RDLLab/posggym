"""Wrapper to discretize continuous actions."""
from typing import Dict, Sequence, Union, cast

import numpy as np
from gymnasium import spaces

from posggym import ActionWrapper, Env


class DiscretizeActions(ActionWrapper):
    """Action wrapper that discretizes continuous action space.

    The base environment :attr:`env` must have an action space of type
    :class:`spaces.Box` for each agent. If :attr:`flatten` is True and original action
    space is multi-dimensional with :attr:`ndim` dimensions, then will create
    discretized space with :attr:`num_actions ** ndim` actions.

    Arguments
    ---------
    env : posggym.Env
        The environment to apply the wrapper
    num_actions : int
        The number of actions to discretize space into. For multi-dimensional
        continuous spaces each dimension will be discretized into this many actions.
    flatten : bool
       Whether to flatten action space into one-dimensional discrete space, or keep
       number of dimensions of the original action space.

    """

    def __init__(self, env: Env, num_actions: int, flatten: bool = False):
        super().__init__(env)
        assert all(
            isinstance(act_space, spaces.Box)
            for act_space in self.action_spaces.values()
        )

        self.num_actions = num_actions
        self.flatten = flatten

        box_action_spaces = cast(Dict[str, spaces.Box], self.action_spaces)
        self._unflat_space: Dict[str, spaces.MultiDiscrete] = {}
        if self.flatten:
            self._unflat_space = {
                i: self.discretize_action_space(  # type: ignore
                    act_space, num_actions=num_actions, flatten=False
                )
                for i, act_space in box_action_spaces.items()
            }

        self._action_spaces = {
            i: self.discretize_action_space(
                act_space, num_actions=num_actions, flatten=flatten
            )
            for i, act_space in box_action_spaces.items()
        }

    def discretize_action_space(
        self, action_space: spaces.Box, num_actions: int, flatten: bool = False
    ) -> Union[spaces.MultiDiscrete, spaces.Discrete]:
        assert isinstance(action_space, spaces.Box), "Action space must be a Box"
        assert (
            len(action_space.shape) > 0
        ), "Action space must be at least 1-dimensional"
        assert (
            action_space.is_bounded()
        ), "Action space must be bounded to discretize it!"

        num_dims = action_space.shape[0]
        md_space = spaces.MultiDiscrete([num_actions] * num_dims)
        if flatten:
            # Compute the product of the individual action space sizes
            n = np.prod(md_space.nvec)
            return spaces.Discrete(n)
        return md_space

    def actions(self, actions):
        if self.flatten:
            # first unflatten
            actions = {
                i: self.multidiscretize_discrete_action(act_i, self._unflat_space[i])
                for i, act_i in actions.items()
            }
        return {
            i: self.undiscretize_action(act_i, self.model.action_spaces[i])
            for i, act_i in actions.items()
        }

    def undiscretize_action(
        self,
        discrete_action: Union[int, Sequence[int], np.ndarray],
        action_space: spaces.Box,
    ) -> np.ndarray:
        if isinstance(discrete_action, int):
            discrete_action = [discrete_action]
        assert (
            len(discrete_action) == action_space.shape[0]
        ), "Discrete action must have the same number of dimensions as the action space"
        assert all(0 <= a < self.num_actions for a in discrete_action), (
            f"Discrete action must be between 0 and `num_actions`={self.num_actions}. "
            f"{discrete_action} invalid."
        )
        # Compute the bin size for each dimension
        bin_sizes = (action_space.high - action_space.low) / (self.num_actions - 1)
        # Convert each discrete action to a continuous value
        continuous_action = (
            action_space.low
            + np.array(discrete_action, dtype=action_space.dtype) * bin_sizes
        )
        return continuous_action

    def multidiscretize_discrete_action(
        self, discrete_action: int, multidiscrete_space: spaces.MultiDiscrete
    ):
        # Compute the individual action space sizes
        nvec = multidiscrete_space.nvec
        # Create an empty array to store the multidiscrete action
        action = np.zeros_like(nvec)
        # Compute the Cartesian product index for each action space
        for i in range(len(nvec)):
            prod = int(np.prod(nvec[i + 1 :]))
            action[i] = int(discrete_action // prod) % nvec[i]
        return action
