"""Wrapper to flatten observations into 1D numpy arrays."""
from gymnasium import spaces

from posggym import Env, ActionWrapper
import numpy as np
from typing import Union, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from posggym.model import AgentID


class DiscretrizeActions(ActionWrapper):
    """Observation wrapper that flattens the observation."""

    def __init__(self, env: Env, num_actions=5, flatten=False):
        super().__init__(env)

        self.num_actions = num_actions
        self.flatten = flatten
        assert all(isinstance(a_s, spaces.Box) for a_s in self.action_spaces)

        box_actions: Dict[AgentID, spaces.Box] = self.action_spaces  # type: ignore

        if self.flatten:
            self.unflat_space: Dict[AgentID, spaces.Discrete] = {
                i: self.discretize_action_space(
                    act_i, num_actions=num_actions, flatten=False
                )
                for i, act_i in box_actions.items()
            }  # type: ignore

        self._action_spaces = {
            i: self.discretize_action_space(
                act_i, num_actions=num_actions, flatten=flatten
            )
            for i, act_i in box_actions.items()
        }

    def actions(self, actions):
        if self.flatten:
            actions = {
                i: self.undiscretize_discrete_action(act_i, self._unflat_space[i])
                for i, act_i in actions.items()
            }

        return {
            i: self.undiscretize_action(act_i, self.model.action_spaces[i])
            for i, act_i in actions.items()
        }

    def flatten_multidiscrete(
        self, action_space: spaces.MultiDiscrete
    ) -> spaces.Discrete:
        # Compute the product of the individual action space sizes
        n = np.prod(action_space.nvec)
        # Create the new Discrete action space with size n
        new_space = spaces.Discrete(n)
        return new_space

    def discretize_action_space(
        self, action_space: spaces.Box, num_actions: int, flatten=False
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

        # Define the discrete action space
        if flatten:
            return self.flatten_multidiscrete(md_space)  # type: ignore
        else:
            return md_space

    def undiscretize_action(
        self, discrete_action: np.ndarray, action_space: spaces.Box
    ) -> np.ndarray:
        assert (
            len(discrete_action) == action_space.shape[0]
        ), "Discrete action must have the same number of dimensions as the action space"

        # Compute the bin size for each dimension
        bin_sizes = (action_space.high - action_space.low) / (self.num_actions - 1)

        # Convert each discrete action to a continuous value
        continuous_action = action_space.low + np.array(discrete_action) * bin_sizes

        return continuous_action

    def undiscretize_discrete_action(
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
