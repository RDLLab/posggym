"""Random generic policies."""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces

from posggym.agents.policy import ObsType, Policy, PolicyID, PolicyState
from posggym.agents.utils import action_distributions
from posggym.utils import seeding


if TYPE_CHECKING:
    import posggym.model as M


class DiscreteFixedDistributionPolicy(Policy[int, ObsType]):
    """A policy that samples from a fixed distribution over discrete actions.

    Defaults to a uniform distribution if no distribution is provided during
    initialization.

    """

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: str,
        policy_id: PolicyID,
        dist: action_distributions.ActionDistribution | None = None,
    ):
        super().__init__(model, agent_id, policy_id)
        self.dist = dist
        self._rng, _ = seeding.np_random()
        self._action_space = self.model.action_spaces[self.agent_id]
        self.pi = self._get_pi(dist, self._rng)

    def _get_pi(
        self, dist: action_distributions.ActionDistribution | None, rng: seeding.RNG
    ) -> action_distributions.ActionDistribution:
        pi = None
        if isinstance(dist, action_distributions.ActionDistribution):
            pi = dist
        elif isinstance(self._action_space, spaces.Discrete):
            pi = action_distributions.DiscreteUniformActionDistribution(
                low=0, high=self._action_space.n - 1, rng=rng
            )
        elif isinstance(self._action_space, spaces.MultiDiscrete):
            pi = action_distributions.DiscreteUniformActionDistribution(
                low=np.zeros_like(self._action_space.nvec),
                high=self._action_space.nvec - 1,
                rng=rng,
            )
        elif isinstance(self._action_space, spaces.Box):
            pi = action_distributions.ContinousUniformActionDistribution(
                low=self._action_space.low, high=self._action_space.high, rng=rng
            )
        else:
            raise ValueError(
                f"Invalid distribution type {type(dist)} for "
                f"{type(self._action_space)} action space"
            )
        return pi

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng, _ = seeding.np_random(seed=seed)
            self.pi = self._get_pi(self.dist, self._rng)

    def get_next_state(
        self,
        obs: ObsType | None,
        state: PolicyState,
    ) -> PolicyState:
        return {}

    def sample_action(self, state: PolicyState) -> int:
        return self.pi.sample()

    def get_pi(self, state: PolicyState) -> action_distributions.ActionDistribution:
        return self.pi

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` no implemented by {self.__class__.__name__} policy"
        )


class RandomPolicy(Policy):
    """The Random policy.

    This policy simply samples actions from `model.action_spaces[agent_id]`.

    """

    def __init__(self, model: M.POSGModel, agent_id: str, policy_id: PolicyID):
        super().__init__(model, agent_id, policy_id)
        self._action_space = model.action_spaces[agent_id]
        self._rng = random.Random()

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._action_space.seed(seed=seed)
            self._rng = random.Random(seed)

    def get_next_state(
        self,
        obs: ObsType | None,
        state: PolicyState,
    ) -> PolicyState:
        return {}

    def sample_action(self, state: PolicyState) -> int:
        return self._action_space.sample()

    def get_pi(self, state: PolicyState) -> action_distributions.ActionDistribution:
        if isinstance(self._action_space, spaces.Discrete):
            dist = {a: 1.0 / self._action_space.n for a in range(self._action_space.n)}
            return action_distributions.DiscreteActionDistribution(dist, self._rng)

        raise NotImplementedError(
            f"`get_pi()` not implemented by {self.__class__.__name__} policy for "
            "non-discrete action spaces."
        )

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` no implemented by {self.__class__.__name__} policy"
        )
