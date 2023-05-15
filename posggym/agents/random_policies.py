"""Random generic policies."""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict, List

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
        agent_id: M.AgentID,
        policy_id: PolicyID,
        dist: Dict[int, float] | None = None,
    ):
        super().__init__(model, agent_id, policy_id)

        action_space = self.model.action_spaces[self.agent_id]
        assert isinstance(action_space, spaces.Discrete)

        self._action_space = list(range(action_space.n))
        if dist is None:
            # use uniform dist
            dist = {a: float(1.0 / action_space.n) for a in self._action_space}

        self.dist = dist
        self._cum_weights: List[float] = []
        for i, a in enumerate(self._action_space):
            prob_sum = 0.0 if i == 0 else self._cum_weights[-1]
            self._cum_weights.append(self.dist[a] + prob_sum)

        self._rng, _ = seeding.std_random()

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng, _ = seeding.std_random(seed=seed)

    def get_next_state(
        self,
        obs: ObsType | None,
        state: PolicyState,
    ) -> PolicyState:
        return {}

    def sample_action(self, state: PolicyState) -> int:
        return self._rng.choices(
            self._action_space, cum_weights=self._cum_weights, k=1
        )[0]

    def get_pi(self, state: PolicyState) -> action_distributions.ActionDistribution:
        return action_distributions.DiscreteActionDistribution(self.dist, self._rng)

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` no implemented by {self.__class__.__name__} policy"
        )


class RandomPolicy(Policy):
    """The Random policy.

    This policy simply samples actions from `model.action_spaces[agent_id]`.

    """

    def __init__(self, model: M.POSGModel, agent_id: M.AgentID, policy_id: PolicyID):
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
