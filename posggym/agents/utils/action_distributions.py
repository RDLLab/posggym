"""Action distributions for policies."""
from __future__ import annotations

import abc
import random
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np


if TYPE_CHECKING:
    from posggym.utils import seeding


class ActionDistribution(abc.ABC):
    """Abstract base class for action distributions."""

    @abc.abstractmethod
    def sample(self) -> Any:
        """Sample an action from the distribution."""

    @abc.abstractmethod
    def pdf(self, action: Any) -> float:
        """Get the probability density of an action."""


class DiscreteActionDistribution(ActionDistribution):
    """Action distribution for discrete actions."""

    def __init__(self, probs: Dict[Any, float], rng: seeding.RNG | None = None):
        self.probs = probs
        self._rng = rng

    def sample(self) -> Any:
        if self._rng is None:
            return random.choices(
                list(self.probs.keys()), weights=list(self.probs.values())
            )[0]
        elif isinstance(self._rng, random.Random):
            return self._rng.choices(
                list(self.probs.keys()), weights=list(self.probs.values())
            )[0]
        return self._rng.choice(list(self.probs.keys()), p=list(self.probs.values()))

    def pdf(self, action: Any) -> float:
        return self.probs[action]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiscreteActionDistribution):
            return False
        return self.probs == other.probs


class MultiDiscreteActionDistribution(ActionDistribution):
    """Action distribution for multi-discrete actions."""

    def __init__(self, probs: List[Dict[Any, float]], rng: seeding.RNG | None = None):
        self.probs = probs
        self._rng = rng

    def sample(self) -> Any:
        if self._rng is None:
            return [
                random.choices(list(p.keys()), weights=list(p.values()))[0]
                for p in self.probs
            ]
        elif isinstance(self._rng, random.Random):
            return [
                self._rng.choices(list(p.keys()), weights=list(p.values()))[0]
                for p in self.probs
            ]
        return [
            self._rng.choice(list(p.keys()), p=list(p.values()))[0] for p in self.probs
        ]

    def pdf(self, action: Any) -> float:
        return float(np.prod([p[action[i]] for i, p in enumerate(self.probs)]))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiDiscreteActionDistribution):
            return False
        return self.probs == other.probs


class NormalActionDistribution(ActionDistribution):
    """Action distribution for continuous normally distributed actions."""

    def __init__(
        self,
        mean: Union[float, np.ndarray],
        stddev: Union[float, np.ndarray],
        rng: np.random.Generator | None = None,
    ):
        self.mean = mean
        self.stddev = stddev
        self._rng = rng

    def sample(self) -> Any:
        if self._rng is None:
            return np.random.normal(loc=self.mean, scale=self.stddev)
        return self._rng.normal(loc=self.mean, scale=self.stddev)

    def pdf(self, action: Any) -> float:
        # ref:
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html#numpy-random-normal
        return np.exp(-((action - self.mean) ** 2) / (2 * self.stddev**2)) / (
            self.stddev * np.sqrt(2 * np.pi)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NormalActionDistribution):
            return False
        return (
            np.isclose(self.mean, other.mean).all()
            and np.isclose(self.stddev, other.stddev).all()
        )


class DeterministicActionDistribution(ActionDistribution):
    """Action distribution for deterministic action distribution."""

    def __init__(self, action: Union[int, float, np.ndarray]):
        self.action = action

    def sample(self) -> Any:
        return self.action

    def pdf(self, action: Any) -> float:
        return 1.0 if np.isclose(action, self.action) else 0.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeterministicActionDistribution):
            return False
        return np.isclose(self.action, other.action).all()


class ContinousUniformActionDistribution(ActionDistribution):
    """Action distribution for continuous uniformly distributed actions."""

    def __init__(
        self,
        low: Union[float, np.ndarray],
        high: Union[float, np.ndarray],
        rng: np.random.Generator | None = None,
    ):
        self.low = low
        self.high = high
        self._rng = rng

    def sample(self) -> Any:
        if self._rng is None:
            return np.random.uniform(low=self.low, high=self.high)
        return self._rng.uniform(low=self.low, high=self.high)

    def pdf(self, action: Any) -> float:
        return 1.0 / np.prod(self.high - self.low)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContinousUniformActionDistribution):
            return False
        return (
            np.isclose(self.low, other.low).all()
            and np.isclose(self.high, other.high).all()
        )


class DiscreteUniformActionDistribution(ActionDistribution):
    """Action distribution for discrete uniformly distributed actions.

    Samples uniformly from the set of integers between low and high, inclusive. Works
    for both scalar and vector actions.
    """

    def __init__(
        self,
        low: Union[int, np.ndarray],
        high: Union[int, np.ndarray],
        rng: np.random.Generator | None = None,
    ):
        self.low = low
        self.high = high
        self._rng = rng

    def sample(self) -> Any:
        if self._rng is None:
            return np.random.randint(low=self.low, high=self.high + 1)
        return self._rng.integers(low=self.low, high=self.high + 1)

    def pdf(self, action: Any) -> float:
        return 1.0 / np.prod(self.high - self.low + 1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiscreteUniformActionDistribution):
            return False
        return (
            np.isclose(self.low, other.low).all()
            and np.isclose(self.high, other.high).all()
        )
