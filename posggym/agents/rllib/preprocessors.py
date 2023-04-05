"""Preprocessor for Rllib policies."""
from typing import Any, Callable

from gymnasium import spaces


ObsPreprocessor = Callable[[Any], Any]


def identity_preprocessor(obs: Any) -> Any:
    """Return the observation unchanged."""
    return obs


def get_flatten_preprocessor(obs_space: spaces.Space) -> ObsPreprocessor:
    """Get the preprocessor function for flattening observations."""

    def flatten_preprocessor(obs: Any) -> Any:
        """Flatten the observation."""
        return spaces.flatten(obs_space, obs)

    return flatten_preprocessor
