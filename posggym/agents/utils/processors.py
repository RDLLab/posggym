"""Input processors."""
import abc
from typing import Any

import numpy as np
from gymnasium import spaces


class Processor(abc.ABC):
    """Abstract class for processing inputs.

    Transforms input and input space from one form into another. Inputs can be, for
    example observations or actions.
    """

    def __init__(self, input_space: spaces.Space):
        self.input_space = input_space

    @abc.abstractmethod
    def __call__(self, input: Any) -> Any:
        """Process the input."""

    @abc.abstractmethod
    def unprocess(self, processed_input: Any) -> Any:
        """Unprocess processed input."""

    @abc.abstractmethod
    def get_processed_space(self) -> spaces.Space:
        """Get the processed space."""


class IdentityProcessor(Processor):
    """Identity processor.

    Leaves the input unchanged.
    """

    def __call__(self, input: Any) -> Any:
        return input

    def unprocess(self, processed_input: Any) -> Any:
        return processed_input

    def get_processed_space(self) -> spaces.Space:
        return self.input_space


class FlattenProcessor(Processor):
    """Processor for flattening inputs."""

    def __call__(self, input: Any) -> Any:
        return spaces.flatten(self.input_space, input)

    def unprocess(self, processed_input: Any) -> Any:
        return spaces.unflatten(self.input_space, processed_input)

    def get_processed_space(self) -> spaces.Space:
        return spaces.flatten_space(self.input_space)


class RescaleProcessor(Processor):
    """Rescales input into given range."""

    def __init__(
        self,
        input_space: spaces.Space,
        min_val: float = -1.0,
        max_val: float = 1.0,
        clip: bool = True,
    ):
        assert isinstance(input_space, spaces.Box)
        super().__init__(input_space)
        self.min_val = min_val
        self.max_val = max_val
        self.clip = clip
        self.rescale_factor = (self.max_val - self.min_val) / (
            self.input_space.high - self.input_space.low
        )

    def __call__(self, input: Any) -> Any:
        if self.clip:
            input = np.clip(input, self.input_space.low, self.input_space.high)
        low = self.input_space.low
        return (input - low) * self.rescale_factor + self.min_val

    def unprocess(self, processed_input: Any) -> Any:
        unprocessed_input = (
            processed_input - self.min_val
        ) / self.rescale_factor + self.input_space.low
        if self.clip:
            unprocessed_input = np.clip(
                unprocessed_input, self.input_space.low, self.input_space.high
            )
        return np.array(unprocessed_input, dtype=self.input_space.dtype)

    def get_processed_space(self) -> spaces.Space:
        return spaces.Box(
            low=self.min_val,
            high=self.max_val,
            shape=self.input_space.shape,
            dtype=self.input_space.dtype,
        )
