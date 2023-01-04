"""Random number generator functions.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/utils/seeding.py

"""
import random
from typing import Optional, Tuple, Union

import numpy as np

from posggym import error


RNG = Union[random.Random, np.random.Generator]


def np_random(seed: Optional[int] = None) -> Tuple[np.random.Generator, int]:
    """Create a numpy random number generator.

    Arguments
    ---------
    seed : the seed used to create the generator.

    Returns
    -------
    rng : the random number generator
    seed : the seed used for the rng (will equal argument seed if one is provided.)

    Raises
    ------
    Error: if seed is not None or a non-negative integer.

    """
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        if isinstance(seed, int) is False:
            raise error.Error(
                f"Seed must be a python integer, actual type: {type(seed)}"
            )
        else:
            raise error.Error(
                f"Seed must be greater or equal to zero, actual value: {seed}"
            )

    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    # np_seed should always be an int if seed is an int | None
    assert isinstance(np_seed, int)
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    return rng, np_seed


def std_random(seed: Optional[int] = None) -> Tuple[random.Random, int]:
    """Create random number generator using python built-in `random.Random`.

    Arguments
    ---------
    seed : the seed used to create the generator.

    Returns
    -------
    rng : the random number generator
    seed : the seed used for the rng (will equal argument seed if one is provided.)

    Raises
    ------
    Error: if seed is not None or a non-negative integer.

    """
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        if isinstance(seed, int) is False:
            raise error.Error(
                f"Seed must be a python integer, actual type: {type(seed)}"
            )
        else:
            raise error.Error(
                f"Seed must be greater or equal to zero, actual value: {seed}"
            )

    if seed is None:
        # We use the np.random library to generate the seed, since currently no well
        # documented/standard way to do this with the builtin random library
        seed_seq = np.random.SeedSequence(None)
        seed = seed_seq.entropy  # type: ignore
        # seed_seq.entropy should always be an int if seed is an int | None
        assert isinstance(seed, int)

    rng = random.Random(seed)
    return rng, seed
