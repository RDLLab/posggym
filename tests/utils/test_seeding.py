"""Tests for random number generator utilities.

Adapted from:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/utils/test_seeding.py

"""
import pickle
from typing import get_args

from posggym import error
from posggym.utils import seeding


def test_invalid_seeds():
    for seed in [-1, "test"]:
        try:
            seeding.np_random(seed)
        except error.Error:
            pass
        else:
            assert False, f"Invalid seed {seed} passed validation for `np_random`"

        try:
            seeding.std_random(seed)
        except error.Error:
            pass
        else:
            assert False, f"Invalid seed {seed} passed validation for `std_random`"


def test_valid_seeds():
    for seed in [0, 1]:
        np_rng, seed1 = seeding.np_random(seed)
        assert seed == seed1
        std_rng, seed2 = seeding.std_random(seed)
        assert seed == seed2


def test_rng_pickle():
    np_rng, _ = seeding.np_random(seed=0)
    pickled = pickle.dumps(np_rng)
    np_rng2 = pickle.loads(pickled)
    assert isinstance(
        np_rng2, get_args(seeding.RNG)
    ), "Unpickled object is not a seeding.RNG"
    assert np_rng.random() == np_rng2.random()

    std_rng, _ = seeding.std_random(seed=0)
    pickled = pickle.dumps(std_rng)
    std_rng2 = pickle.loads(pickled)
    assert isinstance(
        std_rng2, get_args(seeding.RNG)
    ), "Unpickled object is not a seeding.RNG"
    assert std_rng.random() == std_rng2.random()
