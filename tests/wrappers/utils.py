"""Utility things for posggym wrapper."""
import numpy as np
import posggym


def has_wrapper(wrapped_env: posggym.Env, wrapper_type: type) -> bool:
    """Check whether env has a wrapper of given type."""
    while isinstance(wrapped_env, posggym.Wrapper):
        if isinstance(wrapped_env, wrapper_type):
            return True
        wrapped_env = wrapped_env.env
    return False


def assert_equals(a, b, prefix=None):
    """Assert equality of data structures `a` and `b`.

    Arguments
    ---------
    a:
        first data structure
    b:
        second data structure
    prefix:
        prefix for failed assertion message for types and dicts

    """
    assert type(a) == type(b), f"{prefix}Differing types: {a} and {b}"
    if isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), f"{prefix}Key sets differ: {a} and {b}"
        for k in a:
            v_a = a[k]
            v_b = b[k]
            assert_equals(v_a, v_b, prefix)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, tuple):
        for elem_from_a, elem_from_b in zip(a, b):
            assert_equals(elem_from_a, elem_from_b, prefix)
    else:
        assert a == b
