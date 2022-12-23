"""Utility things for posggym wrapper."""
import posggym


def has_wrapper(wrapped_env: posggym.Env, wrapper_type: type) -> bool:
    """Check whether env has a wrapper of given type."""
    while isinstance(wrapped_env, posggym.Wrapper):
        if isinstance(wrapped_env, wrapper_type):
            return True
        wrapped_env = wrapped_env.env
    return False
