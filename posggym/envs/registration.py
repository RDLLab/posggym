"""Functions and classes for registering and loading implemented environments

Utilizes the Open AI Gym registration functionality.
"""
from gym.envs.registration import EnvRegistry, EnvSpec

from posggym.core import Env

# Global registry that all implemented environments are added too
# Environments are registered when posg library is loaded
registry = EnvRegistry()


def register(env_id: str, **kwargs) -> None:
    """Register an environment in the global registry """
    return registry.register(env_id, **kwargs)


def make(env_id: str, **kwargs) -> Env:
    """Make a new instance of the environment with given ID """
    return registry.make(env_id, **kwargs)


def spec(env_id: str) -> EnvSpec:
    """Get the specification of the environment with given ID """
    return registry.spec(env_id)
