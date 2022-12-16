"""Root '__init__' of the posggym package."""

from posggym import envs, error, logger, utils, wrappers
from posggym.core import ActionWrapper, Env, ObservationWrapper, RewardWrapper, Wrapper
from posggym.envs import make, register, spec
from posggym.model import POSGFullModel, POSGModel


__all__ = [
    # core classes
    "Env",
    "Wrapper",
    "ObservationWrapper",
    "ActionWrapper",
    "RewardWrapper",
    "POSGModel",
    "POSGFullModel",
    # registration
    "make",
    "spec",
    "register",
    # module folders
    "envs",
    "utils",
    "wrappers",
    "error",
    "logger",
]
__version__ = "0.0.1"
