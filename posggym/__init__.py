"""Root '__init__' of the posggym package."""
# isort: skip_file
# Need to import model and core before other modules
from posggym.model import POSGFullModel, POSGModel
from posggym.core import (
    ActionWrapper, Env, DefaultEnv, ObservationWrapper, RewardWrapper, Wrapper
)
from posggym import envs, error, logger, utils, wrappers
from posggym.envs import make, register, spec


__all__ = [
    # core classes
    "Env",
    "DefaultEnv",
    "Wrapper",
    "ObservationWrapper",
    "ActionWrapper",
    "RewardWrapper",
    "POSGModel",
    "POSGFullModel",
    # registration
    "make",
    "register",
    "spec",
    # module folders
    "envs",
    "utils",
    "wrappers",
    "error",
    "logger",
]
__version__ = "0.1.0"
