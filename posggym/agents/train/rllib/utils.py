"""Utility functions, classes, and types for rllib training."""
from typing import Dict, Callable

from ray import rllib
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

import posggym
import posggym.model as M
from posggym.agents.policy import PolicyID
from posggym.wrappers import FlattenObservations
from posggym.wrappers.rllib_multi_agent_env import RllibMultiAgentEnv


RllibAlgorithmMap = Dict[M.AgentID, Dict[PolicyID, Algorithm]]
RllibPolicyMap = Dict[M.AgentID, Dict[PolicyID, rllib.policy.policy.Policy]]
RllibEnvCreatorFn = Callable[[Dict], RllibMultiAgentEnv]


def posggym_registered_env_creator(config):
    """Create a new rllib compatible environment from POSGgym environment.

    Config expects:
    "env_id" - name of the posggym env

    and optionally:
    "render_mode" - environment render_mode
    "flatten_obs" - bool whether to use observation flattening wrapper
                   (default=True)
    any other env kwargs to pass to make

    """
    env_kwargs = {k: v for k, v in config.items() if k not in ("env_id", "flatten_obs")}
    env = posggym.make(config["env_id"], **env_kwargs)
    if config.get("flatten_obs", True):
        env = FlattenObservations(env)
    return RllibMultiAgentEnv(env)


def register_posggym_env(env_id: str):
    """Register posggym env with Ray."""
    register_env(env_id, posggym_registered_env_creator)
