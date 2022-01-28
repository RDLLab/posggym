"""Initializes posg implemented environments.

Utilizes the OpenAI Gym env registration functionality.
"""
from posggym.envs.registration import register
from posggym.envs.registration import make
from posggym.envs.registration import spec
from posggym.envs.registration import registry

# Full Model
# -------------------------------------------

register(
    env_id="MABC-v0",
    entry_point="posggym.envs.full_model.mabc:MABCEnv"
)


# Grid World
# -------------------------------------------

register(
    env_id="TwoPaths3x3-v0",
    entry_point="posggym.envs.grid_world.two_paths:TwoPathsEnv",
    max_episode_steps=20,
    kwargs={
        "grid_name": "3x3",
        "action_probs": 1.0,
        "infinite_horizon": False
    }
)

register(
    env_id="TwoPaths3x3Stochastic-v0",
    entry_point="posggym.envs.grid_world.two_paths:TwoPathsEnv",
    max_episode_steps=20,
    kwargs={
        "grid_name": "3x3",
        "action_probs": 0.9,
        "infinite_horizon": False
    }
)

register(
    env_id="TwoPaths4x4-v0",
    entry_point="posggym.envs.grid_world.two_paths:TwoPathsEnv",
    max_episode_steps=20,
    kwargs={
        "grid_name": "4x4",
        "action_probs": 1.0,
        "infinite_horizon": False
    }
)

register(
    env_id="TwoPaths4x4Stochastic-v0",
    entry_point="posggym.envs.grid_world.two_paths:TwoPathsEnv",
    max_episode_steps=20,
    kwargs={
        "grid_name": "4x4",
        "action_probs": 0.9,
        "infinite_horizon": False
    }
)

register(
    env_id="TwoPaths7x7-v0",
    entry_point="posggym.envs.grid_world.two_paths:TwoPathsEnv",
    max_episode_steps=20,
    kwargs={
        "grid_name": "7x7",
        "action_probs": 1.0,
        "infinite_horizon": False
    }
)

register(
    env_id="TwoPaths7x7Stochastic-v0",
    entry_point="posggym.envs.grid_world.two_paths:TwoPathsEnv",
    max_episode_steps=20,
    kwargs={
        "grid_name": "7x7",
        "action_probs": 0.9,
        "infinite_horizon": False
    }
)
