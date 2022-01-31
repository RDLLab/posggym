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

register(
    env_id="MultiAgentTiger-v0",
    entry_point="posggym.envs.full_model.mat:MultiAgentTigerEnv"
)


# Grid World
# -------------------------------------------

# Two Paths
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

# Pursuit-Evasion
register(
    env_id="PursuitEvasion8x8-v0",
    entry_point="posggym.envs.grid_world.pursuit_evasion:PursuitEvasionEnv",
    max_episode_steps=50,
    kwargs={
        "grid_name": "8x8",
        "action_probs": 1.0,
    }
)

register(
    env_id="PursuitEvasion8x8Stochastic-v0",
    entry_point="posggym.envs.grid_world.pursuit_evasion:PursuitEvasionEnv",
    max_episode_steps=50,
    kwargs={
        "grid_name": "8x8",
        "action_probs": 0.9,
    }
)

register(
    env_id="PursuitEvasion16x16-v0",
    entry_point="posggym.envs.grid_world.pursuit_evasion:PursuitEvasionEnv",
    max_episode_steps=100,
    kwargs={
        "grid_name": "16x16",
        "action_probs": 1.0,
    }
)

register(
    env_id="PursuitEvasion16x16Stochastic-v0",
    entry_point="posggym.envs.grid_world.pursuit_evasion:PursuitEvasionEnv",
    max_episode_steps=100,
    kwargs={
        "grid_name": "16x16",
        "action_probs": 0.9,
    }
)

register(
    env_id="PursuitEvasion32x32-v0",
    entry_point="posggym.envs.grid_world.pursuit_evasion:PursuitEvasionEnv",
    max_episode_steps=200,
    kwargs={
        "grid_name": "32x32",
        "action_probs": 1.0,
    }
)

register(
    env_id="PursuitEvasion32x32Stochastic-v0",
    entry_point="posggym.envs.grid_world.pursuit_evasion:PursuitEvasionEnv",
    max_episode_steps=200,
    kwargs={
        "grid_name": "32x32",
        "action_probs": 0.9,
    }
)
