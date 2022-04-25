"""Initializes posg implemented environments.

Utilizes the OpenAI Gym env registration functionality.
"""
from posggym.envs.registration import register
from posggym.envs.registration import make
from posggym.envs.registration import spec
from posggym.envs.registration import registry

from posggym.envs.grid_world import pursuit_evasion
from posggym.envs.grid_world import two_paths
from posggym.envs.grid_world import uav


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
for grid_name in two_paths.grid.SUPPORTED_GRIDS:
    register(
        env_id=f"TwoPaths{grid_name}-v0",
        entry_point="posggym.envs.grid_world.two_paths:TwoPathsEnv",
        max_episode_steps=two_paths.grid.SUPPORTED_GRIDS[grid_name][1],
        kwargs={
            "grid_name": grid_name,
            "action_probs": 1.0,
            "infinite_horizon": False
        }
    )

    register(
        env_id=f"TwoPaths{grid_name}Stochastic-v0",
        entry_point="posggym.envs.grid_world.two_paths:TwoPathsEnv",
        max_episode_steps=two_paths.grid.SUPPORTED_GRIDS[grid_name][1],
        kwargs={
            "grid_name": grid_name,
            "action_probs": 0.9,
            "infinite_horizon": False
        }
    )

    register(
        env_id=f"TwoPaths{grid_name}-v1",
        entry_point="posggym.envs.grid_world.two_paths:TwoPathsEnv",
        max_episode_steps=two_paths.grid.SUPPORTED_GRIDS[grid_name][2],
        kwargs={
            "grid_name": grid_name,
            "action_probs": 1.0,
            "infinite_horizon": True
        }
    )

    register(
        env_id=f"TwoPaths{grid_name}Stochastic-v1",
        entry_point="posggym.envs.grid_world.two_paths:TwoPathsEnv",
        max_episode_steps=two_paths.grid.SUPPORTED_GRIDS[grid_name][2],
        kwargs={
            "grid_name": grid_name,
            "action_probs": 0.9,
            "infinite_horizon": True
        }
    )


# Pursuit-Evasion
for grid_name in pursuit_evasion.grid.SUPPORTED_GRIDS:
    register(
        env_id=f"PursuitEvasion{grid_name}-v0",
        entry_point=(
            "posggym.envs.grid_world.pursuit_evasion:PursuitEvasionEnv"
        ),
        max_episode_steps=pursuit_evasion.grid.SUPPORTED_GRIDS[grid_name][1],
        kwargs={
            "grid_name": grid_name,
            "action_probs": 1.0,
        }
    )

    register(
        env_id=f"PursuitEvasion{grid_name}Stochastic-v0",
        entry_point=(
            "posggym.envs.grid_world.pursuit_evasion:PursuitEvasionEnv"
        ),
        max_episode_steps=pursuit_evasion.grid.SUPPORTED_GRIDS[grid_name][1],
        kwargs={
            "grid_name": grid_name,
            "action_probs": 0.9,
        }
    )


# Unmanned Aerial Vehicle (UAV)
for grid_name in uav.grid.SUPPORTED_GRIDS:
    register(
        env_id=f"UAV{grid_name}-v0",
        entry_point="posggym.envs.grid_world.uav:UAVEnv",
        kwargs={
            "grid_name": grid_name,
        }
    )
