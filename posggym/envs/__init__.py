"""Initializes posg implemented environments.

Utilizes the OpenAI Gym env registration functionality.
"""
from itertools import product

from posggym.envs.registration import register
from posggym.envs.registration import make
from posggym.envs.registration import spec
from posggym.envs.registration import registry

from posggym.envs.grid_world import pursuit_evasion
from posggym.envs.grid_world import two_paths
from posggym.envs.grid_world import uav
from posggym.envs.grid_world import driving


# Full Model
# -------------------------------------------

register(
    env_id="MABC-v0",
    entry_point="posggym.envs.classic.mabc:MABCEnv"
)

register(
    env_id="MultiAgentTiger-v0",
    entry_point="posggym.envs.classic.mat:MultiAgentTigerEnv"
)

register(
    env_id="RockPaperScissors-v0",
    entry_point="posggym.envs.classic.rps:RockPaperScissorsEnv"
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


# Driving
for grid_name in driving.grid.SUPPORTED_GRIDS:
    grid_fn, finite_steps, inf_steps = driving.grid.SUPPORTED_GRIDS[grid_name]
    for n in range(2, grid_fn().supported_num_agents+1):
        env_name_prefix = f"Driving{grid_name}-n{n}"
        register(
            env_id=f"{env_name_prefix}-v0",
            entry_point="posggym.envs.grid_world.driving:DrivingEnv",
            max_episode_steps=finite_steps,
            kwargs={
                "grid": grid_fn(),
                "num_agents": 2,
                "obs_dim": (3, 1, 1),
                "obstacle_collisions": False,
                "infinite_horizon": False
            }
        )

        register(
            env_id=f"{env_name_prefix}-v1",
            entry_point="posggym.envs.grid_world.driving:DrivingEnv",
            max_episode_steps=inf_steps,
            kwargs={
                "grid": grid_fn(),
                "num_agents": 2,
                "obs_dim": (3, 1, 1),
                "obstacle_collisions": False,
                "infinite_horizon": True
            }
        )

        register(
            env_id=f"{env_name_prefix}-v2",
            entry_point="posggym.envs.grid_world.driving:DrivingEnv",
            max_episode_steps=finite_steps,
            kwargs={
                "grid": grid_fn(),
                "num_agents": 2,
                "obs_dim": (3, 1, 1),
                "obstacle_collisions": True,
                "infinite_horizon": False
            }
        )

        register(
            env_id=f"{env_name_prefix}-v3",
            entry_point="posggym.envs.grid_world.driving:DrivingEnv",
            max_episode_steps=inf_steps,
            kwargs={
                "grid": grid_fn(),
                "num_agents": 2,
                "obs_dim": (3, 1, 1),
                "obstacle_collisions": True,
                "infinite_horizon": True
            }
        )


# Driving Gen
for grid_name in driving.gen.SUPPORTED_GEN_PARAMS:
    env_params = driving.gen.SUPPORTED_GEN_PARAMS[grid_name]
    gen_params, finite_steps, inf_steps = env_params   # type: ignore
    register(
        env_id=f"DrivingGen{grid_name}-v0",
        entry_point="posggym.envs.grid_world.driving:DrivingGenEnv",
        max_episode_steps=finite_steps,
        kwargs={
            "num_agents": 2,
            "obs_dim": (3, 1, 1),
            "obstacle_collisions": False,
            "n_grids": None,
            "generator_params": gen_params,
            "infinite_horizon": False
        }
    )

    register(
        env_id=f"DrivingGen{grid_name}-v1",
        entry_point="posggym.envs.grid_world.driving:DrivingGenEnv",
        max_episode_steps=inf_steps,
        kwargs={
            "num_agents": 2,
            "obs_dim": (3, 1, 1),
            "obstacle_collisions": False,
            "n_grids": None,
            "generator_params": gen_params,
            "infinite_horizon": True
        }
    )

    register(
        env_id=f"DrivingGen{grid_name}-v2",
        entry_point="posggym.envs.grid_world.driving:DrivingGenEnv",
        max_episode_steps=finite_steps,
        kwargs={
            "num_agents": 2,
            "obs_dim": (3, 1, 1),
            "obstacle_collisions": True,
            "n_grids": None,
            "generator_params": gen_params,
            "infinite_horizon": False
        }
    )

    register(
        env_id=f"DrivingGen{grid_name}-v3",
        entry_point="posggym.envs.grid_world.driving:DrivingGenEnv",
        max_episode_steps=inf_steps,
        kwargs={
            "num_agents": 2,
            "obs_dim": (3, 1, 1),
            "obstacle_collisions": True,
            "n_grids": None,
            "generator_params": gen_params,
            "infinite_horizon": True
        }
    )


# Level-Based Foraging
# -------------------------------------------
# Ref: github.com/semitable/lb-foraging/blob/master/lbforaging/__init__.py

sizes = range(5, 10, 5)
players = range(2, 8, 2)   # reduced
foods = range(1, 5)
coop = [True, False]

for s, p, f, c in product(sizes, players, foods, coop):
    coop_str = "-coop" if c else ""
    register(
        env_id=f"LBF{s}x{s}-{p}p-{f}f{coop_str}-v2",
        entry_point="posggym.envs.lbf:LBFEnv",
        kwargs={
            "num_agents": p,
            "max_agent_level": 3,
            "field_size": (s, s),
            "max_food": f,
            "sight": 2,
            "max_episode_steps": 50,
            "force_coop": c,
            "normalize_reward": True,
            "grid_observation": False,
            "penalty": 0.0
        },
    )

    register(
        env_id=f"LBFGrid{s}x{s}-{p}p-{f}f{coop_str}-v2",
        entry_point="posggym.envs.lbf:LBFEnv",
        kwargs={
            "num_agents": p,
            "max_agent_level": 3,
            "field_size": (s, s),
            "max_food": f,
            "sight": 2,
            "max_episode_steps": 50,
            "force_coop": c,
            "normalize_reward": True,
            "grid_observation": True,
            "penalty": 0.0
        },
    )
