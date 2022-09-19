"""Initializes posg implemented environments.

Utilizes the OpenAI Gym env registration functionality.
"""
import math
import warnings
from itertools import product

from posggym.envs.registration import register
from posggym.envs.registration import make       # noqa
from posggym.envs.registration import spec       # noqa
from posggym.envs.registration import registry   # noqa

from posggym.envs.grid_world import pursuit_evasion
from posggym.envs.grid_world import two_paths
from posggym.envs.grid_world import uav
from posggym.envs.grid_world import driving
from posggym.envs.highway_env.scenarios import HWSCENARIOS


# Full Model
# -------------------------------------------

register(
    id="MABC-v0",
    entry_point="posggym.envs.classic.mabc:MABCEnv"
)

register(
    id="MultiAgentTiger-v0",
    entry_point="posggym.envs.classic.mat:MultiAgentTigerEnv"
)

register(
    id="RockPaperScissors-v0",
    entry_point="posggym.envs.classic.rps:RockPaperScissorsEnv"
)


# Grid World
# -------------------------------------------

# Two Paths
for grid_name in two_paths.grid.SUPPORTED_GRIDS:
    register(
        id=f"TwoPaths{grid_name}-v0",
        entry_point="posggym.envs.grid_world.two_paths:TwoPathsEnv",
        max_episode_steps=two_paths.grid.SUPPORTED_GRIDS[grid_name][1],
        kwargs={
            "grid_name": grid_name,
            "action_probs": 1.0,
            "infinite_horizon": False
        }
    )

    register(
        id=f"TwoPaths{grid_name}Stochastic-v0",
        entry_point="posggym.envs.grid_world.two_paths:TwoPathsEnv",
        max_episode_steps=two_paths.grid.SUPPORTED_GRIDS[grid_name][1],
        kwargs={
            "grid_name": grid_name,
            "action_probs": 0.9,
            "infinite_horizon": False
        }
    )

    register(
        id=f"TwoPaths{grid_name}-v1",
        entry_point="posggym.envs.grid_world.two_paths:TwoPathsEnv",
        max_episode_steps=two_paths.grid.SUPPORTED_GRIDS[grid_name][2],
        kwargs={
            "grid_name": grid_name,
            "action_probs": 1.0,
            "infinite_horizon": True
        }
    )

    register(
        id=f"TwoPaths{grid_name}Stochastic-v1",
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
        id=f"PursuitEvasion{grid_name}-v0",
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
        id=f"PursuitEvasion{grid_name}Stochastic-v0",
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
        id=f"UAV{grid_name}-v0",
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
            id=f"{env_name_prefix}-v0",
            entry_point="posggym.envs.grid_world.driving:DrivingEnv",
            max_episode_steps=finite_steps,
            kwargs={
                "grid": grid_fn(),
                "num_agents": n,
                "obs_dim": (3, 1, 1),
                "obstacle_collisions": False,
                "infinite_horizon": False
            }
        )

        register(
            id=f"{env_name_prefix}-v1",
            entry_point="posggym.envs.grid_world.driving:DrivingEnv",
            max_episode_steps=inf_steps,
            kwargs={
                "grid": grid_fn(),
                "num_agents": n,
                "obs_dim": (3, 1, 1),
                "obstacle_collisions": False,
                "infinite_horizon": True
            }
        )

        register(
            id=f"{env_name_prefix}-v2",
            entry_point="posggym.envs.grid_world.driving:DrivingEnv",
            max_episode_steps=finite_steps,
            kwargs={
                "grid": grid_fn(),
                "num_agents": n,
                "obs_dim": (3, 1, 1),
                "obstacle_collisions": True,
                "infinite_horizon": False
            }
        )

        register(
            id=f"{env_name_prefix}-v3",
            entry_point="posggym.envs.grid_world.driving:DrivingEnv",
            max_episode_steps=inf_steps,
            kwargs={
                "grid": grid_fn(),
                "num_agents": n,
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
        id=f"DrivingGen{grid_name}-v0",
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


# Level-Based Foraging
# -------------------------------------------
# Ref: github.com/semitable/lb-foraging/blob/master/lbforaging/__init__.py

sizes = [5, 10, 15, 20]
players = [2, 3, 4, 6, 8]   # reduced
foods = [1, 3, 5, 7, 10]
coop = [True, False]
static_layout = [True, False]

for s, n, f, c, sl in product(sizes, players, foods, coop, static_layout):
    if sl and f > math.floor((s-1)/2)**2 or n > math.ceil(s/2)*4 - 4:
        continue

    coop_str = "-coop" if c else ""
    static_layout_str = "-static" if sl else ""
    suffix = f"{s}x{s}-n{n}-f{f}{coop_str}{static_layout_str}-v2"
    register(
        id=f"LBF{suffix}",
        entry_point="posggym.envs.lbf:LBFEnv",
        max_episode_steps=50,
        kwargs={
            "num_agents": n,
            "max_agent_level": 3,
            "field_size": (s, s),
            "max_food": f,
            "sight": 2,
            "max_episode_steps": 50,
            "force_coop": c,
            "static_layout": sl,
            "normalize_reward": True,
            "observation_mode": 'tuple',
            "penalty": 0.0
        },
    )

    register(
        id=f"LBFVector{suffix}",
        entry_point="posggym.envs.lbf:LBFEnv",
        max_episode_steps=50,
        kwargs={
            "num_agents": n,
            "max_agent_level": 3,
            "field_size": (s, s),
            "max_food": f,
            "sight": 2,
            "max_episode_steps": 50,
            "force_coop": c,
            "static_layout": sl,
            "normalize_reward": True,
            "observation_mode": 'vector',
            "penalty": 0.0
        },
    )

    register(
        id=f"LBFGrid{suffix}",
        entry_point="posggym.envs.lbf:LBFEnv",
        max_episode_steps=50,
        kwargs={
            "num_agents": n,
            "max_agent_level": 3,
            "field_size": (s, s),
            "max_food": f,
            "sight": 2,
            "max_episode_steps": 50,
            "force_coop": c,
            "static_layout": sl,
            "normalize_reward": True,
            "observation_mode": 'grid',
            "penalty": 0.0
        },
    )


# Highway Env
# -------------------------------------------
# Ref: https://github.com/eleurent/highway-env

# suppress pandas warning to do with Highway Env obs space
warnings.simplefilter(action='ignore', category=FutureWarning)

num_agents = [2, 3, 4, 6, 8]

for scenario_name, scenario_fn in HWSCENARIOS.items():
    for n in num_agents:
        register(
            id=f"HW{scenario_name}-n{n}-v0",
            entry_point="posggym.envs.highway_env:HWEnv",
            kwargs={
                "n_agents": n,
                "env": scenario_fn(n)
            },
        )
