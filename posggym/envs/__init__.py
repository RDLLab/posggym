"""Registers the internal POSGGym environments.

Based on the Farama Foundation Gymnasium API:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/envs/__init__.py

"""
import math

from posggym.envs.registration import make, pprint_registry, register, registry, spec

# Classic
# -------------------------------------------

register(
    id="MultiAccessBroadcastChannel-v0", entry_point="posggym.envs.classic.mabc:MABCEnv"
)

register(
    id="MultiAgentTiger-v0", entry_point="posggym.envs.classic.tiger:MultiAgentTigerEnv"
)

register(
    id="RockPaperScissors-v0",
    entry_point="posggym.envs.classic.rock_paper_scissors:RockPaperScissorsEnv",
)

# Continuous
# -------------------------------------------
register(
    id="DrivingContinuous-v0",
    entry_point="posggym.envs.continuous.driving_continuous:DrivingContinuousEnv",
    max_episode_steps=100,
    kwargs={
        "world": "14x14RoundAbout",
        "num_agents": 2,
        "obs_dist": 5.0,
        "n_sensors": 16,
    },
)

register(
    id="DroneTeamCapture-v0",
    entry_point="posggym.envs.continuous.drone_team_capture:DroneTeamCaptureEnv",
    max_episode_steps=500,
    kwargs={
        "num_agents": 3,
        "n_communicating_pursuers": None,
        "arena_radius": 430,
        "observation_limit": None,
        "velocity_control": False,
        "capture_radius": 30,
        "use_q_reward": False,
    },
)

register(
    id="PredatorPreyContinuous-v0",
    entry_point="posggym.envs.continuous.predator_prey_continuous:PredatorPreyContinuousEnv",
    max_episode_steps=100,
    kwargs={
        "world": "10x10",
        "num_predators": 2,
        "num_prey": 3,
        "cooperative": True,
        "prey_strength": None,
        "obs_dist": 4,
        "n_sensors": 16,
    },
)

register(
    id="PursuitEvasionContinuous-v0",
    entry_point="posggym.envs.continuous.pursuit_evasion_continuous:PursuitEvasionContinuousEnv",
    max_episode_steps=100,
    kwargs={
        "world": "16x16",
        "max_obs_distance": None,
        "fov": math.pi / 3,
        "n_sensors": 16,
        "normalize_reward": True,
        "use_progress_reward": True,
    },
)

# Grid World
# -------------------------------------------

# Cooperative Reaching
register(
    id="CooperativeReaching-v0",
    entry_point="posggym.envs.grid_world.cooperative_reaching:CooperativeReachingEnv",
    max_episode_steps=50,
    kwargs={
        "size": 5,
        "num_goals": 4,
        "mode": "original",
        "obs_distance": None,
    },
)


# Driving
register(
    id="Driving-v1",
    entry_point="posggym.envs.grid_world.driving:DrivingEnv",
    max_episode_steps=50,
    kwargs={
        "grid": "14x14RoundAbout",
        "num_agents": 2,
        "obs_dim": (3, 1, 1),
    },
)


# Driving Gen
register(
    id="DrivingGen-v1",
    entry_point="posggym.envs.grid_world.driving_gen:DrivingGenEnv",
    max_episode_steps=50,
    kwargs={
        "num_agents": 2,
        "obs_dim": (3, 1, 1),
        "generator_params": "14x14",
        "n_grids": None,
        "shuffle_grid_order": True,
    },
)

# Level-Based Foraging
# Based on github.com/semitable/lb-foraging/
register(
    id="LevelBasedForaging-v3",
    entry_point="posggym.envs.grid_world.level_based_foraging:LevelBasedForagingEnv",
    max_episode_steps=50,
    kwargs={
        "num_agents": 2,
        "max_agent_level": 3,
        "size": 10,
        "max_food": 8,
        "sight": 2,
        "force_coop": False,
        "static_layout": False,
        "observation_mode": "tuple",
    },
)


# Predator-Prey
register(
    id="PredatorPrey-v0",
    entry_point="posggym.envs.grid_world.predator_prey:PredatorPreyEnv",
    max_episode_steps=50,
    kwargs={
        "grid": "10x10",
        "num_predators": 2,
        "num_prey": 3,
        "cooperative": True,
        "prey_strength": None,
        "obs_dim": 2,
    },
)


# Pursuit-Evasion
register(
    id="PursuitEvasion-v1",
    entry_point="posggym.envs.grid_world.pursuit_evasion:PursuitEvasionEnv",
    max_episode_steps=100,
    kwargs={
        "grid": "16x16",
        "max_obs_distance": 12,
        "use_progress_reward": True,
    },
)


# Two Paths
register(
    id="TwoPaths-v0",
    entry_point="posggym.envs.grid_world.two_paths:TwoPathsEnv",
    max_episode_steps=20,
    kwargs={"grid_size": 7, "action_probs": 1.0},
)


# Unmanned Aerial Vehicle (UAV)
register(
    id="UAV-v0",
    entry_point="posggym.envs.grid_world.uav:UAVEnv",
    max_episode_steps=50,
    kwargs={"grid": 5},
)
