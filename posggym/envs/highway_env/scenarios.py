from typing import Optional, Dict

from posggym.envs.highway_env.core import HWMultiAgentHighwayEnv


def _occupancy_grid_obs_config(n_obs_agents: int) -> Dict:
    """Get config for occupancy grid obs.

    https://highway-env.readthedocs.io/en/latest/observations/index.html#occupancy-grid
    """
    return {
        "type": "OccupancyGrid",
        "vehicles_count": n_obs_agents,
        "features": ["presence"],
        # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        # "features_range": {
        #     "x": [-100, 100],
        #     "y": [-100, 100],
        #     "vx": [-20, 20],
        #     "vy": [-20, 20]
        # },
        "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
        "grid_step": [5, 5],
        "absolute": False
    }


def _kinematics_obs_config(n_obs_agents: int) -> Dict:
    """Get config for kinematic obs.

    https://highway-env.readthedocs.io/en/latest/observations/index.html#kinematics
    """
    return {
        "type": "Kinematics",
        "vehicles_count": n_obs_agents,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted"
    }


def get_multiagent_config(n_agents: int) -> Dict:
    """Get Multiagent configuration for highway env."""
    return {
        "controlled_vehicles": n_agents,
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": _occupancy_grid_obs_config(n_agents)
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction"
            }
        },
        "vehicles_count": 0,
    }


def highway(n_agents: int,
            num_lanes: int = 2,
            seed: Optional[int] = None) -> HWMultiAgentHighwayEnv:
    """Create the highway scenario.

    Ref: https://github.com/eleurent/highway-env#highway
    """
    env = HWMultiAgentHighwayEnv(config=None)
    env.seed(seed)
    multi_agent_config = get_multiagent_config(n_agents)
    multi_agent_config.update({
        "lanes_count": num_lanes,
        "initial_lane_id": None,
        "duration": 40,  # [s]
        "ego_spacing": 2,
        "vehicles_density": 1,
        # The reward received when colliding with a vehicle.
        "collision_reward": -1,
        # The reward received when driving on the right-most lanes,
        # linearly mapped to zero for other lanes.
        "right_lane_reward": 0.1,
        # The reward received when driving at full speed, linearly mapped to
        # zero for lower speeds according to config["reward_speed_range"].
        "high_speed_reward": 0.4,
        # The reward received at each lane change action.
        "lane_change_reward": 0,
        "reward_speed_range": [20, 30],
        "normalize_reward": True,
        "offroad_terminal": False
    })
    env.configure(multi_agent_config)
    env.reset()
    return env


HWSCENARIOS = {
    "highway": highway
}
