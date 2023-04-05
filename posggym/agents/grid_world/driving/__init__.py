"""Policies for the Driving-v0 environment."""
import os.path as osp

from posggym.agents.rllib import load_rllib_policy_specs_from_files


ENV_ID = "Driving-v0"
base_dir = osp.dirname(osp.abspath(__file__))
policy_specs = {}

# 7x7RoundAbout-n2-v0
policy_specs.update(
    load_rllib_policy_specs_from_files(
        env_id=ENV_ID,
        env_args={
            "grid": "7x7RoundAbout",
            "num_agents": 2,
            "obs_dim": (3, 1, 1),
            "obstacle_collisions": False,
        },
        policy_file_dir_path=osp.join(base_dir, "driving_7x7roundabout_n2_v0"),
        policy_file_names=[
            "klr_k0_seed0.pkl",
            "klr_k0_seed1.pkl",
            "klr_k0_seed2.pkl",
            "klr_k0_seed3.pkl",
            "klr_k0_seed4.pkl",
            "klr_k1_seed0.pkl",
            "klr_k1_seed1.pkl",
            "klr_k1_seed2.pkl",
            "klr_k1_seed3.pkl",
            "klr_k1_seed4.pkl",
            "klr_k2_seed0.pkl",
            "klr_k2_seed1.pkl",
            "klr_k2_seed2.pkl",
            "klr_k2_seed3.pkl",
            "klr_k2_seed4.pkl",
            "klr_k3_seed0.pkl",
            "klr_k3_seed1.pkl",
            "klr_k3_seed2.pkl",
            "klr_k3_seed3.pkl",
            "klr_k3_seed4.pkl",
            "klr_k4_seed0.pkl",
            "klr_k4_seed1.pkl",
            "klr_k4_seed2.pkl",
            "klr_k4_seed3.pkl",
            "klr_k4_seed4.pkl",
            "klrbr_k4_seed0.pkl",
            "klrbr_k4_seed1.pkl",
            "klrbr_k4_seed2.pkl",
            "klrbr_k4_seed3.pkl",
            "klrbr_k4_seed4.pkl",
        ],
        version=0,
        valid_agent_ids=None,
        nondeterministic=True,
    )
)


# 14x14RoundAbout-n2-v0
policy_specs.update(
    load_rllib_policy_specs_from_files(
        env_id=ENV_ID,
        env_args={
            "grid": "14x14RoundAbout",
            "num_agents": 2,
            "obs_dim": (3, 1, 1),
            "obstacle_collisions": False,
        },
        policy_file_dir_path=osp.join(base_dir, "driving_14x14roundabout_n2_v0"),
        policy_file_names=[
            "klr_k0_seed0.pkl",
            "klr_k0_seed1.pkl",
            "klr_k0_seed2.pkl",
            "klr_k0_seed3.pkl",
            "klr_k0_seed4.pkl",
            "klr_k1_seed0.pkl",
            "klr_k1_seed1.pkl",
            "klr_k1_seed2.pkl",
            "klr_k1_seed3.pkl",
            "klr_k1_seed4.pkl",
            "klr_k2_seed0.pkl",
            "klr_k2_seed1.pkl",
            "klr_k2_seed2.pkl",
            "klr_k2_seed3.pkl",
            "klr_k2_seed4.pkl",
            "klr_k3_seed0.pkl",
            "klr_k3_seed1.pkl",
            "klr_k3_seed2.pkl",
            "klr_k3_seed3.pkl",
            "klr_k3_seed4.pkl",
            "klr_k4_seed0.pkl",
            "klr_k4_seed1.pkl",
            "klr_k4_seed2.pkl",
            "klr_k4_seed3.pkl",
            "klr_k4_seed4.pkl",
            "klrbr_k4_seed0.pkl",
            "klrbr_k4_seed1.pkl",
            "klrbr_k4_seed2.pkl",
            "klrbr_k4_seed3.pkl",
            "klrbr_k4_seed4.pkl",
        ],
        version=0,
        valid_agent_ids=None,
        nondeterministic=True,
    )
)
